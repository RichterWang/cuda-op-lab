# stable_softmax

行方向（对最后一维归约）的数值稳定 softmax。

```
m     = max(x_row)
y_row = exp(x_row - m) / sum(exp(x_row - m))
```

减去行最大值是"稳定"的来源：输入带大正偏置时 `exp` 不会上溢成 `inf`，带大负偏置时分母不会下溢成 0。另外由于最大元素贡献 `exp(0) == 1`，分母恒 `>= 1`，取倒数不需要 epsilon 保护。

## 目录结构

```
stable_softmax.h                  公开的 launcher 声明
kernels/v0_naive.cu               baseline：一个 block 一行，shared memory 树形归约，三趟读
kernels/v1_warp_reduce.cu         同结构，归约换成 __shfl_xor_sync；kernel 按 block size 模板化，
                                  含 256 / 1024 / 自适应三个 launcher
kernels/v2_subwarp.cu             一个 lane 组一行 + float4，零 barrier；含 v2a / v2b 两个 launcher
kernels/copy_baseline.cu          非 softmax：y = x，用于测量真实可达带宽
kernels/traffic_probe.cu          非 softmax：1趟读 / 3趟读探针，用于直接测量真实 DRAM traffic
benchmark/bench_stable_softmax.cu 正确性 + 带宽 benchmark 入口
benchmark/measure_bottleneck.cu   诊断工具：验证 traffic 与 occupancy 推断
CMakeLists.txt                    target 定义
```

## 为什么用带宽而不是 GFLOPS

softmax 是 memory-bound 算子。每个元素的计算量只有一次 `exp` 加一次乘法，瓶颈在访存，所以主指标是**有效带宽 GB/s**。

理论最小访存量是"读一次 + 写一次"，即 `rows * cols * 4 * 2` 字节。benchmark 里的 GB/s 都按这个最小值计算，因此它衡量的是"离理想访存有多远"，而不是实际搬了多少字节。

对标对象不用 cuBLAS，因为 cuBLAS 不提供 softmax，没有现成的工业基准可比。

## 正确性校验

CPU double 精度参考实现 + 三种输入模式：

| pattern | 构造 | 检验什么 |
|---|---|---|
| `normal` | `U(-1, 1)` | 常规精度 |
| `large_shift` | 整行 `+1e4` | 不减 max 的实现会 `exp` 上溢成 `inf` |
| `extreme_negative` | 整行 `-1e4` | 不减 max 的实现会全部下溢成 0，分母变 0 |

除了 max abs / max rel error，还检查 softmax 特有的守恒性 `max|sum(y_row) - 1.0|`。这个指标对归约里的 bug 特别敏感——少加或多加一个元素，abs error 可能仍然很小，但 row sum 会立刻偏离 1。

判定门槛用绝对误差 `< 1e-6`（输出量级在 (0,1)，绝对误差比相对误差更有意义）加 row sum 误差 `< 1e-4`；相对误差只打印不判定，因为输出值极小时它会虚高。benchmark 任一项超标就返回非零，所以它同时挂在 ctest 下当回归测试用。

## 为什么对标 copy 而不是理论峰值

448 GB/s 是由显存时钟乘总线宽度算出的**理论**值，实际不可达。所以 benchmark 里额外跑一个 `y = x` 的纯拷贝 kernel，它搬运的字节数和理想 softmax 完全一致（1读+1写），因此它的 GB/s 就是**真实可达上限**。主指标是 `%copy`。

没有这条基线会误判：cols=1024 的 v0 看 `%peak` 是 77.8%，像是还有 22% 空间；但 copy 本身只能跑到 90.8%，所以 v0 其实已经拿到可达带宽的 85.7%。追一个不存在的天花板会浪费在错误的优化上。

## 版本对比

RTX 3070 Ti Laptop GPU，理论峰值 448.06 GB/s，warmup 5 / iterations 30，`normal` 模式：

| rows | cols | copy GB/s | v0 %copy | v1 %copy | v1自适应 %copy | v2a %copy | v2b %copy |
|---|---|---|---|---|---|---|---|
| 8192 | 128 | 375.21 | 15.9% | 29.3% | 29.3% | 95.9% | **98.8%** |
| 4096 | 1024 | 406.55 | 85.7% | 96.2% | 96.4% | 96.4% | 96.7% |
| 1024 | 8192 | 413.91 | 50.9% | 52.4% | 93.3% | 93.5% | 93.5% |
| 4096 | 1023 | 405.99 | 85.1% | 96.8% | 96.7% | 96.8% | 96.5% |
| 8192 | 32 | (见下) | 2.7% | 5.1% | 5.5% | 73.6% | **95.0%** |

cols ≥ 1024 那几行 v2a / v2b 和 v1自适应 数字几乎相同，这是设计使然：v2 的 dispatcher 在 `cols > 256` 时直接转交 `launch_softmax_adaptive`，跑的是同一个 kernel。v2 只负责窄行。

三种输入模式下性能与精度基本一致（完整输出见 benchmark），说明数值稳定性处理没有可观测的性能代价。

`cols=32` 的 copy 基线显示 646 GB/s、`%peak` 144%，这是测量假象而非错误：该 shape 总数据量只有 1 MB，完整装进 4 MB 的 L2，拷贝根本没碰 DRAM。这一行的 `%copy` 因此没有参考意义，只看 v0→v1 的相对加速。

## v1 验证了什么

v1 相对 v0 **只改了归约方式**，block-per-row 的结构、三趟读、访存模式全部保持不变。这样任何性能变化都只能归因于归约本身。

v0 用 shared memory 树形归约：每次归约 log2(256) = 8 轮，每轮一次 `__syncthreads` 加一次 shared memory 往返，两次归约共 **16 个 block 级 barrier**。v1 用 `__shfl_xor_sync` 在 warp 内做 5 步纯寄存器归约，再通过 8 个 float 的 shared memory 合并 8 个 warp 的 partial，每次归约只需 **1 个 barrier**。

结果证实了「barrier 是短行主瓶颈」的判断：短行加速 1.84x / 1.92x。cols=128 一行只有 128 个元素的有用工作，却要付 16 次全 block 同步，同步开销完全盖过了实际计算。

但同时也暴露出：**另外两个 shape 的瓶颈根本不是 barrier**，v1 在 cols=1024 只有 1.12x，在 cols=8192 只有 1.03x。

## 三个 shape，三种不同的瓶颈

因为 GB/s 的分子是**理想**访存量（1读+1写）而非实际字节数，可以反过来推断真实 DRAM traffic。三个 shape 得出的结论完全不同：

**cols=1024（96.2%）：已到顶。** 一行 4 KB，三趟紧邻发生在同一 block 内，数据留在 L1，第 2、3 趟没走 DRAM。反算验证：若三趟全打 DRAM，实际 traffic 是计数值的 2 倍 → 391 × 2 = 782 GB/s，超过 448 的物理上限，不可能。所以实际 traffic 已接近最小值，只剩 3.8% 空间。

**cols=8192（52.4%）：带宽已打满，但搬了 2 倍字节。** 一行 32 KB，多个 block 并发时挤不进 128 KB 的 L1，互相把对方数据踢出去，三趟读真的都打到 DRAM。反算：216.67 × 2 = 433 GB/s，而 copy ceiling 是 413 GB/s，两者几乎相等（比值 1.05）。**DRAM 带宽已经跑满了**，52.4% 不是访存效率低，而是搬了 2 倍不必要的字节。任何改善访存模式的手段（向量化、对齐）都不会有帮助——路已跑满，只能少跑。

**cols=128（29.3%）：完全不受带宽限制。** 一行仅 512 字节。反算最坏情况 109.76 × 2 = 219 GB/s，而 ceiling 是 375 GB/s，**离带宽上限还很远**。真正的问题是每行的固定开销摊不掉：256 线程处理 128 列，一半线程空转却照样参与归约和 barrier；8192 个 block 各自只有 128 个元素的有用工作。`cols=32` 更极端，224/256 线程空转，掉到 5.1%。

| shape | 瓶颈 | 对应手段 |
|---|---|---|
| cols=128 / 32 | 固定开销 + 线程空转（**未打满带宽**） | 缩小处理一行的线程组 |
| cols=1024 | 无（**已达最小 traffic**） | 不动 |
| cols=8192 | 搬了 2 倍字节（**带宽已打满**） | 提高 L1 命中或减少趟数 |

这些结论是从带宽数字反推的。下一节把它们变成直接测量，其中一条被推翻了。

## 把推断变成测量：探针法

上面的诊断全靠算术反推。本来该用 Nsight Compute 读 `dram__bytes` 验证，但这台机器的 CUDA 来自 conda 环境，不含 `ncu`。

替代方案是**构造一对只差读取趟数的探针 kernel**，用 cudaEvent 计时：

- `probe_1read`：读一趟写一趟，traffic = 2N
- `probe_3read`：读三趟写一趟，traffic = 4N

两者都去掉 `exp`，但 `probe_3read` **保留完整的依赖链**（第 2 趟等第 1 趟的 max，第 3 趟等第 2 趟的 sum），否则编译器会把三趟合并或提前发射 load。这样两者唯一的差异就是读的趟数。

判据很干净：若额外两趟真打 DRAM，`t(3read)/t(1read)` 应趋近 **2.0**；若命中 L1，应趋近 **1.0**。

```
shape        1read ms  3read ms   ratio   copy ms    v1 ms
8192x128       0.0234    0.0687    2.93    0.0224   0.0693
4096x1024      0.0834    0.0860    1.03    0.0825   0.0857
1024x8192      0.1622    0.3087    1.90    0.1620   0.3090
8192x32        0.0161    0.0707    4.39    0.0036   0.0710
```

**cols=1024 → 1.03，cols=8192 → 1.90：原推断成立。** 前者额外两趟确实留在 L1，后者确实打到 DRAM（1.90 已很接近理论上限 2.0）。

**cols=128 → 2.93，cols=32 → 4.39：原推断被推翻。** 比值**超过了 2.0**。4N/2N = 2 是 traffic 能解释的天花板，而 512 字节一行必然在 L1，所以窄行的代价根本不是访存量，而是**三趟之间的串行依赖构成的关键路径**——行太短，block 内没有足够的工作去重叠这三段依赖。README 原先写的「短行主要是 barrier 开销」不准确：barrier 只是关键路径上的一环，趟与趟之间的等待才是主体。

**额外发现：block-per-row 这个映射本身就很贵。** cols=32 时单趟探针 0.0161 ms vs copy 0.0036 ms，**慢 4.5 倍**。这里只有一趟读，没有依赖链，没有多余 traffic，唯一的差别就是映射方式：256 线程去处理 32 个元素，224 个空转；8192 个 block 每个只做 128 字节的有用工作。

这两条合起来直接指出 v2 该做什么：不是省 traffic（本来就在 L1），而是**缩短关键路径 + 换掉映射**。

## occupancy：为什么 1024 线程会崩

block size 实验里 1024 线程在窄行慢 11 倍，当时归因于「一个 SM 只放得下一个 block，没有别的 block 来掩藏 barrier」。`cudaOccupancyMaxActiveBlocksPerMultiprocessor` 是 runtime API，不需要 profiler 就能直接查：

```
block size  blocks/SM  warps resident  occupancy  registers
   128         12            48          100.0%      38
   256          6            48          100.0%      38
   512          3            48          100.0%      38
  1024          1            32           66.7%      38
```

推断成立，而且机制比原先说的更具体：这张卡每 SM 上限 1536 线程，1536 / 1024 = 1.5，**向下取整只能驻留 1 个 block**，占用率被硬件卡在 66.7%。而 128/256/512 都能整除，全部拿到 100%。

barrier 的代价由此放大：一个 block 内所有 warp 都停在同一个 `__syncthreads` 上，只有**别的 block** 能填补这段空窗。1024 线程时没有别的 block，空窗全部暴露。

顺带量到一个对 v2 有用的数字：registers = 38，而满占用预算是 65536 / 1536 ≈ 42。**还有 4 个寄存器余量**，说明 v2 做整行寄存器驻留是可行的，不会立刻掉占用率。

## block size 实验：L1 假设成立，但没有全局最优值

上面对 cols=8192 的诊断是「L1 装不下并发的多行」。这个假设可以用**改一个常量**来验证：把 block 从 256 加到 1024 线程，每个 SM 同时驻留的行数变少，每行分到更多 L1。指令数完全不变，只有缓存行为变。

kernel 因此按 block size 模板化。结果（`%copy`，normal 模式）：

| rows | cols | block 256 | block 1024 | 自适应 |
|---|---|---|---|---|
| 8192 | 128 | 29.4% | **2.6%** | 29.4% |
| 4096 | 1024 | 96.2% | **18.6%** | 96.2% |
| 1024 | 8192 | 52.4% | **93.6%** | 93.5% |
| 4096 | 1023 | 96.6% | 18.7% | 96.5% |
| 8192 | 32 | 5.2% | 0.4% | 5.2% |

**cols=8192：52.4% → 93.6%，1.79x。** L1 thrashing 假设证实。这里的关键是收益并非来自减少指令或减少趟数——三趟读一个都没少，只是第 2、3 趟从 DRAM 变成了 L1 命中。原本 433 GB/s 的真实 DRAM traffic 降到约 200 GB/s。

**同时短行崩了：29.4% → 2.6%，慢 11 倍。** 1024 线程处理 128 列意味着 87.5% 的线程整个 kernel 都在空转，却照样参与两次 block 级 barrier；cols=1024 也从 96.2% 掉到 18.6%。

所以 block size **不存在全局最优值**，两个 regime 的诉求完全相反。`launch_softmax_adaptive` 按 cols 分发（≥4096 用 1024 线程，≥2048 用 512，否则 256），阈值大致对应「每线程 8 个元素」。自适应版本在每个 shape 上都拿到了该 shape 的最好成绩。

这次实验的性价比是这里最值得记的一点：改一个常量，换来长行 1.79x，同时证伪了「block size 可以固定」这个隐含假设。

## v2：换映射，窄行从 29% 到 99%

诊断指向两件事——关键路径太长、映射太浪费——v2 一次解决两个。

**换映射：一个 lane 组处理一行。** 不再是一个 block 一行，而是 `LanesPerRow` 个连续 lane 负责一行，一个 warp 同时处理 `32 / LanesPerRow` 行。cols=128 用 32 lane（一整个 warp 一行），cols=32 用 8 lane（一个 warp 4 行）。空转彻底消失：每个 lane 都有活干。

**删掉所有 barrier。** lane 组永远落在同一个 warp 内，warp 内本身锁步，归约只需 `__shfl_xor_sync`，`__syncthreads` 和 shared memory 一个都不要。这直接砍掉了关键路径上的同步部分。

xor shuffle 在这里有个顺手的性质：`LanesPerRow` 取 2 的幂时，`lane XOR offset` 对 `offset < LanesPerRow` 永远不会跨出组边界，所以分段归约**不需要任何额外 mask**，只是循环少走几步。`LanesPerRow` 做成模板参数，编译期展开。

**v2a 与 v2b 的区别只有一点：行读几趟。**

- v2a：仍然三趟读，但每趟用 `float4`，且无 barrier。
- v2b：`cols == LanesPerRow * 4` 时，一行恰好每 lane 一个 `float4`，于是**把这个 float4 留在寄存器里**，全程只读一次 global。max、sum、输出全部在寄存器上算完。这需要 cols 编译期已知，所以只覆盖 cols ∈ {32, 64, 128}，其余 shape 自动退回 v2a。

结果（`%copy`，normal 模式）：

| shape | v1自适应 | v2a | v2b |
|---|---|---|---|
| 8192×128 | 29.3% | 95.9% | **98.8%** |
| 8192×32 | 5.5% | 73.6% | **95.0%** |

cols=128 提升 **3.4x**，cols=32 提升 **17.3x**。三种输入模式一致，精度全部通过。

两点值得注意：

- **v2a 已经拿下大部分收益**（29.3% → 95.9%），说明窄行的主要代价确实是映射和 barrier，而非 traffic。这印证了探针法的结论：省 traffic 在这里几乎没用，因为数据本来就在 L1。
- **v2b 的额外收益集中在 cols=32**（73.6% → 95.0%）。行越短，三趟的固定开销占比越高，寄存器驻留把它压掉；cols=128 时 v2a 已接近 copy 上限，v2b 只多拿 2.9 个点。

occupancy 那节测到 registers=38、满占用预算 42，正是 v2b 敢做寄存器驻留的依据：每 lane 多留 4 个 float 刚好在预算内。

## 后续优化方向

1. **v3 online softmax**（针对 cols=8192，当前 93.5%）。用满足结合律的 combine 算子把 max 和 sum 两趟合并：

   ```
   (m₁,l₁) ⊕ (m₂,l₂):  m = max(m₁,m₂),  l = l₁·exp(m₁-m) + l₂·exp(m₂-m)
   ```

   结合律成立意味着它能直接塞进 shuffle 归约，不只能串行递推。3读+1写 降到 2读+1写。

   注意优先级已经降了：大 block 已经把这个 shape 拉到 93.5%，剩余空间只有 6.5%，原本预期的 2x 收益不复存在。仍然值得做的理由是它**不依赖 L1 命中**（cols 再大、L1 再挤都成立），而大 block 的效果会随 cols 继续增长而衰减。而且这个 combine 算子就是 flash attention 的核心构件，单独实现一版的价值不只在这个算子本身。

2. **把寄存器驻留推广到 cols=256～1024**：v2b 目前只覆盖 `cols == LanesPerRow * 4`，即一行恰好一个 float4 每 lane。往上扩需要每 lane 持有多个 float4，寄存器压力线性上涨，而 occupancy 预算只剩 4 个寄存器。这些 shape 已在 96% 以上，收益上限个位数，性价比低。

### 已完成

- ~~v2 subwarp-per-row + 向量化~~：见上一节，窄行 29.3% → 98.8%。
- ~~整行驻留寄存器~~：v2b 已在 cols ≤ 128 落地。更大的 cols 见上面第 2 条。

### 明确不做

- **把数据 tile 进 shared memory**：softmax 没有跨线程复用，每个元素在每趟里只被一个线程读一次。搬进 shared memory 是纯多余的往返（global → shared → register，而不是直接 global → register）。shared memory 在这里唯一的正当用途是归约 scratch，只需几个 float。
- **`cp.async` 预取**：一是它只能 global → shared，写不到寄存器，为了用它反而要引入上面那个多余的中转；二是它解决的是延迟掩藏，而 softmax 是带宽受限——cols=1024 已经跑到可达带宽 96%，剩余空间不在延迟上。真正需要 ILP 时，让每个线程多读几个 `float4` 就够了，比软件流水简单得多。

## 构建与运行

```bash
cmake -S . -B build
cmake --build build --target bench_stable_softmax -j
./build/operators/transformer_ops/stable_softmax/bench_stable_softmax
```

或通过 ctest：

```bash
ctest --test-dir build -R stable_softmax
```
