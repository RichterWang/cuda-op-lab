# Flash Attention

fp32 accumulate, bf16 storage, single head-group. Measured on an RTX 3070 Ti Laptop (sm_86).

The operator computes `O = softmax(Q K^T / sqrt(d)) V` with an optional causal mask,
for shapes `[batch, heads, seq_len, head_dim]`.

## What this directory is testing

Attention is the first operator here that is not memory bound. `stable_softmax` and
`rms_norm` both had a hard bandwidth ceiling and the whole exercise was closing the
gap to it. Attention has `O(N^2 d)` arithmetic against `O(N d)` of input, so the
arithmetic intensity grows with sequence length and the bottleneck moves.

That changes what a version has to prove. Removing bytes is no longer automatically
a win, and the benchmark reports TFLOPS, algorithmic HBM traffic, and speedup
side by side so a version that cut traffic while losing wall time cannot hide.

## Versions

| version | idea | status |
|---|---|---|
| `v0_unfused` | three kernels, `S = QK^T` materialized to DRAM | baseline |
| `v1_fused_row` | one block per query row, online softmax, `S` never exists | slower than v0 — see below |
| `v2` | query tiling so a loaded K/V tile serves many rows | not written |

## Results

```
head_dim=64:   block=128  tile_n=64  shared=33808B  regs=48  local=0B  blocks/SM=2
head_dim=128:  block=128  tile_n=32  shared=34064B  regs=48  local=0B  blocks/SM=2
```

| shape | causal | v0 ms | v1 ms | v0 TFLOPS | v1 TFLOPS | speedup |
|---|---|---|---|---|---|---|
| b4 h8 n512 d64 | no | 6.06 | 12.48 | 0.35 | 0.17 | 0.49x |
| b4 h8 n512 d64 | yes | 2.96 | 4.87 | 0.36 | 0.22 | 0.61x |
| b2 h8 n2048 d64 | no | 37.89 | 75.93 | 0.45 | 0.23 | 0.50x |
| b2 h8 n2048 d64 | yes | 23.10 | 37.79 | 0.37 | 0.23 | 0.61x |
| b1 h8 n2048 d128 | yes | 22.17 | 41.48 | 0.39 | 0.21 | 0.53x |
| b1 h8 n4096 d64 | yes | 50.18 | 74.93 | 0.34 | 0.23 | 0.67x |

Accuracy: all kernels within 0.5 bf16 ULP at the row scale.

## v1 is slower than v0, and the reason is not the traffic

This is the useful result in the directory so far, so it is worth stating plainly
rather than burying it.

v1 does exactly what the flash attention paper describes: it never materializes the
score matrix. On paper that removes `5 N^2` fp32 words of DRAM traffic per
(batch, head). It is still 0.49–0.67x of the baseline at every shape measured.

The reason is that **neither kernel was ever bandwidth bound at these shapes.** Both
sit between 0.2 and 0.45 TFLOPS on a card that will do roughly an order of magnitude
more. Removing DRAM traffic cannot buy anything when DRAM was not the constraint —
v1 traded a non-bottleneck for two real ones:

- **Occupancy.** The K and V tiles are held in fp32, 33 KB per block, which caps
  residency at 2 blocks per SM: 256 of 1536 available threads, about 17%. There is
  not enough work in flight to cover shared-memory and `exp()` latency.
- **No query reuse.** Each block loads the whole K/V sequence to serve one query
  row. A loaded tile feeds one dot product and one weighted add, then is discarded,
  and the next block loads the same slab again. The `HBM GB` column shows this
  directly: 17.2 GB for v1 against 1.36 GB for v0 at `n2048 d64` dense.

Both point at the same fix, which is what v2 will do: tile the queries so one
loaded K/V tile is amortized across many query rows.

The fused algorithm is not wrong. The claim it makes — that fusion removes the
`O(N^2)` intermediate — is true and the benchmark confirms the traffic reduction.
What v1 establishes is that the traffic reduction alone is not where the speedup
comes from on this hardware at these shapes. Fusion is what makes the reuse
*possible*; the reuse is what makes it *fast*. v1 exists to separate those two
claims, which is why it stays in the tree instead of being replaced in place.

## Two things that were measurement bugs, not kernel bugs

Both were caught by numbers that did not make sense, and both are worth recording
because the wrong conclusion was available and cheap.

**A 32-way bank conflict, read as an algorithmic limit.** The score loop reads
`k_tile[tid][dim]`. With a row stride of `HeadDim` floats and `HeadDim` a multiple
of 32, the bank index reduces to `dim % 32` — every lane in the warp hits the same
bank, on every one of the `HeadDim` iterations. Padding the stride by one float
made v1 go from 0.31–0.43x to 0.49–0.67x. The remaining gap is the real algorithmic
problem; without the padding fix, roughly a third of it would have been
misattributed to the algorithm.

**ULP measured per element, on an output that cancels.** The first accuracy check
normalized each element's error by that element's own magnitude, which is the right
question for `rms_norm` (output is `x * g / rms`, same order as its input) and the
wrong one here. Attention output is a convex combination of zero-mean V rows, so it
cancels: an element can land near zero while every term producing it was order 1,
and the achievable absolute error is set by the terms, not the result.

The tell was that the *unfused baseline* reported 7.0 ULP at `head_dim=128` while
the fused kernel reported 1.0 on the same input. Two implementations sharing no
arithmetic do not disagree by 7x because one has a bug that spares the other. The
metric was reading cancellation. Normalizing by the row's largest element — the
scale the row's arithmetic actually operated at — puts both under 0.5 ULP.

## Layout

```
flash_attention.h                    shared shapes, FLOP counting, launch decls
kernels/v0_unfused.cu                scores / softmax / PV as three kernels
kernels/v1_fused_row.cu              one block per query row, online softmax
benchmark/bench_flash_attention.cu   TFLOPS + traffic + accuracy, non-zero exit on fail
```

FLOP counting and the causal-triangle correction live in the header so every kernel
and the benchmark agree on what a FLOP is. `attention_flops` counts the two GEMMs
(`QK^T` and `PV`) at `2 N^2 d` each and halves them under a causal mask; the softmax
itself is not counted, since it is `O(N^2)` against the GEMMs' `O(N^2 d)`.

## Running

```bash
cmake -S . -B build && cmake --build build --target bench_flash_attention -j
./build/operators/transformer_ops/flash_attention/bench_flash_attention
```

Exits non-zero if any kernel exceeds 2.0 bf16 ULP at the row scale, so it works as
the regression test under `ctest`.

The `v0` workspace is `batch * heads * N^2 * 4` bytes. Past 1 GB the benchmark skips
v0 for that shape and reports the fused kernels alone rather than failing to
allocate.
