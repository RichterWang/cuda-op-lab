# CPU Matrix Multiplication Cache Optimization Study

This directory implements and compares three matrix multiplication algorithms to study the impact of CPU cache on computational performance.

## Algorithm Implementations

### 1. Naive Algorithm (Triple Loop)
- File: `src/naive.cpp`
- Description: Standard i-j-k triple-loop matrix multiplication
- Characteristics: Simple implementation but poor cache locality

### 2. Cache-Aware Blocked Algorithm
- File: `src/blocked.cpp`
- Description: Divides matrices into fixed-size blocks for computation
- Parameter: `block_size` - block dimension (e.g., 32, 64, 128)
- Characteristics: Explicitly optimizes cache usage, requires tuning for hardware

### 3. Cache-Oblivious Recursive Algorithm
- File: `src/recursive.cpp`
- Description: Recursively divides matrices into smaller sub-matrices
- Parameter: `threshold` - recursion termination threshold (e.g., 32, 64, 128)
- Characteristics: Independent of cache size, automatically adapts to memory hierarchy

## Project Structure

```
cpu_cache/
├── include/
│   └── cpu_matmul.h          # Header: algorithm interface definitions
├── src/
│   ├── naive.cpp             # Naive triple-loop implementation
│   ├── blocked.cpp           # Blocked algorithm implementation
│   ├── recursive.cpp         # Recursive algorithm implementation
│   └── utils.cpp             # Utility functions (correctness verification, etc.)
├── benchmark/
│   └── bench_cpu_matmul.cpp  # Performance benchmark program
├── CMakeLists.txt            # Build configuration
└── README.md                 # This document
```

## Build and Run

### Building

From the project root directory:

```bash
# Configure build
mkdir -p build && cd build
cmake ..

# Compile
make bench_cpu_matmul

# Or build directly in cpu_cache_study directory
cd operators/gemm/cpu_cache_study
mkdir -p build && cd build
cmake ..
make
```

### Running Benchmarks

```bash
# Basic run
./build/bench_cpu_matmul

# Measure cache misses with perf
perf stat -e cache-misses,cache-references,instructions,cycles ./build/bench_cpu_matmul

# Detailed cache analysis
perf stat -e L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses ./build/bench_cpu_matmul
```

## Experimental Design

### Test Matrix Sizes

The program automatically tests the following sizes (including both power-of-2 and non-power-of-2):

- 128×128×128 (Small, power of 2)
- 256×256×256 (Medium, power of 2)
- 512×512×512 (Large, power of 2)
- 1024×1024×1024 (Very large, power of 2)
- 300×300×300 (Medium, non-power of 2)
- 500×500×500 (Large, non-power of 2)
- 768×768×768 (Common neural network size, non-power of 2)

### Parameter Settings

- **Block sizes**: 16, 32, 64, 128
- **Recursive thresholds**: 16, 32, 64, 128

### Metrics Measured

1. **Execution Time** (ms)
2. **Computational Performance** (GFLOPS)
3. **Correctness Verification** (relative error < 1e-4)
4. **Cache Miss Rate** (measured via perf tool)
5. **Speedup** (relative to naive algorithm)

## Expected Results Analysis

### Cache-Aware Blocked Algorithm
- **Advantage**: Significantly improved cache hit rate with appropriate block size
- **Optimal Parameters**: Depends on CPU cache sizes (L1/L2/L3)
- **Typical Speedup**: 2-5x (vs. naive)

### Cache-Oblivious Recursive Algorithm
- **Advantage**: Automatically adapts to memory hierarchy without tuning
- **Performance**: Usually slightly lower than optimally-tuned blocked algorithm, but more stable
- **Typical Speedup**: 1.5-4x (vs. naive)

### Power-of-2 vs Non-Power-of-2
- **Power-of-2**: Generally better performance due to address and cache line alignment
- **Non-Power-of-2**: May encounter cache conflicts, slight performance degradation

## Advanced Experiments

### 1. Different Compilation Optimization Levels

```bash
# O2 optimization
cmake -DCMAKE_BUILD_TYPE=Release ..
make

# O3 + native architecture optimization
cmake -DCMAKE_CXX_FLAGS="-O3 -march=native" ..
make
```

### 2. Detailed Cache Analysis

```bash
# View cache misses at different levels
perf stat -e L1-dcache-load-misses,L2-rqsts.miss,LLC-load-misses ./bench_cpu_matmul

# Use cachegrind for analysis
valgrind --tool=cachegrind ./bench_cpu_matmul

# View cachegrind report
cg_annotate cachegrind.out.<pid>
```

### 3. Matrix Size Sweep

Modify the `test_configs` array in `benchmark/bench_cpu_matmul.cpp` to add more test sizes.

### 4. Matrix Storage Order

Experiment with row-major vs column-major storage order impact on performance.

## Theoretical Background

### Cache Locality Principles

- **Temporal Locality**: Recently accessed data is likely to be accessed again
- **Spatial Locality**: Data at adjacent addresses tends to be accessed sequentially

### Why is the Naive Algorithm Slow?

In the standard i-j-k loop:
```cpp
for (i) {
    for (j) {
        for (k) {
            C[i][j] += A[i][k] * B[k][j];
        }
    }
}
```

- Accessing `B[k][j]` jumps across rows, breaking spatial locality
- When matrices are large, columns of B cannot fully reside in cache
- Results in many cache misses

### How Does Blocking Improve Performance?

Divides large matrices into small blocks that fit in cache:
- High data reuse within blocks
- Reduces main memory accesses
- Lower cache miss rate

### Advantages of Recursive Algorithm

- Automatically decomposes problem to cache-appropriate sizes through recursion
- No need to know specific cache sizes
- Automatically optimizes for multi-level cache hierarchy

## References

1. [Cache-Oblivious Algorithms](https://en.wikipedia.org/wiki/Cache-oblivious_algorithm)
2. [Matrix Multiplication Optimization](https://csapp.cs.cmu.edu/3e/waside/waside-blocking.pdf)
3. [Performance Analysis with perf](https://perf.wiki.kernel.org/index.php/Tutorial)

## Extension Ideas

1. **SIMD Vectorization**: Use AVX/AVX2/AVX-512 instructions
2. **Multi-threading**: OpenMP or pthread parallelization
3. **BLAS Library Comparison**: Compare with OpenBLAS, MKL, etc.
4. **Sparse Matrices**: Test optimization strategies for sparse matrices
5. **Mixed Precision**: Test FP16/BF16 performance

## conclusion:
L1 data cache miss rate is 40.77%, which is an inherent characteristic of the matrix multiplication algorithm, because:
- L1 cache (32–48KB) cannot hold a full matrix block
- The matrix access pattern contains a large number of strided accesses
However, our blocking optimization algorithm successfully brought the L2/L3 cache miss rate down to 0.43%, meaning that 99.57% of L1 misses are satisfied in L2/L3, avoiding expensive main-memory accesses. This is exactly the design goal of cache-aware and cache-oblivious algorithms.

---
test config: small 2n
matrix.size: M=128, N=128, K=128
if sepcial size: 2n

--- formal test ---
Naive (baseline)          | time:     0.8344 ms | GFLOPS:   5.0267 | accurancy: pass

--- Cache-Aware tiling (different block size) ---
Blocked (b=16)            | time:     0.6575 ms | GFLOPS:   6.3788 | accurancy: pass
Blocked (b=32)            | time:     0.6539 ms | GFLOPS:   6.4145 | accurancy: pass
Blocked (b=64)            | time:     0.6652 ms | GFLOPS:   6.3049 | accurancy: pass
Blocked (b=128)           | time:     0.8290 ms | GFLOPS:   5.0593 | accurancy: pass

--- Cache-Oblivious (different threshold) ---
Recursive (t=16)          | time:     0.6691 ms | GFLOPS:   6.2688 | accurancy: pass
Recursive (t=32)          | time:     0.6545 ms | GFLOPS:   6.4089 | accurancy: pass
Recursive (t=64)          | time:     0.6656 ms | GFLOPS:   6.3012 | accurancy: pass
Recursive (t=128)         | time:     0.8285 ms | GFLOPS:   5.0624 | accurancy: pass

---
test config: mid 2n
matrix.size: M=256, N=256, K=256
if sepcial size: 2n

--- formal test ---
Naive (baseline)          | time:     7.5988 ms | GFLOPS:   4.4158 | accurancy: pass

--- Cache-Aware tiling (different block size) ---
Blocked (b=16)            | time:     5.2380 ms | GFLOPS:   6.4059 | accurancy: pass
Blocked (b=32)            | time:     5.1893 ms | GFLOPS:   6.4660 | accurancy: pass
Blocked (b=64)            | time:     5.3844 ms | GFLOPS:   6.2318 | accurancy: pass
Blocked (b=128)           | time:     7.6396 ms | GFLOPS:   4.3921 | accurancy: pass

--- Cache-Oblivious (different threshold) ---
Recursive (t=16)          | time:     5.3644 ms | GFLOPS:   6.2550 | accurancy: pass
Recursive (t=32)          | time:     5.2291 ms | GFLOPS:   6.4168 | accurancy: pass
Recursive (t=64)          | time:     5.9226 ms | GFLOPS:   5.6655 | accurancy: pass
Recursive (t=128)         | time:     8.1040 ms | GFLOPS:   4.1405 | accurancy: pass

---
test config: max 2n
matrix.size: M=512, N=512, K=512
if sepcial size: 2n

--- formal test ---
Naive (baseline)          | time:    86.7167 ms | GFLOPS:   3.0955 | accurancy: pass

--- Cache-Aware tiling (different block size) ---
Blocked (b=16)            | time:    42.4328 ms | GFLOPS:   6.3261 | accurancy: pass
Blocked (b=32)            | time:    43.4879 ms | GFLOPS:   6.1726 | accurancy: pass
Blocked (b=64)            | time:    65.2698 ms | GFLOPS:   4.1127 | accurancy: pass
Blocked (b=128)           | time:    66.2476 ms | GFLOPS:   4.0520 | accurancy: pass

--- Cache-Oblivious (different threshold) ---
Recursive (t=16)          | time:    44.0103 ms | GFLOPS:   6.0994 | accurancy: pass
Recursive (t=32)          | time:    43.1442 ms | GFLOPS:   6.2218 | accurancy: pass
Recursive (t=64)          | time:    64.2393 ms | GFLOPS:   4.1787 | accurancy: pass
Recursive (t=128)         | time:    64.8609 ms | GFLOPS:   4.1386 | accurancy: pass

---
test config: supermax 2n
matrix.size: M=1024, N=1024, K=1024
if sepcial size: 2n

--- formal test ---
Naive (baseline)          | time:  2142.8740 ms | GFLOPS:   1.0022 | accurancy: pass

--- Cache-Aware tiling (different block size) ---
Blocked (b=16)            | time:   360.4417 ms | GFLOPS:   5.9579 | accurancy: pass
Blocked (b=32)            | time:   440.7401 ms | GFLOPS:   4.8724 | accurancy: pass
Blocked (b=64)            | time:   525.9704 ms | GFLOPS:   4.0829 | accurancy: pass
Blocked (b=128)           | time:   534.9673 ms | GFLOPS:   4.0142 | accurancy: pass

--- Cache-Oblivious (different threshold) ---
Recursive (t=16)          | time:   369.3722 ms | GFLOPS:   5.8139 | accurancy: pass
Recursive (t=32)          | time:   436.8446 ms | GFLOPS:   4.9159 | accurancy: pass
Recursive (t=64)          | time:   528.2183 ms | GFLOPS:   4.0655 | accurancy: pass
Recursive (t=128)         | time:   538.5755 ms | GFLOPS:   3.9873 | accurancy: pass

---
test config: mid not 2n
matrix.size: M=300, N=300, K=300
if sepcial size: not 2n

--- formal test ---
Naive (baseline)          | time:     9.7680 ms | GFLOPS:   5.5283 | accurancy: pass

--- Cache-Aware tiling (different block size) ---
Blocked (b=16)            | time:     8.3460 ms | GFLOPS:   6.4702 | accurancy: pass
Blocked (b=32)            | time:     8.3232 ms | GFLOPS:   6.4879 | accurancy: pass
Blocked (b=64)            | time:     8.5036 ms | GFLOPS:   6.3503 | accurancy: pass
Blocked (b=128)           | time:     8.7408 ms | GFLOPS:   6.1779 | accurancy: pass

--- Cache-Oblivious (different threshold) ---
Recursive (t=16)          | time:     8.2666 ms | GFLOPS:   6.5323 | accurancy: pass
Recursive (t=32)          | time:     7.6877 ms | GFLOPS:   7.0242 | accurancy: pass
Recursive (t=64)          | time:     8.1871 ms | GFLOPS:   6.5958 | accurancy: pass
Recursive (t=128)         | time:     8.4818 ms | GFLOPS:   6.3665 | accurancy: pass

---
test config: max not 2n
matrix.size: M=500, N=500, K=500
if sepcial size: not 2n

--- formal test ---
Naive (baseline)          | time:    49.4286 ms | GFLOPS:   5.0578 | accurancy: pass

--- Cache-Aware tiling (different block size) ---
Blocked (b=16)            | time:    38.7332 ms | GFLOPS:   6.4544 | accurancy: pass
Blocked (b=32)            | time:    38.5029 ms | GFLOPS:   6.4930 | accurancy: pass
Blocked (b=64)            | time:    39.3884 ms | GFLOPS:   6.3471 | accurancy: pass
Blocked (b=128)           | time:    40.6834 ms | GFLOPS:   6.1450 | accurancy: pass

--- Cache-Oblivious (different threshold) ---
Recursive (t=16)          | time:    38.4035 ms | GFLOPS:   6.5098 | accurancy: pass
Recursive (t=32)          | time:    37.0164 ms | GFLOPS:   6.7538 | accurancy: pass
Recursive (t=64)          | time:    38.4331 ms | GFLOPS:   6.5048 | accurancy: pass
Recursive (t=128)         | time:    40.8939 ms | GFLOPS:   6.1134 | accurancy: pass

---
test config: NN usual
matrix.size: M=768, N=768, K=768
if sepcial size: not 2n

--- formal test ---
Naive (baseline)          | time:   273.7709 ms | GFLOPS:   3.3092 | accurancy: pass

--- Cache-Aware tiling (different block size) ---
Blocked (b=16)            | time:   142.1541 ms | GFLOPS:   6.3732 | accurancy: pass
Blocked (b=32)            | time:   140.2188 ms | GFLOPS:   6.4611 | accurancy: pass
Blocked (b=64)            | time:   156.3520 ms | GFLOPS:   5.7944 | accurancy: pass
Blocked (b=128)           | time:   213.7266 ms | GFLOPS:   4.2389 | accurancy: pass

--- Cache-Oblivious (different threshold) ---
Recursive (t=16)          | time:   147.5855 ms | GFLOPS:   6.1386 | accurancy: pass
Recursive (t=32)          | time:   142.4976 ms | GFLOPS:   6.3578 | accurancy: pass
Recursive (t=64)          | time:   145.5396 ms | GFLOPS:   6.2249 | accurancy: pass
Recursive (t=128)         | time:   216.7782 ms | GFLOPS:   4.1792 | accurancy: pass

## Appendix perf_output
 Performance counter stats for './operators/gemm/cpu_cache/bench_cpu_matmul':

        63,191,569      cpu_atom/cache-misses/           #    0.08% of all cache refs           (0.00%)
        33,587,680      cpu_core/cache-misses/           #    0.18% of all cache refs           (100.00%)
    83,919,954,349      cpu_atom/cache-references/                                              (0.00%)
    18,678,791,112      cpu_core/cache-references/                                              (100.00%)
   <not supported>      cpu_atom/L1-dcache-load-misses/                                       
   124,503,964,264      cpu_core/L1-dcache-load-misses/                                         (100.00%)
                 0      cpu_atom/LLC-load-misses/                                               (0.00%)
         1,318,230      cpu_core/LLC-load-misses/                                               (100.00%)

     112.379565512 seconds time elapsed

     112.367278000 seconds user
       0.011000000 seconds sys

## Appendix perf_output
This part the program only runs on p-core, so L1 cache, as it is so small, it cannot store the whole matrix as the 128 * 128 matrix with float can take up to 64kB. And almost all the L1 miss turns to L2/3 hit, whcih means the cycles grow is prevented from ~8cycles to ~196cycles.
 Performance counter stats for './operators/gemm/cpu_cache/bench_cpu_matmul':

     <not counted>      cpu_atom/cache-misses/                                                  (0.00%)
        81,142,852      cpu_core/cache-misses/           #    0.43% of all cache refs         
     <not counted>      cpu_atom/cache-references/                                              (0.00%)
    18,713,870,845      cpu_core/cache-references/                                            
     <not counted>      cpu_atom/L1-dcache-loads/                                               (0.00%)
   309,478,931,556      cpu_core/L1-dcache-loads/                                             
   <not supported>      cpu_atom/L1-dcache-load-misses/                                       
   126,186,390,116      cpu_core/L1-dcache-load-misses/  #   40.77% of all L1-dcache accesses 

     113.227248256 seconds time elapsed

     113.217793000 seconds user
       0.010000000 seconds sys