# Shared Memory SGEMM Implementation - Design Review

## Overview

This document captures the design principles and implementation details of the shared memory optimized SGEMM (Single-precision General Matrix Multiply) kernel.


### Run 3 Tile-Size Comparison 8 x 8

- Device NVIDIA GeForce RTX 3070 Ti Laptop GPU
- Matrix shape M=1024 N=512 K=256
- Warmup iterations 5
- Timed iterations 30
- Shared-memory tile 8 x 8 threads

 Implementation  Average time ms  Throughput GFLOPS  Relative to cuBLAS  Maximum relative error 
  ---  ---  ---  ---  --- 
   cuBLAS  0.0533  5038.00  100.00%  6.91413879e-06 
    Naive CUDA  0.3020  888.93  17.64%  7.40907080e-06 
     Shared-memory CUDA  0.4004  670.39  13.31%  7.40907080e-06 

     ### Run 3 Observation

     The 8 x 8 tile reduced shared-memory throughput to 670.39 GFLOPS. Compared with the 16 x 16 tile it performs less computation per tile provides less data reuse and requires 32 K tiles with 64 block-wide synchronization points. The added shared-memory loads and barriers outweigh the benefit of reduced thread-block size.

     ### Why Naive Can Still Win

     The naive kernel does not necessarily fetch every operand from DRAM. Its B accesses are coalesced while repeated A accesses within a warp can be served by broadcast and cache mechanisms. Repeated data can also hit L1 or L2 cache. The basic shared-memory kernel adds explicit global-to-shared copies and two syncthreads barriers per tile. When hardware caching already handles the naive access pattern well those explicit copies and barriers can cost more than the avoided global-memory traffic.
     
## Core Design Philosophy

### Primary Goal: Data Reuse Through Shared Memory

The main optimization strategy is to **reduce global memory accesses** by caching frequently-reused data in shared memory, rather than fetching from global memory repeatedly.

**Key Principle:** Spatial reuse, not temporal overlapping
- Load data once from global memory (slow, ~400 cycles)
- Reuse it multiple times from shared memory (fast, ~20 cycles)
- Reduces memory bandwidth pressure by `BLOCK_SIZE` times

---

## Memory Hierarchy

```
Global Memory (DRAM)
    ↓ Load once per tile
Shared Memory (on-chip SRAM, per block)
    ↓ Read multiple times
Registers (per thread)
    ↓ Accumulate results
Global Memory (write back once)
```

### Memory Visibility

| Memory Type | Visibility | Lifetime | Speed | Usage |
|-------------|-----------|----------|-------|-------|
| **Registers** | Single thread | Thread execution | Fastest (~1 cycle) | `sum` accumulator |
| **Shared Memory** | All threads in block | Block execution | Fast (~20 cycles) | `tileA`, `tileB` |
| **Global Memory** | All threads | Program duration | Slow (~400 cycles) | `A`, `B`, `C` |

---

## Architecture Design

### 1. Block and Tile Relationship

```
Block Size = Tile Size = 32 * 32
- 32×32 = 1024 threads per block
- Each thread: loads 1 element, computes 1 output
- One-to-one mapping: thread(ty, tx) ↔ tile[ty][tx]
```

**Why matching sizes?**
- Simple and intuitive indexing
- Load balanced (no idle threads)
- Clean code structure

### 2. Shared Memory Organization

Each block has **independent** shared memory:

```
Block 0:
  ├─ tileA[64][64]  (16 KB)
  └─ tileB[64][64]  (16 KB)

Block 1:
  ├─ tileA[64][64]  (independent 16 KB)
  └─ tileB[64][64]  (independent 16 KB)
```

**Blocks cannot access each other's shared memory.**

### 3. Tile Reuse and Overwriting

```
Tile lifecycle in a single block:

Tile 0: Load A[0:64, 0:64], B[0:64, 0:64] → Compute → Accumulate
         ↓ Overwrite
Tile 1: Load A[0:64, 64:128], B[64:128, 0:64] → Compute → Accumulate
         ↓ Overwrite
Tile 2: Load A[0:64, 128:192], B[128:192, 0:64] → Compute → Accumulate
```

**Key Point:** Tiles are processed **serially** (one after another), and shared memory is **reused and overwritten** for each tile. No need to preserve previous tile data because contributions have already been accumulated in registers.

---

## Parallelism Model

### Grid Level (M×N Plane): Parallel

```
Matrix C divided into blocks:
    0    64   128  192
  ┌────┬────┬────┬────┐
0 │B0,0│B0,1│B0,2│B0,3│  ← All blocks execute in parallel
  ├────┼────┼────┼────┤
64│B1,0│B1,1│B1,2│B1,3│  ← Independent computation
  └────┴────┴────┴────┘

Perpendicular to K direction: Parallel execution
```

### Block Level (K Direction): Serial

```
Within each block, tiles are processed sequentially:

for (tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
    Load tile → Compute → Accumulate
    ↓ (serial iteration)
    Load next tile → Compute → Accumulate
}

Parallel to K direction: Serial execution (outer loop)
```

### Thread Level: Parallel Within Tile

```
Within each tile iteration:
- 4096 threads load data in parallel
- 4096 threads compute in parallel
- Each thread accumulates to its private register
```

---

## Implementation Details

### Double Loop Structure

#### Outer Loop: Tile Traversal (Serial)

```cuda
for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
    // Traverse K dimension serially
    // Process one tile at a time
}
```

**Purpose:** Iterate through K dimension because shared memory cannot hold all K elements simultaneously.

#### Inner Loop: Tile Computation (Per Thread)

```cuda
for (int k = 0; k < BLOCK_SIZE; ++k) {
    sum += tileA[ty][k] * tileB[k][tx];
}
```

**Purpose:** Compute partial dot product for current tile. Each thread executes this loop independently, but all threads execute simultaneously.

### Index Mapping

```
Thread Identity:
  ├─ In block: (threadIdx.y, threadIdx.x) = (ty, tx)
  ├─ Block ID: (blockIdx.y, blockIdx.x)
  └─ Global position: (row, col)
      where row = blockIdx.y * BLOCK_SIZE + ty
            col = blockIdx.x * BLOCK_SIZE + tx

Shared Memory:
  ├─ tileA[ty][tx] ← Loaded by thread(ty, tx)
  └─ tileB[ty][tx] ← Loaded by thread(ty, tx)

Register:
  └─ sum ← Private accumulator for thread(ty, tx)

Output:
  └─ C[row][col] = sum ← Thread(ty, tx) writes result
```

### Synchronization Points

#### Sync Point 1: After Loading

```cuda
tileA[ty][tx] = A[...];  // All threads load in parallel
tileB[ty][tx] = B[...];

__syncthreads();  // ★ Wait for all threads to finish loading
```

**Why needed?**
- Fast threads might start computing before slow threads finish loading
- Reading incomplete data leads to incorrect results

#### Sync Point 2: After Computing

```cuda
for (int k = 0; k < BLOCK_SIZE; ++k) {
    sum += tileA[ty][k] * tileB[k][tx];
}

__syncthreads();  // ★ Wait for all threads to finish computing
```

**Why needed?**
- Fast threads might start loading next tile (overwriting data)
- Slow threads still reading from current tile
- Data race: wrong results

### Boundary Handling

```cuda
// Edge cases when matrix dimensions not multiple of BLOCK_SIZE

// For tileA
int a_col = tile_idx * BLOCK_SIZE + tx;
if (row < M && a_col < K) {
    tileA[ty][tx] = A[row * K + a_col];
} else {
    tileA[ty][tx] = 0.0f;  // Padding with zeros
}

// For tileB
int b_row = tile_idx * BLOCK_SIZE + ty;
if (b_row < K && col < N) {
    tileB[ty][tx] = B[b_row * N + col];
} else {
    tileB[ty][tx] = 0.0f;  // Padding with zeros
}
```

**Why padding with zeros?**
- Simplifies computation (no need for conditional in inner loop)
- Zero padding doesn't affect sum results
- Maintains uniform code path for all threads

---

## Data Flow Visualization

### Complete Execution Flow for Block(1, 2)

```
Task: Compute C[64:128, 128:192]

1. Initialization
   ├─ 4096 threads start
   ├─ Each thread: sum = 0.0f (register)
   └─ Shared memory: tileA[64][64], tileB[64][64]

2. Tile 0 (K=0-63)
   ├─ Parallel load: tileA ← A[64:128, 0:64]
   ├─ Parallel load: tileB ← B[0:64, 128:192]
   ├─ __syncthreads()
   ├─ Parallel compute: sum += tileA × tileB
   └─ __syncthreads()

3. Tile 1 (K=64-127)
   ├─ Overwrite tileA ← A[64:128, 64:128]
   ├─ Overwrite tileB ← B[64:128, 128:192]
   ├─ __syncthreads()
   ├─ Compute: sum += tileA × tileB (accumulate)
   └─ __syncthreads()

4. Tile 2, 3, 4... (repeat pattern)

5. Write Back
   └─ Each thread: C[row * N + col] = sum
```

### Single Element Reuse Example

```
Loading: tileA[5][10] = A[global_position]
         ↓ (from global memory, 1 time)

Used by 64 threads:
  ├─ Thread(5, 0): sum += tileA[5][10] * tileB[10][0]
  ├─ Thread(5, 1): sum += tileA[5][10] * tileB[10][1]
  ├─ Thread(5, 2): sum += tileA[5][10] * tileB[10][2]
  └─ ... (64 threads total)
         ↑ (from shared memory, 64 times)

Result: 1 global memory read → 64 shared memory reads
Speedup: ~20× per access (due to latency difference)
```

---

## Performance Analysis

### Memory Access Reduction

**Naive Implementation:**
```
Each thread computes C[i][j]:
  - Reads A[i][k] K times from global memory
  - Reads B[k][j] K times from global memory
  - Total: 2K global memory reads per output element
```

**Tiling Implementation:**
```
Each thread computes C[i][j]:
  - Reads from global memory: 2K / BLOCK_SIZE times
  - Reads from shared memory: 2K times (fast)
  - Total: 2K / BLOCK_SIZE global memory reads per output element

Reduction factor: BLOCK_SIZE (64×)
```

### Computation Intensity

```
For a 64×64 tile:
  - Load: 2 × 64×64 = 8,192 floats (32 KB)
  - Compute: 64×64×64 = 262,144 FMA operations
  - Compute-to-memory ratio: 262,144 / 8,192 = 32

Higher ratio → Better GPU utilization
```

---

## Common Misconceptions

### ❌ Misconception 1: All Tiles Processed in Parallel

**Wrong:** "All tiles along K dimension are computed simultaneously"

**Correct:** Tiles must be processed **serially** because:
- Shared memory capacity is limited
- Results must be accumulated in the same `sum` register
- Cannot store intermediate results for all tiles

### ❌ Misconception 2: Computation Hides Memory Access

**Wrong:** "Tiling overlaps computation and memory access"

**Correct:** Basic tiling does **not** overlap:
```
Timeline:
  [Load Tile 0] → [Compute Tile 0] → [Load Tile 1] → [Compute Tile 1]
  
No overlap in basic implementation!
```

Overlapping requires **double buffering** (advanced optimization).

### ✅ Correct Understanding: Spatial Reuse

**Tiling achieves spatial reuse, not temporal overlapping:**
- Load data once → reuse many times
- Reduces **number** of memory accesses
- Not about hiding latency through overlapping

---

## Key Takeaways

1. **Shared Memory Purpose**: Cache frequently-reused data to reduce global memory traffic

2. **Tile Processing**: Serial traversal of K dimension, each tile overwrites previous in shared memory

3. **Parallelism Model**:
   - Grid level (M×N): Parallel blocks
   - Block level (K): Serial tiles
   - Thread level: Parallel execution within tile

4. **Index Mapping**: Thread(ty, tx) → tile[ty][tx] → C[row][col] (one-to-one correspondence)

5. **Accumulation**: Single `sum` register per thread accumulates contributions from all tiles

6. **Synchronization**: Two barriers per tile (after load, after compute) prevent data races

7. **Performance Gain**: Reduces global memory accesses by factor of `BLOCK_SIZE`

---

## Code Verification Checklist

- [ ] tileA indexing: `tileA[ty][tx]` (not `[tx][ty]`)
- [ ] tileB indexing: `tileB[ty][tx]` (not `[tx][ty]`)
- [ ] A address: `A[row * K + a_col]` (not `+ col`)
- [ ] B address: `B[b_row * N + col]` (not `+ col` alone)
- [ ] Boundary check for A: `row < M && a_col < K`
- [ ] Boundary check for B: `b_row < K && col < N`
- [ ] Two `__syncthreads()` per tile iteration
- [ ] Final boundary check: `row < M && col < N`

---

## Further Optimizations (Not Implemented)

1. **Double Buffering**: Overlap loading next tile with computing current tile
2. **Register Tiling**: Each thread computes multiple output elements
3. **Vectorized Memory Access**: Use float4 for coalesced loads
4. **Warp Shuffle**: Reduce shared memory usage for certain patterns
5. **Bank Conflict Avoidance**: Pad shared memory arrays

These optimizations can provide additional 2-5× speedup on top of basic tiling.

---

*Document created as part of SGEMM kernel implementation review.*

---

## Experimental Record

### Run 1 Baseline Comparison

- Device NVIDIA GeForce RTX 3070 Ti Laptop GPU
- Matrix shape M=1024 N=512 K=256
- Warmup iterations 5
- Timed iterations 30
- Shared-memory tile 32 x 32 threads

 Implementation  Average time ms  Throughput GFLOPS  Relative to cuBLAS  Maximum relative error 
  ---  ---  ---  ---  --- 
   cuBLAS  0.0531  5050.94  100.00%  6.91413879e-06 
    Naive CUDA  0.3022  888.32  17.59%  7.40907080e-06 
     Shared-memory CUDA  0.3648  735.81  14.57%  7.40907080e-06 

     ### Observation

     The shared-memory implementation passed correctness validation but it reached 82.8% of the naive kernel throughput in this configuration. The 32 x 32 block uses 1024 threads which reduces scheduling flexibility while shared-memory loads and synchronization add overhead. The next experiment should compare tile sizes and introduce register tiling so that each thread computes multiple output elements.


     ### Run 2 Tile-Size Comparison 16 x 16

     - Device NVIDIA GeForce RTX 3070 Ti Laptop GPU
     - Matrix shape M=1024 N=512 K=256
     - Warmup iterations 5
     - Timed iterations 30
     - Shared-memory tile 16 x 16 threads

      Implementation  Average time ms  Throughput GFLOPS  Relative to cuBLAS  Maximum relative error 
       ---  ---  ---  ---  --- 
        cuBLAS  0.0531  5050.94  100.00%  6.91413879e-06 
         Naive CUDA  0.3016  890.13  17.62%  7.40907080e-06 
          Shared-memory CUDA  0.3395  790.62  15.65%  7.40907080e-06 

          ### Run 2 Observation

          Reducing the tile from 32 x 32 to 16 x 16 improved shared-memory throughput from 735.81 to 790.62 GFLOPS. The kernel is still slower than naive reaching 88.8% of naive throughput. Smaller blocks improve scheduling flexibility but the larger number of K tiles also increases synchronization overhead.
