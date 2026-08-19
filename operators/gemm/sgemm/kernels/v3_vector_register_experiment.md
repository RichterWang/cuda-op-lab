# SGEMM V3 Vector Register Experiment Report

## 1. Scope

This report documents the current SGEMM optimization status on an NVIDIA GeForce RTX 3070 Ti Laptop GPU. It compares the CPU reference cuBLAS naive CUDA basic shared-memory tiling V2 tiled-register and V3 vector-register implementations.

The benchmark measures kernel execution with CUDA events. Input generation and host-device transfers are excluded from timed regions. Each implementation uses five warmup iterations and thirty timed iterations.

## 2. Current Implementations

### V2 Tiled Register

V2 combines shared-memory tiling with register tiling.

- Thread block 16 x 16 threads 256 threads per block.
- Output tile 32 x 32 elements of C.
- K tile 32 elements.
- Per-thread output tile 2 x 2 elements.
- Per-thread accumulators 4 floats.

Each block cooperatively loads A and B tiles into shared memory. Each thread then reads a small A fragment and B fragment and accumulates four output elements. This distributes shared-memory loading and synchronization overhead across more arithmetic than the basic shared-memory version.

### V3 Vector Register

V3 expands the register tile and adds vectorized global-memory loads.

- Thread block 16 x 16 threads 256 threads per block.
- Output tile 64 x 64 elements of C.
- K tile 64 elements.
- Per-thread output tile 4 x 4 elements.
- Per-thread accumulators 16 floats.
- Global-memory load width float4.
- Inner loops explicitly unrolled.

Each thread loads four consecutive values at a time from global memory through float4 operations. All threads cooperatively fill two 64 x 64 shared-memory tiles. During each K step a thread loads four A values and four B values from shared memory into local registers then performs sixteen fused multiply-add style accumulations for its 4 x 4 output tile.

## 3. V3 Improvements over V2

V3 increases both data reuse and work per thread.

First the output tile grows from 32 x 32 to 64 x 64 while the block remains 16 x 16 threads. The same number of threads therefore produces four times as many output elements per block.

Second each thread grows from a 2 x 2 output tile to a 4 x 4 output tile. The number of accumulators increases from four to sixteen. A and B values loaded from shared memory are reused across more multiply-accumulate operations.

Third global-memory transfers use float4 loads. Each load instruction moves four consecutive floats and reduces scalar load instruction overhead when addresses are correctly aligned.

Fourth V3 explicitly stages small A and B fragments in thread-local registers before updating the output accumulators. This avoids repeatedly issuing the same shared-memory reads for every output element.

Finally the inner loops are unrolled to expose independent arithmetic operations to the compiler and improve instruction scheduling.

## 4. Experimental Results

### Experiment A K Equals 256

Matrix shape M equals 1024 N equals 512 K equals 256.

- cuBLAS 0.0532 ms 5041.23 GFLOPS.
- Naive CUDA 0.3019 ms 889.13 GFLOPS 17.64 percent of cuBLAS.
- Shared-memory CUDA 0.4007 ms 669.99 GFLOPS 13.29 percent of cuBLAS.
- V2 tiled-register 0.1850 ms 1451.25 GFLOPS 28.79 percent of cuBLAS.
- V3 vector-register 0.1025 ms 2617.95 GFLOPS 51.93 percent of cuBLAS.

The maximum relative error of all custom CUDA kernels was 7.40907080e-06.

At this K value V3 is approximately 1.80 times faster than V2 and 2.94 times faster than naive CUDA.

### Experiment B K Equals 2048

Matrix shape M equals 1024 N equals 512 K equals 2048.

- cuBLAS 0.1851 ms 11599.29 GFLOPS.
- Naive CUDA 2.2288 ms 963.50 GFLOPS 8.31 percent of cuBLAS.
- Shared-memory CUDA 2.8348 ms 757.54 GFLOPS 6.53 percent of cuBLAS.
- V2 tiled-register 1.3170 ms 1630.59 GFLOPS 14.06 percent of cuBLAS.
- V3 vector-register 0.6204 ms 3461.22 GFLOPS 29.84 percent of cuBLAS.

The maximum relative error of all custom CUDA kernels was 5.60283661e-05.

At this K value V3 is approximately 2.12 times faster than V2 and 3.59 times faster than naive CUDA.

## 5. Interpretation

V3 produces a real and significant improvement. Its throughput rises from 2617.95 GFLOPS at K equals 256 to 3461.22 GFLOPS at K equals 2048. The larger K value gives the kernel more computation per launch and makes register reuse more valuable.

However the relative performance against cuBLAS falls from 51.93 percent to 29.84 percent. V3 does not become slower. Instead cuBLAS scales more effectively as K increases. cuBLAS throughput grows by approximately 2.30 times while V3 throughput grows by approximately 1.32 times.

The custom kernels mostly show near-linear execution-time growth with K. cuBLAS amortizes its fixed costs and uses deeper architecture-specific pipelining more effectively when the reduction dimension is large.

The larger absolute and relative errors at K equals 2048 are expected. Each output element performs eight times as many FP32 accumulation steps so floating-point rounding error increases. The current maximum relative error remains below the benchmark threshold of 1e-4.

## 6. Current V3 Limitations

### No Load-Compute Overlap

V3 still executes each K tile in three sequential phases load the tile synchronize the block compute the tile and synchronize again. It does not load the next tile while computing the current tile. At K equals 2048 the kernel processes thirty-two K tiles and reaches sixty-four block-wide barriers.

### High Register Pressure

Each thread holds sixteen output accumulators temporary A and B register fragments and multiple index variables. This can reduce the number of resident warps and blocks per SM. Actual register usage and occupancy should be measured with Nsight Compute.

### Large Shared-Memory Allocation

The two 64 x 64 float tiles require approximately 32 KiB of shared memory per block. Combined with register pressure this may limit concurrent block residency.

### Possible Shared-Memory Bank Conflicts

The B fragment access pattern uses columns separated by groups of four. Depending on the generated instructions and bank mapping this layout may create bank conflicts. Padding transposition or a swizzled shared-memory layout may reduce these conflicts.

### Incomplete Vector Tail Handling

The current float4 path writes zeros whenever fewer than four valid values remain. It does not perform a scalar fallback for the final one to three elements. The tested N and K values are divisible by four so the current experiments are valid but irregular shapes such as N equals 513 or K equals 257 can produce incorrect results.

### Scalar Output Stores

Results are written to C one float at a time. Four adjacent output values may be stored with float4 operations when alignment and boundary conditions permit.

### Fixed Kernel Configuration

The current 64 x 64 output tile 64-element K tile and 4 x 4 thread tile are fixed. They may not be optimal for every matrix shape or GPU architecture.

## 7. Proposed V4 Directions

### Priority 1 Correct Vector Tail Handling

Add a scalar fallback for one to three remaining elements and verify irregular dimensions such as M equals 1000 N equals 513 and K equals 257. Correctness for general shapes should be established before additional performance optimization.

### Priority 2 Profile the Current Kernel

Use Nsight Compute to measure achieved occupancy registers per thread shared-memory bank conflicts barrier stalls memory throughput and arithmetic pipeline utilization. V4 parameters should be selected from measured bottlenecks.

### Priority 3 Shared-Memory Layout Optimization

Test padding or transposing the B tile in shared memory. A layout that avoids bank conflicts can improve shared-memory bandwidth without changing the mathematical algorithm.

### Priority 4 Double Buffering

Use two shared-memory tile buffers. While the kernel computes the current tile it can prepare the next tile. On Ampere-class GPUs an advanced version can investigate asynchronous global-to-shared copies.

### Priority 5 Tune Register and K Tiles

Compare thread tiles such as 4 x 4 4 x 8 and 8 x 4 together with K tiles such as 16 32 and 64. Larger register tiles increase reuse but can reduce occupancy through register pressure.

### Priority 6 Vectorized Output Stores

When output columns are aligned and four consecutive values are valid store them through float4. Retain a scalar fallback for boundaries.

### Priority 7 Warp-Level Tiling

Assign well-defined subtiles to individual warps and organize register fragments around warp execution. This creates a clearer path toward architecture-aware SGEMM kernels.

## 8. Conclusion

V3 validates the main optimization hypothesis larger per-thread register tiles explicit register staging and vectorized global loads substantially improve SGEMM throughput. V3 reaches 2617.95 GFLOPS for K equals 256 and 3461.22 GFLOPS for K equals 2048.

The next version should first make vector access correct for arbitrary shapes then use profiling to choose between shared-memory layout changes double buffering tile-size tuning vectorized stores and warp-level tiling.
