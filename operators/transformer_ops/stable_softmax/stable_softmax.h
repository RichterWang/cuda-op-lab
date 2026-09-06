#pragma once

#include <cuda_runtime.h>

namespace cuda_op_lab::stable_softmax {

// Row-wise numerically stable softmax.
//
// x, y: device pointers to [rows, cols] row-major fp32 buffers.
// Each row of `cols` elements is normalized independently:
//     m      = max(x_row)
//     y_row  = exp(x_row - m) / sum(exp(x_row - m))
//
// Subtracting the row max is what keeps exp() from overflowing when the inputs
// carry a large positive shift, and keeps the denominator non-zero when they
// carry a large negative shift.
//
// y may alias x (in-place is allowed for the current kernels).

// v0: one block per row, shared-memory tree reduction, three passes over the row.
void launch_softmax_naive(const float* x, float* y, int rows, int cols, cudaStream_t stream = nullptr);

// v1: same as v0, but the two reductions use __shfl_xor_sync within each warp
// and only one __syncthreads() each to combine the per-warp partials.
void launch_softmax_warp_reduce(const float* x, float* y, int rows, int cols, cudaStream_t stream = nullptr);

// v1 with a 1024-thread block instead of 256. Experiment for long rows: fewer
// rows resident per SM means each row gets a larger share of L1, which may keep
// its data alive across the three passes and cut real DRAM traffic.
void launch_softmax_warp_reduce_big_block(const float* x, float* y, int rows, int cols, cudaStream_t stream = nullptr);

// v1 with block size chosen from cols. The big-block experiment showed the two
// row-length regimes want opposite settings, so neither constant is right on
// its own.
void launch_softmax_adaptive(const float* x, float* y, int rows, int cols, cudaStream_t stream = nullptr);

// v2a: one row per lane group (8/16/32 lanes) instead of one row per block, with
// float4 loads. No __syncthreads and no shared memory, since a group never spans
// a warp boundary. Still three passes over the row. Targets cols <= 256; wider
// rows are forwarded to launch_softmax_adaptive.
void launch_softmax_subwarp(const float* x, float* y, int rows, int cols, cudaStream_t stream = nullptr);

// v2b: v2a plus register residency. When cols == LanesPerRow * 4 each lane owns
// exactly one float4 of the row, so passes 2 and 3 reuse registers and the row is
// read once. That is the minimum possible traffic (1 read + 1 write). Shapes that
// do not fit fall back to v2a.
void launch_softmax_subwarp_register(const float* x, float* y, int rows, int cols, cudaStream_t stream = nullptr);

// v3: online softmax. Fuses the max pass and the sum pass into one by carrying a
// (max, sum) pair through the reduction, so the row is read twice instead of
// three times. The combine operator
//     (m1,l1) (+) (m2,l2) -> ( max(m1,m2), l1*exp(m1-m) + l2*exp(m2-m) )
// is associative, which is what lets it ride the same shuffle reduction tree a
// plain sum would use instead of only working as a serial recurrence.
//
// Keeps the block-per-row mapping and the block-size thresholds of
// launch_softmax_adaptive, so comparing the two isolates the pass count.
//
// Worth it only once the row stops fitting in L1: 31% slower than v1 at
// cols=8192, 20% faster at cols=32768, where it is the only version that gets
// past the ~223 GB/s wall the three-pass kernels share. Fusing costs extra
// instructions per element (a rescale branch, and a two-value reduction tree),
// and that cost only pays off when the removed pass was really hitting DRAM.
void launch_softmax_online(const float* x, float* y, int rows, int cols, cudaStream_t stream = nullptr);

// v3b: v3 with four independent running pairs per thread instead of one. Same
// traffic, same number of exp() calls, four times shorter dependency chain.
//
// Kept as a recorded negative result. It was written to test the theory that v3
// lost to v1 because fusing moved exp() into the loop-carried dependency, so
// shortening the chain should have recovered the gap. It does not: v3b is
// consistently slower than v3 at every shape. The chain was not the problem, the
// added per-element instructions were.
void launch_softmax_online_ilp(const float* x, float* y, int rows, int cols, cudaStream_t stream = nullptr);

// Not a softmax: plain y = x copy, used as the achievable-bandwidth roofline.
// It moves the same traffic an ideal softmax would (one read + one write per
// element), so its GB/s is the real ceiling for the kernels above.
void launch_copy_baseline(const float* x, float* y, int rows, int cols, cudaStream_t stream = nullptr);

// Traffic probes, not softmax. Same block-per-row mapping and dependency chain
// as v1, but with the exp() removed, so the only difference between them is how
// many times the row is read. Comparing their runtimes measures directly how
// much of the extra traffic actually reaches DRAM, which the bandwidth numbers
// alone can only suggest.
void launch_probe_1read(const float* x, float* y, int rows, int cols, cudaStream_t stream = nullptr);
void launch_probe_3read(const float* x, float* y, int rows, int cols, cudaStream_t stream = nullptr);

// Same probes with a 1024-thread block, matching what launch_softmax_adaptive
// picks for long rows. The 256-thread pair measures traffic under the settings
// v1 shipped with; this pair measures whether any DRAM traffic is left to remove
// once a large block has already pulled passes 2 and 3 into L1. That is exactly
// the headroom a two-pass (online) algorithm could claim.
void launch_probe_1read_big_block(const float* x, float* y, int rows, int cols, cudaStream_t stream = nullptr);
void launch_probe_3read_big_block(const float* x, float* y, int rows, int cols, cudaStream_t stream = nullptr);

}  // namespace cuda_op_lab::stable_softmax
