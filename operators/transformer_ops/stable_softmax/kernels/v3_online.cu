// v3: online softmax. Fuses the max pass and the sum pass into one, cutting the
// row from three passes to two (read, then read + write).
//
// The trick is that the running pair (max, sum-of-exp-relative-to-that-max) can
// be combined associatively:
//
//   (m1, l1) (+) (m2, l2)  ->  m = max(m1, m2)
//                              l = l1 * exp(m1 - m) + l2 * exp(m2 - m)
//
// Each l is rescaled to the new shared max before being added. Associativity is
// what matters here: it means the pair can go through a shuffle reduction tree
// exactly like a plain sum, instead of only being updated serially element by
// element. Serial recurrence alone would be useless on a GPU.
//
// Numerical stability is preserved. Every exp() argument is <= 0 because m is the
// max of the two, so no term can overflow, and the identity element (-inf, 0)
// keeps empty partials from contributing.
//
// What this version does NOT change: the block-per-row mapping, the block sizes,
// and the fact that the output pass has to read the row again. Only the pass
// count changes, which is exactly what makes it a clean measurement of "does
// removing one pass help?"
//
// Measured answer: it depends entirely on whether the row fits in L1, and the
// split is sharp.
//
//   cols=8192  (32 KB row, fits)     v1 387 GB/s   v3 266 GB/s   -31%
//   cols=32768 (128 KB row, does not) v1 223 GB/s   v3 269 GB/s   +20%
//
// The traffic probes explain both halves. At cols=8192 with 1024 threads the
// probe ratio 3read/1read is 1.04: a large block already keeps the row in L1
// across all three passes, so the pass v3 removes costs almost nothing, and
// paying extra instructions for it is a straight loss. At cols=32768 the ratio
// is 1.83: the row cannot stay resident, the third pass really does go to DRAM,
// and removing it wins. v3 is the only version that breaks past ~223 GB/s at
// that shape, which is where every three-pass variant here piles up.
//
// So the cost of fusing is real but fixed, and the benefit scales with row
// length. What makes the fused pass more expensive per element than v1's two
// separate passes:
//   - v1's sum pass has the row max as a loop-invariant, so all its exp() calls
//     are independent. v3's running sum must be rescaled whenever a new max
//     appears, adding a branch and a possible second exp() per element.
//   - the reduction tree carries two values and calls combine at every step
//     (2 shuffles + 2 exp + fmax + 2 mul + add) instead of doing two plain
//     shuffle reductions. Short rows are dominated by this, which is why
//     cols=32 is where v3 looks worst.
//
// The dependency chain is NOT the main cost, which is worth recording because it
// was the first guess. launch_softmax_online_ilp below gives each thread four
// independent chains, and it is consistently slower than v3, not faster. The
// extra accumulators sit in the frequently-rescaling regime and burn registers
// for nothing.
//
// The combine operator is kept as a standalone, reusable device function because
// it is the same primitive flash attention uses to merge partial attention
// blocks, where the tiles never fit in L1 and the two-pass form is the only
// option. That reuse, plus the long-row win above, is why this version stays.

#include <cuda_runtime.h>

#include <cfloat>

#include "stable_softmax.h"

namespace cuda_op_lab::stable_softmax {
namespace {

constexpr int kWarpSize = 32;
constexpr unsigned kFullMask = 0xffffffffu;

// Running (max, sum of exp relative to max) for some subset of a row.
struct MaxSum
{
    float max;
    float sum;
};

// Identity element for the combine below: contributes nothing.
__device__ inline MaxSum max_sum_identity()
{
    return MaxSum{-FLT_MAX, 0.0f};
}

// Fold one raw element into a running pair. This is the serial update.
__device__ inline MaxSum max_sum_accumulate(MaxSum state, float value)
{
    if (value <= state.max)
    {
        // Common case once a few elements are in: no rescale needed.
        state.sum += __expf(value - state.max);
        return state;
    }

    // New max, so the accumulated sum has to be rescaled onto it.
    state.sum = state.sum * __expf(state.max - value) + 1.0f;
    state.max = value;
    return state;
}

// The associative combine. Both branches rescale onto the larger max, so every
// exp() argument is <= 0.
__device__ inline MaxSum max_sum_combine(MaxSum a, MaxSum b)
{
    const float merged_max = fmaxf(a.max, b.max);

    // A pair that never saw an element stays neutral: exp(-FLT_MAX - m) would
    // underflow to 0 anyway, but skipping it avoids relying on that.
    const float scaled_a = (a.sum > 0.0f) ? a.sum * __expf(a.max - merged_max) : 0.0f;
    const float scaled_b = (b.sum > 0.0f) ? b.sum * __expf(b.max - merged_max) : 0.0f;

    return MaxSum{merged_max, scaled_a + scaled_b};
}

// Reduce a pair across a full warp using the same xor-shuffle pattern a plain
// sum would use. This is only valid because combine is associative.
__device__ inline MaxSum warp_reduce_max_sum(MaxSum state)
{
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1)
    {
        MaxSum other;
        other.max = __shfl_xor_sync(kFullMask, state.max, offset);
        other.sum = __shfl_xor_sync(kFullMask, state.sum, offset);
        state = max_sum_combine(state, other);
    }
    return state;
}

template <int BlockSize>
__global__ void softmax_online_kernel(const float *__restrict__ x, float *__restrict__ y, int rows, int cols)
{
    constexpr int kWarpsPerBlock = BlockSize / kWarpSize;

    const int row = blockIdx.x;
    if (row >= rows) return;

    const float *x_row = x + static_cast<size_t>(row) * cols;
    float *y_row = y + static_cast<size_t>(row) * cols;

    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp_id = threadIdx.x >> 5;

    __shared__ float warp_max[kWarpsPerBlock];
    __shared__ float warp_sum[kWarpsPerBlock];

    // Pass 1: one sweep produces both the row max and the row sum. v1 needed two
    // sweeps for this.
    MaxSum state = max_sum_identity();
    for (int col = threadIdx.x; col < cols; col += BlockSize) state = max_sum_accumulate(state, x_row[col]);

    state = warp_reduce_max_sum(state);

    if (lane == 0)
    {
        warp_max[warp_id] = state.max;
        warp_sum[warp_id] = state.sum;
    }
    __syncthreads();

    // Combine the per-warp pairs. Same operator, so the merge is a fold rather
    // than a special case.
    MaxSum row_state{warp_max[0], warp_sum[0]};
#pragma unroll
    for (int index = 1; index < kWarpsPerBlock; ++index) row_state = max_sum_combine(row_state, MaxSum{warp_max[index], warp_sum[index]});

    // Pass 2: normalize. The row has to be read once more because the earlier
    // values were not kept anywhere.
    const float inverse_sum = 1.0f / row_state.sum;
    for (int col = threadIdx.x; col < cols; col += BlockSize) y_row[col] = __expf(x_row[col] - row_state.max) * inverse_sum;
}

// v3b: identical traffic and identical arithmetic to the kernel above, with one
// change. Each thread keeps four independent running pairs instead of one, so
// the fused pass has four independent dependency chains instead of one.
//
// This exists to test a specific explanation for v3 being slower than v1: that
// fusing the passes put exp() inside the loop-carried dependency, and the cost
// of serializing exp latency exceeds the cost of the load it removed. If that is
// right, shortening the chain by 4x should recover most of the gap without
// touching a single byte of traffic. If v3b is no faster, the explanation is
// wrong and the cost lies somewhere else.
template <int BlockSize, int Accumulators>
__global__ void softmax_online_ilp_kernel(const float *__restrict__ x, float *__restrict__ y, int rows, int cols)
{
    constexpr int kWarpsPerBlock = BlockSize / kWarpSize;
    constexpr int kStride = BlockSize * Accumulators;

    const int row = blockIdx.x;
    if (row >= rows) return;

    const float *x_row = x + static_cast<size_t>(row) * cols;
    float *y_row = y + static_cast<size_t>(row) * cols;

    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp_id = threadIdx.x >> 5;

    __shared__ float warp_max[kWarpsPerBlock];
    __shared__ float warp_sum[kWarpsPerBlock];

    MaxSum state[Accumulators];
#pragma unroll
    for (int slot = 0; slot < Accumulators; ++slot) state[slot] = max_sum_identity();

    // Each slot advances by BlockSize, so every individual load is still a
    // fully coalesced BlockSize-wide access. Only the accumulator changes.
    for (int base = threadIdx.x; base < cols; base += kStride)
    {
#pragma unroll
        for (int slot = 0; slot < Accumulators; ++slot)
        {
            const int col = base + slot * BlockSize;
            if (col < cols) state[slot] = max_sum_accumulate(state[slot], x_row[col]);
        }
    }

    // Fold the private chains together, then reduce as before.
    MaxSum thread_state = state[0];
#pragma unroll
    for (int slot = 1; slot < Accumulators; ++slot) thread_state = max_sum_combine(thread_state, state[slot]);

    thread_state = warp_reduce_max_sum(thread_state);

    if (lane == 0)
    {
        warp_max[warp_id] = thread_state.max;
        warp_sum[warp_id] = thread_state.sum;
    }
    __syncthreads();

    MaxSum row_state{warp_max[0], warp_sum[0]};
#pragma unroll
    for (int index = 1; index < kWarpsPerBlock; ++index) row_state = max_sum_combine(row_state, MaxSum{warp_max[index], warp_sum[index]});

    const float inverse_sum = 1.0f / row_state.sum;
    for (int col = threadIdx.x; col < cols; col += BlockSize) y_row[col] = __expf(x_row[col] - row_state.max) * inverse_sum;
}

}  // namespace

void launch_softmax_online(const float *x, float *y, int rows, int cols, cudaStream_t stream)
{
    if (rows <= 0 || cols <= 0) return;

    // Same thresholds as launch_softmax_adaptive, so a v1-vs-v3 comparison
    // isolates the pass count rather than mixing in a block-size change.
    if (cols >= 4096)
        softmax_online_kernel<1024><<<dim3(rows), dim3(1024), 0, stream>>>(x, y, rows, cols);
    else if (cols >= 2048)
        softmax_online_kernel<512><<<dim3(rows), dim3(512), 0, stream>>>(x, y, rows, cols);
    else
        softmax_online_kernel<256><<<dim3(rows), dim3(256), 0, stream>>>(x, y, rows, cols);
}

void launch_softmax_online_ilp(const float *x, float *y, int rows, int cols, cudaStream_t stream)
{
    if (rows <= 0 || cols <= 0) return;

    constexpr int kAccumulators = 4;

    if (cols >= 4096)
        softmax_online_ilp_kernel<1024, kAccumulators><<<dim3(rows), dim3(1024), 0, stream>>>(x, y, rows, cols);
    else if (cols >= 2048)
        softmax_online_ilp_kernel<512, kAccumulators><<<dim3(rows), dim3(512), 0, stream>>>(x, y, rows, cols);
    else
        softmax_online_ilp_kernel<256, kAccumulators><<<dim3(rows), dim3(256), 0, stream>>>(x, y, rows, cols);
}

}  // namespace cuda_op_lab::stable_softmax
