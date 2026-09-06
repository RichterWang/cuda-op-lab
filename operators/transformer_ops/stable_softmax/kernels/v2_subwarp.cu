// v2: one row per lane group instead of one row per block.
//
// Motivation comes from a direct measurement, not from guessing. The 1-read vs
// 3-read probes gave t(3read)/t(1read) = 2.93 at cols=128 and 4.39 at cols=32.
// A ratio above 2.0 cannot be explained by traffic, since 4N/2N = 2 is the
// ceiling and a 512-byte row is certainly resident in L1. What the extra passes
// cost at these shapes is *critical path*: pass 2 cannot issue its loads until
// pass 1's reduction has resolved, and pass 3 waits on pass 2. With only 128
// elements per row there is not enough work inside a block to overlap those
// dependencies.
//
// The same probes also showed that at cols=32 a single-pass block-per-row kernel
// is already 4.5x slower than a plain copy (0.0161 ms vs 0.0036 ms). That gap is
// the mapping itself: 256 threads assigned to a 32-element row leaves 224 of them
// idle, and 8192 blocks each do 128 bytes of useful work.
//
// So v2 attacks both, with LanesPerRow as a template parameter:
//
//   * Shrink the group that owns a row to match the row: 8 lanes for cols=32,
//     32 lanes for cols=128. Idle lanes disappear and a 256-thread block now
//     retires 32 rows instead of 1, cutting the block count by the same factor.
//
//   * Drop __syncthreads and shared memory entirely. A group never spans a warp
//     boundary, so shuffles are the only communication needed. v1 still paid 2
//     block-wide barriers per row; v2 pays zero.
//
//   * Vectorize to float4 so each lane issues one wide load instead of four.
//
// v2b goes further: when cols == LanesPerRow * VecWidth, every lane's share of
// the row is exactly one float4 sitting in registers. Passes 2 and 3 then reuse
// those registers instead of re-reading, so traffic drops from 3 reads + 1 write
// to 1 read + 1 write, the theoretical minimum, and the critical path collapses
// to one load plus two reductions.
//
// Keeping v2a and v2b as separate entry points is deliberate: v2a isolates the
// gain from remapping and vectorizing, v2b adds the register reuse on top. Run
// together they would be indistinguishable.

#include <cuda_runtime.h>

#include <cfloat>

#include "stable_softmax.h"

namespace cuda_op_lab::stable_softmax {
namespace {

constexpr int kWarpSize = 32;
constexpr int kBlockSize = 256;

// Lane mask for one group.
//
// A full 0xffffffff mask would be wrong here. Groups smaller than a warp mean
// several of them share one warp, and the tail block can leave some of those
// groups inactive while others keep running, so the warp diverges. Passing a
// mask that names threads which have already returned is undefined behaviour.
//
// __activemask() would also "work" but its value depends on runtime convergence,
// which makes failures irreproducible. Computing the mask from lane position is
// exact and independent of control flow.
template <int LanesPerRow>
__device__ inline unsigned group_mask()
{
    if constexpr (LanesPerRow == kWarpSize)
    {
        return 0xffffffffu;  // 1u << 32 is undefined, so special-case it
    }
    else
    {
        const int lane_in_warp = threadIdx.x & (kWarpSize - 1);
        const int group_in_warp = lane_in_warp / LanesPerRow;
        constexpr unsigned base = (1u << LanesPerRow) - 1u;
        return base << (group_in_warp * LanesPerRow);
    }
}

// Segmented reduction with no extra work.
//
// Because LanesPerRow is a power of two and every offset stays strictly below
// it, `lane XOR offset` can never leave the aligned group the lane belongs to.
// The segmentation is therefore free: the identical loop body used for a full
// warp simply runs fewer iterations. v1's warp_reduce_max is the LanesPerRow=32
// case of this.
template <int LanesPerRow>
__device__ inline float group_reduce_max(float value, unsigned mask)
{
#pragma unroll
    for (int offset = LanesPerRow / 2; offset > 0; offset >>= 1) value = fmaxf(value, __shfl_xor_sync(mask, value, offset));
    return value;
}

template <int LanesPerRow>
__device__ inline float group_reduce_sum(float value, unsigned mask)
{
#pragma unroll
    for (int offset = LanesPerRow / 2; offset > 0; offset >>= 1) value += __shfl_xor_sync(mask, value, offset);
    return value;
}

// ---------------------------------------------------------------------------
// v2a: subwarp mapping + vectorization, still three passes over the row.
// ---------------------------------------------------------------------------
template <int LanesPerRow, int VecWidth>
__global__ void softmax_subwarp_kernel(const float *__restrict__ x, float *__restrict__ y, int rows, int cols)
{
    constexpr int kRowsPerBlock = kBlockSize / LanesPerRow;

    const int lane = threadIdx.x % LanesPerRow;
    const int row = blockIdx.x * kRowsPerBlock + static_cast<int>(threadIdx.x) / LanesPerRow;

    // Mask must be computed before the early return, while the whole warp is
    // still converged, because it only depends on threadIdx.
    const unsigned mask = group_mask<LanesPerRow>();
    if (row >= rows) return;

    const float *x_row = x + static_cast<size_t>(row) * cols;
    float *y_row = y + static_cast<size_t>(row) * cols;

    // pass 1: row max
    float thread_max = -FLT_MAX;
    if constexpr (VecWidth == 4)
    {
        // Only reached when cols % 4 == 0, so every row start is 16-byte
        // aligned given a cudaMalloc base. The dispatcher enforces this rather
        // than the kernel handling a tail, which keeps the hot loop free of
        // per-iteration bounds logic.
        const float4 *x4 = reinterpret_cast<const float4 *>(x_row);
        const int vec_cols = cols / 4;
        for (int index = lane; index < vec_cols; index += LanesPerRow)
        {
            const float4 value = x4[index];
            thread_max = fmaxf(thread_max, fmaxf(fmaxf(value.x, value.y), fmaxf(value.z, value.w)));
        }
    }
    else
    {
        for (int col = lane; col < cols; col += LanesPerRow) thread_max = fmaxf(thread_max, x_row[col]);
    }
    const float row_max = group_reduce_max<LanesPerRow>(thread_max, mask);

    // pass 2: denominator
    float thread_sum = 0.0f;
    if constexpr (VecWidth == 4)
    {
        const float4 *x4 = reinterpret_cast<const float4 *>(x_row);
        const int vec_cols = cols / 4;
        for (int index = lane; index < vec_cols; index += LanesPerRow)
        {
            const float4 value = x4[index];
            thread_sum += __expf(value.x - row_max) + __expf(value.y - row_max) + __expf(value.z - row_max) + __expf(value.w - row_max);
        }
    }
    else
    {
        for (int col = lane; col < cols; col += LanesPerRow) thread_sum += __expf(x_row[col] - row_max);
    }
    const float row_sum = group_reduce_sum<LanesPerRow>(thread_sum, mask);

    // pass 3: normalize. row_sum >= 1 because the max element contributes
    // exp(0) == 1, so the reciprocal needs no epsilon guard.
    const float inv_sum = 1.0f / row_sum;
    if constexpr (VecWidth == 4)
    {
        const float4 *x4 = reinterpret_cast<const float4 *>(x_row);
        float4 *y4 = reinterpret_cast<float4 *>(y_row);
        const int vec_cols = cols / 4;
        for (int index = lane; index < vec_cols; index += LanesPerRow)
        {
            const float4 value = x4[index];
            float4 result;
            result.x = __expf(value.x - row_max) * inv_sum;
            result.y = __expf(value.y - row_max) * inv_sum;
            result.z = __expf(value.z - row_max) * inv_sum;
            result.w = __expf(value.w - row_max) * inv_sum;
            y4[index] = result;
        }
    }
    else
    {
        for (int col = lane; col < cols; col += LanesPerRow) y_row[col] = __expf(x_row[col] - row_max) * inv_sum;
    }
}

// ---------------------------------------------------------------------------
// v2b: the row lives in registers, so it is read exactly once.
//
// Requires cols == LanesPerRow * 4, checked by the dispatcher. Cost is 4
// registers per thread for the values plus 4 for the exponentials; the
// occupancy probe measured 38 registers for the v1-shaped kernel against a
// budget of 42 (65536 / 1536) for full occupancy, so this stays affordable.
// ---------------------------------------------------------------------------
template <int LanesPerRow>
__global__ void softmax_subwarp_register_kernel(const float *__restrict__ x, float *__restrict__ y, int rows, int cols)
{
    constexpr int kRowsPerBlock = kBlockSize / LanesPerRow;

    const int lane = threadIdx.x % LanesPerRow;
    const int row = blockIdx.x * kRowsPerBlock + static_cast<int>(threadIdx.x) / LanesPerRow;

    const unsigned mask = group_mask<LanesPerRow>();
    if (row >= rows) return;

    const float4 *x4 = reinterpret_cast<const float4 *>(x + static_cast<size_t>(row) * cols);
    float4 *y4 = reinterpret_cast<float4 *>(y + static_cast<size_t>(row) * cols);

    // The one and only read of this row.
    const float4 value = x4[lane];

    const float row_max = group_reduce_max<LanesPerRow>(fmaxf(fmaxf(value.x, value.y), fmaxf(value.z, value.w)), mask);

    // Hold the exponentials so pass 3 does not recompute them. Without this the
    // kernel would call __expf twice per element for no reason.
    float4 exponential;
    exponential.x = __expf(value.x - row_max);
    exponential.y = __expf(value.y - row_max);
    exponential.z = __expf(value.z - row_max);
    exponential.w = __expf(value.w - row_max);

    const float row_sum =
        group_reduce_sum<LanesPerRow>(exponential.x + exponential.y + exponential.z + exponential.w, mask);
    const float inv_sum = 1.0f / row_sum;

    float4 result;
    result.x = exponential.x * inv_sum;
    result.y = exponential.y * inv_sum;
    result.z = exponential.z * inv_sum;
    result.w = exponential.w * inv_sum;
    y4[lane] = result;
}

template <int LanesPerRow>
inline int grid_for(int rows)
{
    constexpr int kRowsPerBlock = kBlockSize / LanesPerRow;
    return (rows + kRowsPerBlock - 1) / kRowsPerBlock;
}

}  // namespace

void launch_softmax_subwarp(const float *x, float *y, int rows, int cols, cudaStream_t stream)
{
    if (rows <= 0 || cols <= 0) return;

    // Above this width the group would have to loop many times and the
    // block-per-row versions already reach 93-96% of the copy ceiling, so there
    // is nothing to win. Hand off rather than pretend to cover every shape.
    if (cols > 256)
    {
        launch_softmax_adaptive(x, y, rows, cols, stream);
        return;
    }

    // cols % 4 != 0 is dispatched to a scalar instantiation instead of adding a
    // tail path inside the vectorized kernel. Two clean kernels beat one kernel
    // with two code paths, and it removes the class of bug where the tail is
    // simply forgotten.
    if (cols % 4 == 0)
    {
        const int vec_cols = cols / 4;
        if (vec_cols <= 8)
            softmax_subwarp_kernel<8, 4><<<grid_for<8>(rows), kBlockSize, 0, stream>>>(x, y, rows, cols);
        else if (vec_cols <= 16)
            softmax_subwarp_kernel<16, 4><<<grid_for<16>(rows), kBlockSize, 0, stream>>>(x, y, rows, cols);
        else
            softmax_subwarp_kernel<32, 4><<<grid_for<32>(rows), kBlockSize, 0, stream>>>(x, y, rows, cols);
    }
    else
    {
        if (cols <= 8)
            softmax_subwarp_kernel<8, 1><<<grid_for<8>(rows), kBlockSize, 0, stream>>>(x, y, rows, cols);
        else if (cols <= 16)
            softmax_subwarp_kernel<16, 1><<<grid_for<16>(rows), kBlockSize, 0, stream>>>(x, y, rows, cols);
        else
            softmax_subwarp_kernel<32, 1><<<grid_for<32>(rows), kBlockSize, 0, stream>>>(x, y, rows, cols);
    }
}

void launch_softmax_subwarp_register(const float *x, float *y, int rows, int cols, cudaStream_t stream)
{
    if (rows <= 0 || cols <= 0) return;

    // Register residency needs cols to match LanesPerRow * 4 exactly. Anything
    // else falls back to v2a, which keeps this launcher usable as a general
    // entry point while the fast path stays branch-free.
    switch (cols)
    {
        case 32: softmax_subwarp_register_kernel<8><<<grid_for<8>(rows), kBlockSize, 0, stream>>>(x, y, rows, cols); return;
        case 64: softmax_subwarp_register_kernel<16><<<grid_for<16>(rows), kBlockSize, 0, stream>>>(x, y, rows, cols); return;
        case 128: softmax_subwarp_register_kernel<32><<<grid_for<32>(rows), kBlockSize, 0, stream>>>(x, y, rows, cols); return;
        default: launch_softmax_subwarp(x, y, rows, cols, stream); return;
    }
}

}  // namespace cuda_op_lab::stable_softmax
