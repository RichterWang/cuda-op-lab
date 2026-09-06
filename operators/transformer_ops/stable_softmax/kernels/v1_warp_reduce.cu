// v1: same structure as v0 (one block per row, three passes over the row),
// with the ONLY change being how the two reductions are done.
//
// v0 used a shared-memory tree reduction: log2(256) = 8 rounds per reduction,
// each round with a __syncthreads() and a shared-memory round trip, so 16
// block-wide barriers per row in total.
//
// v1 reduces within each warp using __shfl_xor_sync (5 register-only steps, no
// barrier at all), then combines the per-warp partials through a handful of
// floats in shared memory. That is 1 barrier per reduction, 2 per row.
//
// Keeping everything else identical is deliberate: it isolates the barrier cost
// so the speedup can be attributed to the reduction change alone.
//
// The kernel is templated on block size so the same code can be launched with
// 256 or 1024 threads. That second configuration is an experiment for long
// rows: cols=8192 measured ~433 GB/s of real DRAM traffic against a 413 GB/s
// copy ceiling, i.e. bandwidth is already saturated and the deficit comes from
// moving 2x the necessary bytes because passes 2 and 3 miss L1. A bigger block
// means fewer rows resident per SM at once, so each row gets a larger share of
// L1 and may keep its data across the three passes. Same instruction count,
// potentially less DRAM traffic.

#include <cuda_runtime.h>

#include <cfloat>

#include "stable_softmax.h"

namespace cuda_op_lab::stable_softmax {
namespace {

constexpr int kWarpSize = 32;
constexpr unsigned kFullMask = 0xffffffffu;

// Butterfly (xor) shuffle: after 5 steps every lane holds the warp-wide result,
// so no extra broadcast step is needed.
__device__ inline float warp_reduce_max(float value)
{
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) value = fmaxf(value, __shfl_xor_sync(kFullMask, value, offset));
    return value;
}

__device__ inline float warp_reduce_sum(float value)
{
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) value += __shfl_xor_sync(kFullMask, value, offset);
    return value;
}

template <int BlockSize>
__global__ void softmax_warp_reduce_kernel(const float *__restrict__ x, float *__restrict__ y, int rows, int cols)
{
    constexpr int kWarpsPerBlock = BlockSize / kWarpSize;

    const int row = blockIdx.x;
    if (row >= rows) return;

    const float *x_row = x + static_cast<size_t>(row) * cols;
    float *y_row = y + static_cast<size_t>(row) * cols;

    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp_id = threadIdx.x >> 5;

    // Two separate buffers so the sum reduction can write without waiting for
    // every thread to finish reading the max results.
    __shared__ float warp_max[kWarpsPerBlock];
    __shared__ float warp_sum[kWarpsPerBlock];

    // pass 1: row max
    float thread_max = -FLT_MAX;
    for (int col = threadIdx.x; col < cols; col += BlockSize) thread_max = fmaxf(thread_max, x_row[col]);

    thread_max = warp_reduce_max(thread_max);
    if (lane == 0) warp_max[warp_id] = thread_max;
    __syncthreads();

    // Every thread folds the partials itself. All threads read the same
    // addresses, which broadcasts rather than conflicting, and it avoids a
    // second barrier that a reduce-then-broadcast scheme would need.
    float row_max = warp_max[0];
#pragma unroll
    for (int index = 1; index < kWarpsPerBlock; ++index) row_max = fmaxf(row_max, warp_max[index]);

    // pass 2: denominator over the shifted values, so exp() cannot overflow
    float thread_sum = 0.0f;
    for (int col = threadIdx.x; col < cols; col += BlockSize) thread_sum += __expf(x_row[col] - row_max);

    thread_sum = warp_reduce_sum(thread_sum);
    if (lane == 0) warp_sum[warp_id] = thread_sum;
    __syncthreads();

    float row_sum = warp_sum[0];
#pragma unroll
    for (int index = 1; index < kWarpsPerBlock; ++index) row_sum += warp_sum[index];

    // pass 3: normalize. row_sum >= 1 because the max element contributes
    // exp(0) == 1, so no epsilon guard is needed on the reciprocal.
    const float inv_sum = 1.0f / row_sum;
    for (int col = threadIdx.x; col < cols; col += BlockSize) y_row[col] = __expf(x_row[col] - row_max) * inv_sum;
}

}  // namespace

void launch_softmax_warp_reduce(const float *x, float *y, int rows, int cols, cudaStream_t stream)
{
    if (rows <= 0 || cols <= 0) return;

    constexpr int kBlockSize = 256;
    softmax_warp_reduce_kernel<kBlockSize><<<dim3(rows), dim3(kBlockSize), 0, stream>>>(x, y, rows, cols);
}

void launch_softmax_warp_reduce_big_block(const float *x, float *y, int rows, int cols, cudaStream_t stream)
{
    if (rows <= 0 || cols <= 0) return;

    constexpr int kBlockSize = 1024;
    softmax_warp_reduce_kernel<kBlockSize><<<dim3(rows), dim3(kBlockSize), 0, stream>>>(x, y, rows, cols);
}

// The 1024-thread experiment showed block size is not a global choice: it took
// cols=8192 from 52% to 96% of the copy ceiling, and simultaneously took
// cols=128 from 29% down to 3%. Long rows want a big block so fewer rows are
// resident per SM and each row's data survives L1 across the three passes.
// Short rows want a small block because a big one leaves most threads idle and
// pays block-wide barriers for almost no work.
//
// So dispatch on cols. The thresholds aim for roughly 8 elements per thread,
// which is what put cols=8192 at 1024 threads and cols=1024 at 256.
void launch_softmax_adaptive(const float *x, float *y, int rows, int cols, cudaStream_t stream)
{
    if (rows <= 0 || cols <= 0) return;

    const dim3 grid(rows);

    if (cols >= 4096)
        softmax_warp_reduce_kernel<1024><<<grid, dim3(1024), 0, stream>>>(x, y, rows, cols);
    else if (cols >= 2048)
        softmax_warp_reduce_kernel<512><<<grid, dim3(512), 0, stream>>>(x, y, rows, cols);
    else
        softmax_warp_reduce_kernel<256><<<grid, dim3(256), 0, stream>>>(x, y, rows, cols);
}

}  // namespace cuda_op_lab::stable_softmax
