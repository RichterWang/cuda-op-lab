// v0 baseline: row-wise stable softmax, one block per row.
//
// Strategy (deliberately simple, this is the reference point for later versions):
//   pass 1: read the row from global memory, block-reduce the max
//   pass 2: read the row again, block-reduce sum(exp(x - max))
//   pass 3: read the row a third time, write exp(x - max) / sum
//
// So global traffic is 3 reads + 1 write per element, while the theoretical
// minimum is 1 read + 1 write. That 2x excess is exactly what v1+ will remove
// (keep the row in registers / shared memory, warp-shuffle reduction, or the
// online softmax single-pass formulation).

#include <cuda_runtime.h>

#include <cfloat>

#include "stable_softmax.h"

namespace cuda_op_lab::stable_softmax {
namespace {

constexpr int kBlockSize = 256;

// Block-wide max reduction over shared memory. Every thread must call this.
__device__ float block_reduce_max(float value, float* scratch)
{
    const int tid = threadIdx.x;
    scratch[tid] = value;
    __syncthreads();

    // Tree reduction; kBlockSize is a power of two so no odd-size handling needed.
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride) scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
        __syncthreads();
    }

    // Broadcast: every thread reads the same slot, no bank conflict (same address).
    return scratch[0];
}

__device__ float block_reduce_sum(float value, float* scratch)
{
    const int tid = threadIdx.x;
    scratch[tid] = value;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride) scratch[tid] += scratch[tid + stride];
        __syncthreads();
    }

    return scratch[0];
}

__global__ void softmax_naive_kernel(const float* __restrict__ x, float* __restrict__ y, int rows, int cols)
{
    const int row = blockIdx.x;
    if (row >= rows) return;

    const float* x_row = x + static_cast<size_t>(row) * cols;
    float* y_row = y + static_cast<size_t>(row) * cols;

    __shared__ float scratch[kBlockSize];

    // pass 1: row max. Grid-stride style loop so cols can exceed blockDim.x.
    float thread_max = -FLT_MAX;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) thread_max = fmaxf(thread_max, x_row[col]);

    const float row_max = block_reduce_max(thread_max, scratch);
    __syncthreads();  // scratch is about to be reused by the sum reduction

    // pass 2: denominator, computed on the shifted values so exp() cannot overflow.
    float thread_sum = 0.0f;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) thread_sum += __expf(x_row[col] - row_max);

    const float row_sum = block_reduce_sum(thread_sum, scratch);
    __syncthreads();

    // pass 3: normalize and write back.
    // row_sum >= 1 always, because the max element contributes exp(0) == 1,
    // so the reciprocal is safe without any epsilon guard.
    const float inv_sum = 1.0f / row_sum;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) y_row[col] = __expf(x_row[col] - row_max) * inv_sum;
}

}  // namespace

void launch_softmax_naive(const float* x, float* y, int rows, int cols, cudaStream_t stream)
{
    if (rows <= 0 || cols <= 0) return;

    const dim3 block(kBlockSize);
    const dim3 grid(rows);

    softmax_naive_kernel<<<grid, block, 0, stream>>>(x, y, rows, cols);
}

}  // namespace cuda_op_lab::stable_softmax
