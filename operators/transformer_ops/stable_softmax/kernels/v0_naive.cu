#include <cuda_runtime.h>

#include <cfloat>

#include "stable_softmax.h"

namespace cuda_op_lab::stable_softmax {
namespace {

constexpr int kBlockSize = 256;

// Block-wide max reduction over shared memory, call by each thread
__device__ float block_reduce_max(float value, float* scratch)
{
    const int tid = threadIdx.x;
    scratch[tid] = value;
    __syncthreads();

    // Tree reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride) scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
        __syncthreads();
    }

    // so boardcast to each thread
    return scratch[0];
}

__device__ float block_reduce_sum(float value, float* scratch)
{
    const int tid = threadIdx.x;
    scratch[tid] = value;
    __syncthreads();

    // Tree sum_up
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

    // caculate by x_row, x_row fixed x and y offset
    const float* x_row = x + static_cast<size_t>(row) * cols;
    float* y_row = y + static_cast<size_t>(row) * cols;

    __shared__ float scratch[kBlockSize];

    // get max value per row
    float thread_max = -FLT_MAX;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) thread_max = fmaxf(thread_max, x_row[col]);

    const float row_max = block_reduce_max(thread_max, scratch);
    __syncthreads();

    // get the denominator of the final project
    float thread_sum = 0.0f;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) thread_sum += __expf(x_row[col] - row_max);

    const float row_sum = block_reduce_sum(thread_sum, scratch);
    __syncthreads();

    // final normalize and write back.
    const float inv_sum = 1.0f / row_sum;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) y_row[col] = __expf(x_row[col] - row_max) * inv_sum;
}

}  // namespace

void launch_softmax_naive(const float* x, float* y, int rows, int cols, cudaStream_t stream)
{
    if (rows <= 0 || cols <= 0) return;

    // Launch one block per row, with kBlockSize threads per block. Each thread handles multiple columns in a grid-stride loop.
    const dim3 block(kBlockSize);
    const dim3 grid(rows);

    // 0 represent the dynamic shared memory size of each block.
    softmax_naive_kernel<<<grid, block, 0, stream>>>(x, y, rows, cols);
}

}  // namespace cuda_op_lab::stable_softmax
