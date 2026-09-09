#include <cuda_runtime.h>

#include <cfloat>

#include "stable_softmax.h"

namespace cuda_op_lab::stable_softmax {
namespace {

constexpr int kWarpSize = 32;
constexpr unsigned kFullMask = 0xffffffffu;

// xor shffle per thread, no extra boardcast needed
__device__ __forceinline__ float warp_reduce_sum(float value)
{
    #pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) value += __shfl_xor_sync(kFullMask, value, offset);
    return value;
}

__device__ __forceinline__ float warp_reduce_min(float value)
{
    #pragma unroll
    for(int i = kWarpSize / 2; i > 0; i >>=1) value = fmax(value, __shfl_xor_sync(kFullMask, value, i));
    return value;
}

template <int BlockSize>
__global__ void softmax_warp_reduce_kernel(const float *__restrict__ x, float *__restrict__ y, int rows, int cols)
{
    constexpr int kWarpsPerBlock = BlockSize / kWarpSize; // can get num when compiling, warp per block(row)

    const int row = blockIdx.x;
    if (row >= rows) return;

    const float *x_row = x + static_cast<size_t>(row) * cols;
    float *y_row = y + static_cast<size_t>(row) * cols;

    // define warp parameters
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp_id = threadIdx.x >> 5;

    // Two buffer can avoid sync and wait
    __shared__ float warp_max[kWarpsPerBlock];
    __shared__ float warp_sum[kWarpsPerBlock];

    // caculate row max
    float thread_max = -FLT_MAX;
    for (int col = threadIdx.x; col < cols; col += BlockSize) thread_max = fmaxf(thread_max, x_row[col]);

    thread_max = warp_reduce_max(thread_max);
    if (lane == 0) warp_max[warp_id] = thread_max; // lane get by frame, warp lane(32)
    __syncthreads();

    // collect warp max to get row max
    float row_max = warp_max[0];
    #pragma unroll
    for (int index = 1; index < kWarpsPerBlock; ++index) row_max = fmaxf(row_max, warp_max[index]);

    // calculate denominator
    float thread_sum = 0.0f;
    for (int col = threadIdx.x; col < cols; col += BlockSize) thread_sum += __expf(x_row[col] - row_max);

    thread_sum = warp_reduce_sum(thread_sum);
    if (lane == 0) warp_sum[warp_id] = thread_sum;
    __syncthreads();

    float row_sum = warp_sum[0];
    #pragma unroll
    for (int index = 1; index < kWarpsPerBlock; ++index) row_sum += warp_sum[index];

    // normalize and wrote back
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
