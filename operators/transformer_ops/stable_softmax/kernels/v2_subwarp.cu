#include <cuda_runtime.h>

#include <cfloat>

#include "stable_softmax.h"

namespace cuda_op_lab::stable_softmax {
namespace {

constexpr int kWarpSize = 32;
constexpr int kBlockSize = 256;

// define the mask according to lanes
template <int ThreadsPerRow>
__device__ inline unsigned group_mask()
{
    if constexpr (ThreadsPerRow == kWarpSize)
    {
        return 0xffffffffu; //return all threads
    }
    else
    {
        const int threadIdx_in_warp = threadIdx.x & (kWarpSize - 1);
        const int group_in_warp = threadIdx_in_warp / ThreadsPerRow;
        constexpr unsigned base_mask = (1u << ThreadsPerRow) - 1u;
        return base_mask << (group_in_warp * ThreadsPerRow);
    }
}

// masked reduce
template <int ThreadsPerRow>
__device__ __forceinline__ float group_reduce_max(float value, unsigned mask)
{
    #pragma unroll
    for(int offset = ThreadsPerRow / 2; offset > 0; offset >>= 1) value = fmaxf(value, __shfl_xor_sync(mask, value, offset));
    return value;
}

template <int ThreadsPerRow>
__device__ __forceinline__ float group_reduce_sum(float value, unsigned mask)
{
    #pragma unroll
    for (int offset = ThreadsPerRow / 2; offset > 0; offset >>= 1) value += __shfl_xor_sync(mask, value, offset);
    return value;
}

// v2a: subwarp mapping + vectorization, 3 memory access
// =================================================================================================
template <int ThreadsPerRow, int VecWidth>
__global__ void softmax_subwarp_kernel(const float *__restrict__ x, float *__restrict__ y, int rows, int cols)
{
    constexpr int kRowsPerBlock = kBlockSize / ThreadsPerRow;

    const int lane = threadIdx.x % ThreadsPerRow; // define the col offset within the row
    const int row = blockIdx.x * kRowsPerBlock + static_cast<int>(threadIdx.x) / ThreadsPerRow;

    const unsigned mask = group_mask<ThreadsPerRow>();
    if (row >= rows) return;

    const float *x_row = x + static_cast<size_t>(row) * cols;
    float *y_row = y + static_cast<size_t>(row) * cols;

    // caculate row max
    float thread_max = -FLT_MAX;
    if constexpr (VecWidth == 4)
    {
        const float4 *x4 = reinterpret_cast<const float4 *>(x_row);
        const int vec_cols = cols / 4;
        for (int index = lane; index < vec_cols; index += ThreadsPerRow)
        {
            const float4 value = x4[index];
            thread_max = fmaxf(thread_max, fmaxf(fmaxf(value.x, value.y), fmaxf(value.z, value.w)));
        }
    }
    else
    {
        for (int col = lane; col < cols; col += ThreadsPerRow) thread_max = fmaxf(thread_max, x_row[col]);
    }
    const float row_max = group_reduce_max<ThreadsPerRow>(thread_max, mask);

    // caculate denominator
    float thread_sum = 0.0f;
    if constexpr (VecWidth == 4)
    {
        const float4 *x4 = reinterpret_cast<const float4 *>(x_row);
        const int vec_cols = cols / 4;
        for (int index = lane; index < vec_cols; index += ThreadsPerRow)
        {
            const float4 value = x4[index];
            thread_sum += __expf(value.x - row_max) + __expf(value.y - row_max) + __expf(value.z - row_max) + __expf(value.w - row_max);
        }
    }
    else
    {
        for (int col = lane; col < cols; col += ThreadsPerRow) thread_sum += __expf(x_row[col] - row_max);
    }
    const float row_sum = group_reduce_sum<ThreadsPerRow>(thread_sum, mask);

    // normalize
    const float inv_sum = 1.0f / row_sum;
    if constexpr (VecWidth == 4)
    {
        const float4 *x4 = reinterpret_cast<const float4 *>(x_row);
        float4 *y4 = reinterpret_cast<float4 *>(y_row);
        const int vec_cols = cols / 4;
        for (int index = lane; index < vec_cols; index += ThreadsPerRow)
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
        for (int col = lane; col < cols; col += ThreadsPerRow) y_row[col] = __expf(x_row[col] - row_max) * inv_sum;
    }
}

// v2b: the row lives in registers, so it is read exactly once.
// =================================================================================================
template <int ThreadsPerRow>
__global__ void softmax_subwarp_register_kernel(const float *__restrict__ x, float *__restrict__ y, int rows, int cols)
{
    constexpr int kRowsPerBlock = kBlockSize / ThreadsPerRow;

    const int lane = threadIdx.x % ThreadsPerRow;
    const int row = blockIdx.x * kRowsPerBlock + static_cast<int>(threadIdx.x) / ThreadsPerRow;

    const unsigned mask = group_mask<ThreadsPerRow>();
    if (row >= rows) return;

    const float4 *x4 = reinterpret_cast<const float4 *>(x + static_cast<size_t>(row) * cols);
    float4 *y4 = reinterpret_cast<float4 *>(y + static_cast<size_t>(row) * cols);

    const float4 value = x4[lane];

    const float row_max = group_reduce_max<ThreadsPerRow>(fmaxf(fmaxf(value.x, value.y), fmaxf(value.z, value.w)), mask);

    // caculate denominator and write back to y
    float4 exponential;
    exponential.x = __expf(value.x - row_max);
    exponential.y = __expf(value.y - row_max);
    exponential.z = __expf(value.z - row_max);
    exponential.w = __expf(value.w - row_max);

    const float row_sum = group_reduce_sum<ThreadsPerRow>(exponential.x + exponential.y + exponential.z + exponential.w, mask);
    const float inv_sum = 1.0f / row_sum;

    float4 result;
    result.x = exponential.x * inv_sum;
    result.y = exponential.y * inv_sum;
    result.z = exponential.z * inv_sum;
    result.w = exponential.w * inv_sum;
    y4[lane] = result;
}

template <int ThreadsPerRow>
inline int grid_for(int rows)
{
    constexpr int kRowsPerBlock = kBlockSize / ThreadsPerRow;
    return (rows + kRowsPerBlock - 1) / kRowsPerBlock;
}

}  // namespace

void launch_softmax_subwarp(const float *x, float *y, int rows, int cols, cudaStream_t stream)
{
    if (rows <= 0 || cols <= 0) return;

    if (cols > 256)
    {
        launch_softmax_adaptive(x, y, rows, cols, stream);
        return;
    }

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

    switch (cols)
    {
        case 32: softmax_subwarp_register_kernel<8><<<grid_for<8>(rows), kBlockSize, 0, stream>>>(x, y, rows, cols); return;
        case 64: softmax_subwarp_register_kernel<16><<<grid_for<16>(rows), kBlockSize, 0, stream>>>(x, y, rows, cols); return;
        case 128: softmax_subwarp_register_kernel<32><<<grid_for<32>(rows), kBlockSize, 0, stream>>>(x, y, rows, cols); return;
        default: launch_softmax_subwarp(x, y, rows, cols, stream); return;
    }
}

}  // namespace cuda_op_lab::stable_softmax
