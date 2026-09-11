#include <cuda_runtime.h>

#include <cfloat>

#include "stable_softmax.h"

namespace cuda_op_lab::stable_softmax {
namespace {

constexpr int kWarpSize = 32;
constexpr unsigned kFullMask = 0xffffffffu;

// struct of each thread
struct MaxSum
{
    float max; // store current max
    float sum; // store current sum
};

// safety initialization
__device__ inline MaxSum max_sum_identity()
{
    return MaxSum{-FLT_MAX, 0.0f};
}

// online accmulate
__device__ __forceinline__ MaxSum max_sum_accumulate(MaxSum state, float value)
{
    if (value <= state.max)
    {
        state.sum += __expf(value - state.max);
        return state;
    }

    // rescaled
    state.sum = state.sum * __expf(state.max - value) + 1.0f;
    state.max = value;
    return state;
}

// combine process still rescaled
__device__ inline MaxSum max_sum_combine(MaxSum a, MaxSum b)
{
    const float merged_max = fmaxf(a.max, b.max);

    // scale aviod overflow
    const float scaled_a = (a.sum > 0.0f) ? a.sum * __expf(a.max - merged_max) : 0.0f;
    const float scaled_b = (b.sum > 0.0f) ? b.sum * __expf(b.max - merged_max) : 0.0f;

    return MaxSum{merged_max, scaled_a + scaled_b};
}

// reduce process
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

// v3a kernel, no instruction level parallelism.(loop-carried dependency)
template <int BlockSize>
__global__ void softmax_online_kernel(const float *__restrict__ x, float *__restrict__ y, int rows, int cols)
{
    constexpr int kWarpsPerBlock = BlockSize / kWarpSize;

    const int row = blockIdx.x;
    if (row >= rows) return;

    const float *x_row = x + static_cast<size_t>(row) * cols;
    float *y_row = y + static_cast<size_t>(row) * cols;

    const int lane = threadIdx.x & (kWarpSize - 1); // inside warp id
    const int warp_id = threadIdx.x >> 5; // /32 get warp id

    __shared__ float warp_max[kWarpsPerBlock]; // each warp store its max value
    __shared__ float warp_sum[kWarpsPerBlock]; // each warp store its sum value

    // online scan
    MaxSum state = max_sum_identity(); // each thread init
    for (int col = threadIdx.x; col < cols; col += BlockSize) state = max_sum_accumulate(state, x_row[col]); // each block resibonsible for a row

    // reduce each warp
    state = warp_reduce_max_sum(state);
    if (lane == 0)
    {
        warp_max[warp_id] = state.max;
        warp_sum[warp_id] = state.sum;
    }
    __syncthreads();

    // reduce across the shared mem
    MaxSum row_state{warp_max[0], warp_sum[0]};
    #pragma unroll
    for (int index = 1; index < kWarpsPerBlock; ++index) row_state = max_sum_combine(row_state, MaxSum{warp_max[index], warp_sum[index]});

    // Normalize process
    const float inverse_sum = 1.0f / row_state.sum;
    for (int col = threadIdx.x; col < cols; col += BlockSize) y_row[col] = __expf(x_row[col] - row_state.max) * inverse_sum;
}

// v3b kernel: Instrcution level parallelism kernel
template <int BlockSize, int Accumulators>
__global__ void softmax_online_ilp_kernel(const float *__restrict__ x, float *__restrict__ y, int rows, int cols)
{
    constexpr int kWarpsPerBlock = BlockSize / kWarpSize; // how many warps in a block
    constexpr int kStride = BlockSize * Accumulators; // how many threads in a block * how many accumulators per thread

    const int row = blockIdx.x;
    if (row >= rows) return;

    const float *x_row = x + static_cast<size_t>(row) * cols;
    float *y_row = y + static_cast<size_t>(row) * cols;

    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp_id = threadIdx.x >> 5;

    __shared__ float warp_max[kWarpsPerBlock];
    __shared__ float warp_sum[kWarpsPerBlock];

    MaxSum state[Accumulators]; // array of accmulator
    #pragma unroll
    for (int slot = 0; slot < Accumulators; ++slot) state[slot] = max_sum_identity(); // init thread buffer

    // caculate the max
    for (int base = threadIdx.x; base < cols; base += kStride)
    {
        #pragma unroll
        for (int slot = 0; slot < Accumulators; ++slot)
        {
            const int col = base + slot * BlockSize; // between blocksize
            if (col < cols) state[slot] = max_sum_accumulate(state[slot], x_row[col]);
        }
    }

    // reduce the max
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

    // reduce between shared mem
    MaxSum row_state{warp_max[0], warp_sum[0]};
    #pragma unroll
    for (int index = 1; index < kWarpsPerBlock; ++index) row_state = max_sum_combine(row_state, MaxSum{warp_max[index], warp_sum[index]});

    // normalize and write in
    const float inverse_sum = 1.0f / row_state.sum;
    for (int col = threadIdx.x; col < cols; col += BlockSize) y_row[col] = __expf(x_row[col] - row_state.max) * inverse_sum;
}

}  // namespace

void launch_softmax_online(const float *x, float *y, int rows, int cols, cudaStream_t stream)
{
    if (rows <= 0 || cols <= 0) return;

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
