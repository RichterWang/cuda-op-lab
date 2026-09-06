// Traffic probes. These are not softmax kernels; they exist to measure how many
// bytes actually reach DRAM, using only cudaEvent timing.
//
// The bandwidth numbers in the benchmark always divide by the *ideal* traffic
// (one read + one write per element), so extra passes over a row show up as
// extra time rather than extra bytes. That makes it possible to infer real DRAM
// traffic from the timings, but only by arithmetic. These probes turn the
// inference into a direct measurement:
//
//   probe_1read : reads the row once, writes once          -> traffic 2N
//   probe_3read : reads the row three times, writes once   -> traffic 4N
//
// If a shape is genuinely DRAM bound with all three passes missing cache, then
// t(3read) / t(1read) should approach 2.0. If passes 2 and 3 hit in L1, the
// ratio should stay near 1.0. If the shape is bound by per-row overhead rather
// than bandwidth, both probes will be far below the copy ceiling and the ratio
// says little.
//
// probe_3read keeps the same data dependency chain as the real kernel (pass 2
// needs the row max, pass 3 needs the row sum) so the compiler cannot merge the
// passes or hoist the loads. Only the exp() calls are removed, since the point
// is to isolate traffic, not arithmetic.
//
// Both probes are templated on block size, because the block-size experiment
// showed that L1 hit rate depends on how many rows are resident per SM. The
// 256-thread probes match launch_softmax_warp_reduce; the 1024-thread ones match
// what launch_softmax_adaptive picks for long rows. Comparing the two answers a
// question the 256-thread probe cannot: once a large block has already pulled
// passes 2 and 3 into L1, is there any DRAM traffic left for a two-pass
// algorithm to remove?

#include <cuda_runtime.h>

#include <cfloat>

#include "stable_softmax.h"

namespace cuda_op_lab::stable_softmax {
namespace {

constexpr int kWarpSize = 32;
constexpr unsigned kFullMask = 0xffffffffu;

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

// One read + one write per element, same block-per-row mapping as the softmax
// kernels so the comparison is like for like.
template <int BlockSize>
__global__ void probe_1read_kernel(const float *__restrict__ x, float *__restrict__ y, int rows, int cols)
{
    const int row = blockIdx.x;
    if (row >= rows) return;

    const float *x_row = x + static_cast<size_t>(row) * cols;
    float *y_row = y + static_cast<size_t>(row) * cols;

    for (int col = threadIdx.x; col < cols; col += BlockSize) y_row[col] = x_row[col] * 0.5f;
}

// Three reads + one write. Same shape of dependency chain as v1: the second
// pass cannot start before the first reduction completes, and the third cannot
// start before the second.
template <int BlockSize>
__global__ void probe_3read_kernel(const float *__restrict__ x, float *__restrict__ y, int rows, int cols)
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

    // pass 1
    float thread_max = -FLT_MAX;
    for (int col = threadIdx.x; col < cols; col += BlockSize) thread_max = fmaxf(thread_max, x_row[col]);

    thread_max = warp_reduce_max(thread_max);
    if (lane == 0) warp_max[warp_id] = thread_max;
    __syncthreads();

    float row_max = warp_max[0];
#pragma unroll
    for (int index = 1; index < kWarpsPerBlock; ++index) row_max = fmaxf(row_max, warp_max[index]);

    // pass 2, depends on row_max
    float thread_sum = 0.0f;
    for (int col = threadIdx.x; col < cols; col += BlockSize) thread_sum += (x_row[col] - row_max);

    thread_sum = warp_reduce_sum(thread_sum);
    if (lane == 0) warp_sum[warp_id] = thread_sum;
    __syncthreads();

    float row_sum = warp_sum[0];
#pragma unroll
    for (int index = 1; index < kWarpsPerBlock; ++index) row_sum += warp_sum[index];

    // pass 3, depends on row_sum
    const float scale = 1.0f / (fabsf(row_sum) + 1.0f);
    for (int col = threadIdx.x; col < cols; col += BlockSize) y_row[col] = (x_row[col] - row_max) * scale;
}

}  // namespace

void launch_probe_1read(const float *x, float *y, int rows, int cols, cudaStream_t stream)
{
    if (rows <= 0 || cols <= 0) return;
    probe_1read_kernel<256><<<dim3(rows), dim3(256), 0, stream>>>(x, y, rows, cols);
}

void launch_probe_3read(const float *x, float *y, int rows, int cols, cudaStream_t stream)
{
    if (rows <= 0 || cols <= 0) return;
    probe_3read_kernel<256><<<dim3(rows), dim3(256), 0, stream>>>(x, y, rows, cols);
}

void launch_probe_1read_big_block(const float *x, float *y, int rows, int cols, cudaStream_t stream)
{
    if (rows <= 0 || cols <= 0) return;
    probe_1read_kernel<1024><<<dim3(rows), dim3(1024), 0, stream>>>(x, y, rows, cols);
}

void launch_probe_3read_big_block(const float *x, float *y, int rows, int cols, cudaStream_t stream)
{
    if (rows <= 0 || cols <= 0) return;
    probe_3read_kernel<1024><<<dim3(rows), dim3(1024), 0, stream>>>(x, y, rows, cols);
}

}  // namespace cuda_op_lab::stable_softmax
