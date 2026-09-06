// v1: warp shuffle reduction and 128-bit vectorized access, in one step.
//
// Two changes over v0, applied together because they are independent and both
// were already established as wins during the softmax work. Staging them here
// would produce a version whose only novelty is a result measured elsewhere.
//
//   1. The block reduction becomes __shfl_xor_sync within each warp plus a
//      single shared-memory combine across warps. v0 needed log2(256) = 8
//      barriers; this needs one.
//
//   2. Each thread moves 16 bytes per access instead of 2. On bf16 that is 8
//      elements, twice what a float4 carries in fp32.
//
// The 8-elements-per-access figure is the thing worth remembering. It changes the
// arithmetic of every later decision: hidden=4096 spread over 1024 threads leaves
// each thread only 4 elements, which is half a vector. So at the shape that
// matters most, a 1024-thread block and full-width access are mutually
// exclusive, and the register-resident version will have to pick one. In fp32
// that conflict does not arise.
//
// Alignment is a real constraint, not a formality. A 16-byte access requires the
// address to be 16-byte aligned, and since each row starts at
// x + row * cols, every row start is only aligned when cols is a multiple of 8.
// cudaMalloc gives a 256-byte-aligned base, so the base itself is never the
// problem, but cols=4095 breaks row alignment for all rows except the first. The
// launcher checks this and falls back to a scalar path rather than reading
// garbage.
//
// Precision boundary is unchanged from v0: bf16 loads convert to fp32
// immediately, all accumulation is fp32, and conversion back happens only at the
// store. The vector load moves 8 bf16 values as one 16-byte word, then unpacks
// them; it does not perform any bf16 arithmetic.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "rms_norm.h"

namespace cuda_op_lab::rms_norm {
namespace {

constexpr int kWarpSize = 32;
constexpr unsigned kFullMask = 0xffffffffu;

// bf16 elements per 16-byte access. The fp32 equivalent is 4.
constexpr int kVecWidth = 8;

// 16 bytes of bf16, moved as one unit. float4 is used purely as a 16-byte
// carrier with the right alignment; nothing here interprets it as floats.
union alignas(16) BF16x8
{
    float4 raw;
    __nv_bfloat16 elem[kVecWidth];
};

__device__ inline float warp_reduce_sum(float value)
{
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) value += __shfl_xor_sync(kFullMask, value, offset);
    return value;
}

// Block-wide sum: one shuffle reduction per warp, then a single barrier and a
// short fold over the per-warp results.
template <int BlockSize>
__device__ inline float block_reduce_sum(float value, float *shared)
{
    constexpr int kWarpsPerBlock = BlockSize / kWarpSize;

    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp_id = threadIdx.x >> 5;

    value = warp_reduce_sum(value);
    if (lane == 0) shared[warp_id] = value;
    __syncthreads();

    float total = shared[0];
#pragma unroll
    for (int index = 1; index < kWarpsPerBlock; ++index) total += shared[index];
    return total;
}

// Vectorized path. Requires cols % kVecWidth == 0 so that every row start is
// 16-byte aligned and no tail handling is needed.
template <int BlockSize>
__global__ void rmsnorm_warp_vec_kernel(const __nv_bfloat16 *__restrict__ x, const __nv_bfloat16 *__restrict__ gamma, __nv_bfloat16 *__restrict__ y,
                                        int rows, int cols, float eps)
{
    constexpr int kWarpsPerBlock = BlockSize / kWarpSize;

    const int row = blockIdx.x;
    if (row >= rows) return;

    const int vec_cols = cols / kVecWidth;

    const BF16x8 *x_row = reinterpret_cast<const BF16x8 *>(x + static_cast<size_t>(row) * cols);
    BF16x8 *y_row = reinterpret_cast<BF16x8 *>(y + static_cast<size_t>(row) * cols);
    const BF16x8 *gamma_vec = reinterpret_cast<const BF16x8 *>(gamma);

    __shared__ float warp_partial[kWarpsPerBlock];

    // Pass 1: sum of squares, 8 elements per load.
    float thread_sum = 0.0f;
    for (int index = threadIdx.x; index < vec_cols; index += BlockSize)
    {
        const BF16x8 chunk = x_row[index];
#pragma unroll
        for (int slot = 0; slot < kVecWidth; ++slot)
        {
            const float value = __bfloat162float(chunk.elem[slot]);
            thread_sum += value * value;
        }
    }

    const float total = block_reduce_sum<BlockSize>(thread_sum, warp_partial);
    const float scale = rsqrtf(total / static_cast<float>(cols) + eps);

    // Pass 2: normalize. The row is read a second time; only a register-resident
    // version can avoid that, and only when the row is small enough to hold.
    for (int index = threadIdx.x; index < vec_cols; index += BlockSize)
    {
        const BF16x8 chunk = x_row[index];
        const BF16x8 weight = gamma_vec[index];

        BF16x8 out;
#pragma unroll
        for (int slot = 0; slot < kVecWidth; ++slot)
        {
            const float value = __bfloat162float(chunk.elem[slot]);
            const float gain = __bfloat162float(weight.elem[slot]);
            out.elem[slot] = __float2bfloat16(value * scale * gain);
        }
        y_row[index] = out;
    }
}

// Scalar fallback for rows that are not a multiple of kVecWidth. Keeps the warp
// shuffle reduction, drops only the wide access, so an odd cols costs bandwidth
// but stays correct.
template <int BlockSize>
__global__ void rmsnorm_warp_scalar_kernel(const __nv_bfloat16 *__restrict__ x, const __nv_bfloat16 *__restrict__ gamma,
                                           __nv_bfloat16 *__restrict__ y, int rows, int cols, float eps)
{
    constexpr int kWarpsPerBlock = BlockSize / kWarpSize;

    const int row = blockIdx.x;
    if (row >= rows) return;

    const __nv_bfloat16 *x_row = x + static_cast<size_t>(row) * cols;
    __nv_bfloat16 *y_row = y + static_cast<size_t>(row) * cols;

    __shared__ float warp_partial[kWarpsPerBlock];

    float thread_sum = 0.0f;
    for (int col = threadIdx.x; col < cols; col += BlockSize)
    {
        const float value = __bfloat162float(x_row[col]);
        thread_sum += value * value;
    }

    const float total = block_reduce_sum<BlockSize>(thread_sum, warp_partial);
    const float scale = rsqrtf(total / static_cast<float>(cols) + eps);

    for (int col = threadIdx.x; col < cols; col += BlockSize)
    {
        const float value = __bfloat162float(x_row[col]);
        const float gain = __bfloat162float(gamma[col]);
        y_row[col] = __float2bfloat16(value * scale * gain);
    }
}

// Pick a block size that gives every thread at least one full vector to move,
// while staying large enough to hide latency.
//
// The rule is vec_cols / BlockSize >= 1, i.e. cols >= BlockSize * 8. A block
// larger than that leaves some threads with nothing to load in pass 1 while
// still paying for them in the reduction. Note how quickly this saturates:
// 1024 threads want cols >= 8192, so hidden=4096 lands on 512 threads. In fp32
// the same shape would have used 1024.
int select_block_size(int cols)
{
    if (cols >= 1024 * kVecWidth) return 1024;
    if (cols >= 512 * kVecWidth) return 512;
    if (cols >= 256 * kVecWidth) return 256;
    if (cols >= 128 * kVecWidth) return 128;
    return 64;
}

}  // namespace

void launch_rmsnorm_warp_vec(const __nv_bfloat16 *x, const __nv_bfloat16 *gamma, __nv_bfloat16 *y, int rows, int cols, float eps,
                             cudaStream_t stream)
{
    if (rows <= 0 || cols <= 0) return;

    const int block_size = select_block_size(cols);

    // A row start is 16-byte aligned only when cols is a multiple of 8, since
    // rows are laid out back to back. cudaMalloc already guarantees the base is
    // aligned, so cols is the only thing that can break it.
    const bool can_vectorize = (cols % kVecWidth == 0);

    if (can_vectorize)
    {
        switch (block_size)
        {
            case 1024:
                rmsnorm_warp_vec_kernel<1024><<<dim3(rows), dim3(1024), 0, stream>>>(x, gamma, y, rows, cols, eps);
                break;
            case 512:
                rmsnorm_warp_vec_kernel<512><<<dim3(rows), dim3(512), 0, stream>>>(x, gamma, y, rows, cols, eps);
                break;
            case 256:
                rmsnorm_warp_vec_kernel<256><<<dim3(rows), dim3(256), 0, stream>>>(x, gamma, y, rows, cols, eps);
                break;
            case 128:
                rmsnorm_warp_vec_kernel<128><<<dim3(rows), dim3(128), 0, stream>>>(x, gamma, y, rows, cols, eps);
                break;
            default:
                rmsnorm_warp_vec_kernel<64><<<dim3(rows), dim3(64), 0, stream>>>(x, gamma, y, rows, cols, eps);
                break;
        }
        return;
    }

    switch (block_size)
    {
        case 1024:
            rmsnorm_warp_scalar_kernel<1024><<<dim3(rows), dim3(1024), 0, stream>>>(x, gamma, y, rows, cols, eps);
            break;
        case 512:
            rmsnorm_warp_scalar_kernel<512><<<dim3(rows), dim3(512), 0, stream>>>(x, gamma, y, rows, cols, eps);
            break;
        case 256:
            rmsnorm_warp_scalar_kernel<256><<<dim3(rows), dim3(256), 0, stream>>>(x, gamma, y, rows, cols, eps);
            break;
        case 128:
            rmsnorm_warp_scalar_kernel<128><<<dim3(rows), dim3(128), 0, stream>>>(x, gamma, y, rows, cols, eps);
            break;
        default:
            rmsnorm_warp_scalar_kernel<64><<<dim3(rows), dim3(64), 0, stream>>>(x, gamma, y, rows, cols, eps);
            break;
    }
}

}  // namespace cuda_op_lab::rms_norm
