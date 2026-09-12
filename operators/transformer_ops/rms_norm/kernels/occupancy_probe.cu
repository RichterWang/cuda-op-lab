// occupancy probe kernel
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "rms_norm.h"

namespace cuda_op_lab::rms_norm {
// include unname namespcace
namespace {

constexpr int kWarpSize = 32;
constexpr unsigned kFullMask = 0xffffffffu;
constexpr int kVecWidth = 8;

union alignas(16) BF16x8
{
    float4 raw;
    __nv_bfloat16 elem[kVecWidth];
};

__device__ __forceinline__ float warp_reduce_sum(float value)
{
    #pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) value += __shfl_xor_sync(kFullMask, value, offset);
    return value;
}

template <int BlockSize>
__device__ __forceinline__ float block_reduce_sum(float value, float *shared)
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

// same as rmsnorm_warp_vec_kernel
template <int BlockSize>
__global__ void probe_vec_kernel(const __nv_bfloat16 *__restrict__ x, const __nv_bfloat16 *__restrict__ gamma, __nv_bfloat16 *__restrict__ y, int rows, int cols, float eps)
{
    constexpr int kWarpsPerBlock = BlockSize / kWarpSize;

    const int row = blockIdx.x;
    if (row >= rows) return;

    const int vec_cols = cols / kVecWidth;

    const BF16x8 *x_row = reinterpret_cast<const BF16x8 *>(x + static_cast<size_t>(row) * cols);
    BF16x8 *y_row = reinterpret_cast<BF16x8 *>(y + static_cast<size_t>(row) * cols);
    const BF16x8 *gamma_vec = reinterpret_cast<const BF16x8 *>(gamma);

    __shared__ float warp_partial[kWarpsPerBlock];

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

//Type-erased handle get kernel address
const void *kernel_address(int block_size)
{
    switch (block_size)
    {
        case 1024:
            return reinterpret_cast<const void *>(&probe_vec_kernel<1024>);
        case 512:
            return reinterpret_cast<const void *>(&probe_vec_kernel<512>);
        case 256:
            return reinterpret_cast<const void *>(&probe_vec_kernel<256>);
        case 128:
            return reinterpret_cast<const void *>(&probe_vec_kernel<128>);
        default:
            return reinterpret_cast<const void *>(&probe_vec_kernel<64>);
    }
}

// activately requests above 48 KB per block are not available by default and must be opted
void enable_large_shared(int block_size, size_t dynamic_shared_bytes)
{
    if (dynamic_shared_bytes <= 48 * 1024) return;

    switch (block_size)
    {
        case 1024:
            cudaFuncSetAttribute(probe_vec_kernel<1024>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(dynamic_shared_bytes)); // to enable large shared mem
            break;
        case 512:
            cudaFuncSetAttribute(probe_vec_kernel<512>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(dynamic_shared_bytes));
            break;
        case 256:
            cudaFuncSetAttribute(probe_vec_kernel<256>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(dynamic_shared_bytes));
            break;
        case 128:
            cudaFuncSetAttribute(probe_vec_kernel<128>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(dynamic_shared_bytes));
            break;
        default:
            cudaFuncSetAttribute(probe_vec_kernel<64>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(dynamic_shared_bytes));
            break;
    }
}

}  // namespace

// lauch not ceiling kernel
void launch_rmsnorm_probe(const __nv_bfloat16 *x, const __nv_bfloat16 *gamma, __nv_bfloat16 *y, int rows, int cols, float eps, int block_size, size_t dynamic_shared_bytes, cudaStream_t stream)
{
    if (rows <= 0 || cols <= 0) return;
    if (cols % kVecWidth != 0) return;  // probe only covers the vectorized path

    enable_large_shared(block_size, dynamic_shared_bytes);

    switch (block_size)
    {
        case 1024:
            probe_vec_kernel<1024><<<dim3(rows), dim3(1024), dynamic_shared_bytes, stream>>>(x, gamma, y, rows, cols, eps);
            break;
        case 512:
            probe_vec_kernel<512><<<dim3(rows), dim3(512), dynamic_shared_bytes, stream>>>(x, gamma, y, rows, cols, eps);
            break;
        case 256:
            probe_vec_kernel<256><<<dim3(rows), dim3(256), dynamic_shared_bytes, stream>>>(x, gamma, y, rows, cols, eps);
            break;
        case 128:
            probe_vec_kernel<128><<<dim3(rows), dim3(128), dynamic_shared_bytes, stream>>>(x, gamma, y, rows, cols, eps);
            break;
        default:
            probe_vec_kernel<64><<<dim3(rows), dim3(64), dynamic_shared_bytes, stream>>>(x, gamma, y, rows, cols, eps);
            break;
    }
}

// static hardware occupancy situation, max_blocks_per_sm comes from the occupancy API
ProbeInfo query_rmsnorm_probe(int block_size, size_t dynamic_shared_bytes)
{
    enable_large_shared(block_size, dynamic_shared_bytes);

    const void *kernel = kernel_address(block_size);

    cudaFuncAttributes attributes{};
    cudaFuncGetAttributes(&attributes, kernel);

    int max_blocks = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks, kernel, block_size, dynamic_shared_bytes); // get static occupancy

    ProbeInfo info{};
    info.registers_per_thread = attributes.numRegs;
    info.static_shared_bytes = static_cast<int>(attributes.sharedSizeBytes);
    info.max_blocks_per_sm = max_blocks;
    info.resident_threads_per_sm = max_blocks * block_size;
    return info;
}

}  // namespace cuda_op_lab::rms_norm
