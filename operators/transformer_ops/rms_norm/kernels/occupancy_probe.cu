// Occupancy probe. Not a shipping kernel: it is v1's kernel with the block size
// and the occupancy turned into knobs, so the two can be varied independently.
//
// The question it answers. v1 reaches 99.9% of the copy ceiling at cols=4096 and
// 5120, but only 87.7% at 8192 and 83.8% at 11008. Two explanations fit that
// split and they are easy to confuse, because select_block_size happens to
// switch from 512 to 1024 threads at exactly the boundary where the numbers drop:
// occupancy, and tail waste in the final loop iteration.
//
// A plain block-size sweep cannot separate them, since changing the block size
// moves occupancy, tail shape and per-thread element count together. So the probe
// also takes a dynamic shared memory request that the kernel never reads. Shared
// memory is reserved per block at launch out of a fixed per-SM budget, whether or
// not the kernel body touches it, so asking for a large amount lowers blocks per
// SM while leaving block size, loop structure and instruction mix identical.
//
// Both explanations turned out to be wrong, and the measurements are worth
// keeping precisely for that:
//
//   Occupancy. The occupancy API grants 2 blocks per SM at 512 threads and 1 at
//   1024, so both sit at 66.7% resident threads, yet 512 is 77% faster at
//   cols=4096 (0.1620 vs 0.2860 ms). The 1536-thread per-SM limit never binds:
//   at 42 registers per thread the register file caps 512-thread blocks at 2,
//   not the 3 that 1536/512 would suggest. Forcing occupancy down further, at
//   fixed block size 512, moves cols=8192 only from 96.9% to 95.9%. A
//   bandwidth-bound kernel does not need occupancy to hide latency.
//
//   Tail waste. cols=8192 at 1024 threads has vec_cols exactly 1024, so every
//   thread loads one vector and there is no tail, and it still measures 91.0%.
//   cols=5120 at 512 threads leaves three quarters of the block idle in the last
//   iteration and reaches 99.6%. Tail occupancy and throughput are uncorrelated
//   here.
//
//   What does track performance is L1 capacity. Shared memory and L1 come out of
//   the same 100 KB per SM, so the shared request shrinks L1 by the same amount.
//   At fixed occupancy, going from 0 to 40 KB of shared costs cols=8192 96.9% ->
//   89.4% and cols=11008 90.7% -> 80.8%, while cols=4096 and 5120 do not move.
//   The shapes that degrade are the ones whose rows stop fitting in L1 once it
//   shrinks, which points at pass 2 re-reads escaping to L2 as the real cause of
//   the original gap.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "rms_norm.h"

namespace cuda_op_lab::rms_norm {
namespace {

constexpr int kWarpSize = 32;
constexpr unsigned kFullMask = 0xffffffffu;
constexpr int kVecWidth = 8;

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

// Identical arithmetic to rmsnorm_warp_vec_kernel. Kept as a separate copy so
// that experimenting here cannot change the behaviour of the measured kernel.
//
// The dynamic shared memory the launch requests is never referenced. It exists
// only to consume the per-SM shared budget and thereby cap blocks per SM.
template <int BlockSize>
__global__ void probe_vec_kernel(const __nv_bfloat16 *__restrict__ x, const __nv_bfloat16 *__restrict__ gamma, __nv_bfloat16 *__restrict__ y,
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

// Type-erased handle to a specific template instantiation, so the launcher and
// the attribute query can share one dispatch.
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

// Requests above 48 KB per block are not available by default and must be opted
// into per kernel. Without this, a large request silently fails the launch.
void enable_large_shared(int block_size, size_t dynamic_shared_bytes)
{
    if (dynamic_shared_bytes <= 48 * 1024) return;

    switch (block_size)
    {
        case 1024:
            cudaFuncSetAttribute(probe_vec_kernel<1024>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(dynamic_shared_bytes));
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

void launch_rmsnorm_probe(const __nv_bfloat16 *x, const __nv_bfloat16 *gamma, __nv_bfloat16 *y, int rows, int cols, float eps, int block_size,
                          size_t dynamic_shared_bytes, cudaStream_t stream)
{
    if (rows <= 0 || cols <= 0) return;
    if (cols % kVecWidth != 0) return;  // probe covers the vectorized path only

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

// Reports what the hardware will actually do with this configuration, rather
// than what the arithmetic suggests. max_blocks_per_sm comes from the occupancy
// API, so it already accounts for registers, shared memory, and the thread and
// block limits together.
ProbeInfo query_rmsnorm_probe(int block_size, size_t dynamic_shared_bytes)
{
    enable_large_shared(block_size, dynamic_shared_bytes);

    const void *kernel = kernel_address(block_size);

    cudaFuncAttributes attributes{};
    cudaFuncGetAttributes(&attributes, kernel);

    int max_blocks = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks, kernel, block_size, dynamic_shared_bytes);

    ProbeInfo info{};
    info.registers_per_thread = attributes.numRegs;
    info.static_shared_bytes = static_cast<int>(attributes.sharedSizeBytes);
    info.max_blocks_per_sm = max_blocks;
    info.resident_threads_per_sm = max_blocks * block_size;
    return info;
}

}  // namespace cuda_op_lab::rms_norm
