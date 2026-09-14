// v2: keep the row in registers between the two passes.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "rms_norm.h"

namespace cuda_op_lab::rms_norm {
namespace {

constexpr int kWarpSize = 32;
constexpr unsigned kFullMask = 0xffffffffu;
constexpr int kVecWidth = 8;

// Registers consumed per cached vector: 16 / 4
constexpr int kRegistersPerVector = 4;

// experiment defined register ceiling of one thread per vec(to avoid register spill)
constexpr int kMaxVecPerThread = 4;

// cpp 11 & cpp 20
union alignas(16) BF16x8
{
    float4 row;
    __nv_bfloat16 elem[kVecWidth];
};

__device__ __forceinline__ float warp_reduce_sum(float value)
{
    #pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) value += __shfl_xor_sync(kFullMask, value, offset);
    return value;
}

// tree reduction may costy
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

// template param, to make compile finate so that the space will malloc in register
template <int BlockSize, int VecPerThread>
__global__ void rmsnorm_resident_kernel(const __nv_bfloat16 *__restrict__ x, const __nv_bfloat16 *__restrict__ gamma, __nv_bfloat16 *__restrict__ y, int rows, int cols, float eps)
{
    constexpr int kWarpsPerBlock = BlockSize / kWarpSize;

    const int row = blockIdx.x; // each block process a row
    if (row >= rows) return;

    const int vec_cols = cols / kVecWidth; // get vector index

    // vec read gamma_vec, y_row, x_row
    const BF16x8 *x_row = reinterpret_cast<const BF16x8 *>(x + static_cast<size_t>(row) * cols);
    BF16x8 *y_row = reinterpret_cast<BF16x8 *>(y + static_cast<size_t>(row) * cols);
    const BF16x8 *gamma_vec = reinterpret_cast<const BF16x8 *>(gamma);

    __shared__ float warp_partial[kWarpsPerBlock]; // store the data per warp

    // for each thread vec store, register resident
    BF16x8 cache[VecPerThread];

    // read the row and do square accmulataion(only use needed space)
    float thread_sum = 0.0f; // FP32
    #pragma unroll
    for (int slot = 0; slot < VecPerThread; ++slot)
    {
        const int index = threadIdx.x + slot * BlockSize; // col offset of each thread
        if (index >= vec_cols) break; // other data will not write in

        // get the buffer register
        cache[slot] = x_row[index];

        // all caculate down in FP32 mod
        #pragma unroll
        for (int part = 0; part < kVecWidth; ++part)
        {
            const float value = __bfloat162float(cache[slot].elem[part]);
            thread_sum += value * value;
        }
    }

    const float total = block_reduce_sum<BlockSize>(thread_sum, warp_partial);
    const float scale = rsqrtf(total / static_cast<float>(cols) + eps); // get the RMS data of each row

    // caculate the y result, only need to read gamma, which will stay in L1 cache after servial trail
    #pragma unroll
    for (int slot = 0; slot < VecPerThread; ++slot)
    {
        const int index = threadIdx.x + slot * BlockSize;
        if (index >= vec_cols) break;

        const BF16x8 weight = gamma_vec[index]; // read from DRAM, a row share it

        BF16x8 out;
        #pragma unroll
        for (int part = 0; part < kVecWidth; ++part)
        {
            const float value = __bfloat162float(cache[slot].elem[part]);
            const float gain = __bfloat162float(weight.elem[part]);
            out.elem[part] = __float2bfloat16(value * scale * gain);
        }
        y_row[index] = out;
    }
}

// template param config struct
struct Config
{
    int block_size;
    int vec_per_thread;
    bool supported;
};

Config select_config(int cols)
{
    if (cols % kVecWidth != 0) return {0, 0, false};  // vectorized path only

    const int vec_cols = cols / kVecWidth;

    // Small rows: one vector per thread, block size shrunk to fit the row rather
    // than leaving most of the block idle.
    if (vec_cols <= 128) return {128, 1, vec_cols > 0};
    if (vec_cols <= 256) return {256, 1, true};
    if (vec_cols <= 512) return {512, 1, true};

    const int needed = (vec_cols + 512 - 1) / 512;
    if (needed > kMaxVecPerThread) return {0, 0, false};

    return {512, needed, true};
}

}  // namespace

void launch_rmsnorm_resident(const __nv_bfloat16 *x, const __nv_bfloat16 *gamma, __nv_bfloat16 *y, int rows, int cols, float eps, cudaStream_t stream)
{
    if (rows <= 0 || cols <= 0) return;

    const Config config = select_config(cols); // choose the support

    // if not cacheable, use V1 rather than register resident
    if (!config.supported)
    {
        launch_rmsnorm_warp_vec(x, gamma, y, rows, cols, eps, stream);
        return;
    }

    const dim3 grid(rows);

    if (config.block_size == 128)
    {
        rmsnorm_resident_kernel<128, 1><<<grid, dim3(128), 0, stream>>>(x, gamma, y, rows, cols, eps);
        return;
    }
    if (config.block_size == 256)
    {
        rmsnorm_resident_kernel<256, 1><<<grid, dim3(256), 0, stream>>>(x, gamma, y, rows, cols, eps);
        return;
    }

    switch (config.vec_per_thread)
    {
        case 1:
            rmsnorm_resident_kernel<512, 1><<<grid, dim3(512), 0, stream>>>(x, gamma, y, rows, cols, eps);
            break;
        case 2:
            rmsnorm_resident_kernel<512, 2><<<grid, dim3(512), 0, stream>>>(x, gamma, y, rows, cols, eps);
            break;
        case 3:
            rmsnorm_resident_kernel<512, 3><<<grid, dim3(512), 0, stream>>>(x, gamma, y, rows, cols, eps);
            break;
        default:
            rmsnorm_resident_kernel<512, 4><<<grid, dim3(512), 0, stream>>>(x, gamma, y, rows, cols, eps);
            break;
    }
}

// report on the hardware status of each execute
ResidentInfo query_rmsnorm_resident(int cols)
{
    // choose the config
    const Config config = select_config(cols);

    // fill in the basic info
    ResidentInfo info{};
    info.supported = config.supported;
    info.block_size = config.block_size;
    info.vec_per_thread = config.vec_per_thread;
    info.expected_cache_registers = config.vec_per_thread * kRegistersPerVector;

    if (!config.supported) return info;

    const void *kernel = nullptr;
    if (config.block_size == 128)
    {
        kernel = reinterpret_cast<const void *>(&rmsnorm_resident_kernel<128, 1>);
    }
    else if (config.block_size == 256)
    {
        kernel = reinterpret_cast<const void *>(&rmsnorm_resident_kernel<256, 1>);
    }
    else
    {
        switch (config.vec_per_thread)
        {
            case 1:
                kernel = reinterpret_cast<const void *>(&rmsnorm_resident_kernel<512, 1>);
                break;
            case 2:
                kernel = reinterpret_cast<const void *>(&rmsnorm_resident_kernel<512, 2>);
                break;
            case 3:
                kernel = reinterpret_cast<const void *>(&rmsnorm_resident_kernel<512, 3>);
                break;
            default:
                kernel = reinterpret_cast<const void *>(&rmsnorm_resident_kernel<512, 4>);
                break;
        }
    }

    cudaFuncAttributes attributes{};
    cudaFuncGetAttributes(&attributes, kernel);

    int max_blocks = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks, kernel, config.block_size, 0);

    info.registers_per_thread = attributes.numRegs;
    info.local_bytes_per_thread = static_cast<int>(attributes.localSizeBytes);
    info.max_blocks_per_sm = max_blocks;
    info.resident_threads_per_sm = max_blocks * config.block_size;
    return info;
}

}  // namespace cuda_op_lab::rms_norm
