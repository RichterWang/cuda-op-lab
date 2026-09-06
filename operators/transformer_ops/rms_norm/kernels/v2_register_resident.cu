// v2: keep the row in registers between the two passes.
//
// v1 reads the row twice: once to accumulate the sum of squares, once to
// normalize. v2 keeps each thread's slice of the row in registers after pass 1,
// so pass 2 reads nothing but gamma. The reduction and the access width are
// unchanged from v1; the only difference is where the data lives between passes.
//
// Why this is worth trying, and why the original reason was wrong. The obvious
// argument is that it halves the read traffic, but that argument does not hold:
// the copy baseline moves one read and one write per element, and v1 already
// reaches 99.9% of it at cols=4096 and 5120. If pass 2 were really costing a
// second trip to DRAM, v1 could not be at the ceiling. It is at the ceiling
// because pass 2 mostly hits L1.
//
// The occupancy experiment (see measure_occupancy.cu) showed where that stops
// being true. Shrinking L1 by requesting shared memory, at unchanged occupancy,
// costs cols=8192 96.9% -> 89.4% and cols=11008 90.7% -> 80.8%, while cols=4096
// and 5120 do not move at all. Rows of 16 KB and 21.5 KB stop fitting; rows of
// 8 KB and 10 KB keep fitting. So the prediction for v2 is narrow and testable:
// large rows should gain, small rows should not, because on small rows there is
// nothing left to remove.
//
// What it costs. Each cached vector is 16 bytes, so 4 registers per thread per
// vector. v1 measures 42 registers per thread; caching 3 vectors adds 12, which
// can push blocks/SM down. That mattered less than expected: halving occupancy
// on this kernel cost cols=8192 only 1 percentage point, because a
// bandwidth-bound kernel does not need many resident warps to hide latency.
//
// The structural constraint. A local array stays in registers only if every
// index into it is known at compile time; otherwise it spills to local memory,
// which is DRAM with a cache in front, and the whole point is lost. So the
// per-thread vector count has to be a template parameter with fully unrolled
// loops over it, which in turn means only a fixed set of (block size, vectors
// per thread) combinations exist. Shapes outside that set fall back to v1
// rather than being handled by a slow generic path.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "rms_norm.h"

namespace cuda_op_lab::rms_norm {
namespace {

constexpr int kWarpSize = 32;
constexpr unsigned kFullMask = 0xffffffffu;
constexpr int kVecWidth = 8;

// Registers consumed per cached vector: 16 bytes / 4 bytes per register.
constexpr int kRegistersPerVector = 4;

// Ceiling on cached vectors per thread. At 4 vectors a thread holds 16 registers
// of row data on top of v1's 42, which is where the register file starts to bite.
// Shapes needing more than this fall back to v1.
constexpr int kMaxVecPerThread = 4;

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

// VecPerThread is a template parameter, not a runtime value, so that `cache` can
// be indexed by a compile-time constant after unrolling and therefore live in
// registers. Making it a runtime loop bound would put the array in local memory
// and turn the saved L1 reads into DRAM reads, which is worse than v1.
//
// The strided assignment (thread t owns vectors t, t + BlockSize, ...) is kept
// from v1 rather than switching to contiguous per-thread chunks. Strided is what
// keeps a warp's 32 accesses adjacent in memory and therefore coalesced; a
// contiguous split would have each thread walking its own distant region and
// break that.
template <int BlockSize, int VecPerThread>
__global__ void rmsnorm_resident_kernel(const __nv_bfloat16 *__restrict__ x, const __nv_bfloat16 *__restrict__ gamma, __nv_bfloat16 *__restrict__ y, int rows, int cols, float eps)
{
    constexpr int kWarpsPerBlock = BlockSize / kWarpSize;

    const int row = blockIdx.x;
    if (row >= rows) return;

    const int vec_cols = cols / kVecWidth;

    const BF16x8 *x_row = reinterpret_cast<const BF16x8 *>(x + static_cast<size_t>(row) * cols);
    BF16x8 *y_row = reinterpret_cast<BF16x8 *>(y + static_cast<size_t>(row) * cols);
    const BF16x8 *gamma_vec = reinterpret_cast<const BF16x8 *>(gamma);

    __shared__ float warp_partial[kWarpsPerBlock];

    // The row data, held across the reduction. This is the whole point of v2.
    BF16x8 cache[VecPerThread];

    // Pass 1: read the row once, square-accumulate in fp32, and keep the bf16
    // source in registers. Slots past the end of the row are left unwritten and
    // never read back, since pass 2 applies the same bound.
    float thread_sum = 0.0f;
    #pragma unroll
    for (int slot = 0; slot < VecPerThread; ++slot)
    {
        const int index = threadIdx.x + slot * BlockSize;
        if (index >= vec_cols) break;

        cache[slot] = x_row[index];

    #pragma unroll
        for (int part = 0; part < kVecWidth; ++part)
        {
            const float value = __bfloat162float(cache[slot].elem[part]);
            thread_sum += value * value;
        }
    }

    const float total = block_reduce_sum<BlockSize>(thread_sum, warp_partial);
    const float scale = rsqrtf(total / static_cast<float>(cols) + eps);

    // Pass 2: no load of x. gamma is still read, but every row reads the same
    // cols-element vector, so after the first few blocks it is resident in cache
    // and contributes almost no DRAM traffic.
    #pragma unroll
    for (int slot = 0; slot < VecPerThread; ++slot)
    {
        const int index = threadIdx.x + slot * BlockSize;
        if (index >= vec_cols) break;

        const BF16x8 weight = gamma_vec[index];

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

// The set of (block size, vectors per thread) pairs that exist as instantiations.
// Chosen so that BlockSize * VecPerThread covers vec_cols with as little surplus
// as possible, since a thread whose slots all fall past the end of the row still
// costs registers and still joins the reduction.
//
// Block size is 512 for every multi-vector case. That is not arbitrary: the block
// size sweep found 512 fastest at every shape measured, by a wide margin over
// 1024 (which leaves threads with no work at cols=4096) and over 128 and 256
// (which give each thread too many vectors).
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

    const Config config = select_config(cols);

    // Rows too long to cache, or not a multiple of the vector width, go to v1.
    // Falling back is the honest option here: a generic v2 that spilled to local
    // memory would be slower than v1 while looking like a new optimization.
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

// Reports what configuration a shape lands on and what the hardware grants for
// it. The register count is the number to watch: if it exceeds v1's 42 by much
// more than 4 * vec_per_thread, the cache has partly spilled to local memory and
// the version is not doing what it claims.
ResidentInfo query_rmsnorm_resident(int cols)
{
    const Config config = select_config(cols);

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
