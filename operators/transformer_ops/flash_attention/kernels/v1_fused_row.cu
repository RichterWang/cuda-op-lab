// v1: fused attention, one block per query row, online softmax.
//
// The whole point is that S never exists. Each block owns one query row, holds the
// query vector and the fp32 output accumulator in registers, and streams the
// key/value sequence through shared memory in tiles of kTileN rows. The softmax
// state is a running (m, l) pair updated per tile:
//
//     m_new = max(m, tile_max)
//     c     = exp(m - m_new)
//     l     = c * l + sum_j exp(s_j - m_new)
//     o_d   = c * o_d + sum_j exp(s_j - m_new) * V[j][d]
//
// The correctness argument is that the same rescale factor c applies to both l and
// o, so the pair (l, o) after any number of tiles is exactly what a full softmax
// over those tiles would have produced. Nothing about the final division by l
// depends on having seen all the tiles at once, so a single streaming pass is
// enough. This is the online softmax combine operator from stable_softmax, with
// the output accumulator carried along for the ride.
//
// Layout choice: one warp per block would map more naturally onto the reduction,
// but the K/V tile has to be loaded cooperatively and a single warp loading a
// 64x64 bf16 tile is 8 KB through 32 lanes. kBlockSize = 128 gives four warps to
// spread the tile load across, at the cost of a cross-warp reduction for the tile
// max and sum. That reduction is over four values and happens once per tile, so it
// is not on the critical path the way it was in stable_softmax's v0.
//
// Traffic, per (batch, head): Q read once (N*d), O written once (N*d), and K and V
// read once per query row (N * N*d each, since every block walks the whole
// sequence). Total roughly 2*N^2*d elements against v0's 5*N^2 fp32 words plus
// 4*N*d. At d=64 that is 128*N^2 bf16 bytes vs 20*N^2 fp32 bytes, so v1 moves
// *more* bytes than v0 on paper.
//
// That is not a mistake, and it is the most useful thing this version teaches.
//
// Measured on a 3070 Ti laptop, v1 is 0.49-0.67x of v0 across every shape in the
// benchmark. It loses, and the reason is worth being precise about because the
// obvious explanation is the wrong one:
//
//   Not the traffic. Both kernels sit at 0.2-0.45 TFLOPS against a card that will
//   do an order of magnitude more, so neither is anywhere near a bandwidth or a
//   compute ceiling. Removing S from DRAM cannot help when DRAM was not the limit.
//
//   Occupancy. The two fp32 tiles cost 33 KB per block, which caps residency at 2
//   blocks per SM = 256 of 1536 available threads, about 17%. There is not enough
//   work in flight to cover the shared-memory and exp() latency in the inner loop.
//
//   No query reuse. Every block loads the entire K/V sequence to serve one query
//   row. The loaded tile is used for a single dot product and a single weighted
//   add, so the arithmetic per byte of shared memory traffic is as low as it can
//   be, and the same slab is loaded seq_len times over.
//
// Both point the same direction: one query row per block is too little work per
// block. Tiling queries so a loaded K/V tile serves many rows at once fixes the
// reuse directly, and turns the row-parallel/dimension-parallel transpose below
// into a proper tile GEMM. That is v2's job.
//
// Keeping v1 in the tree rather than deleting it is deliberate: it establishes
// that fusion alone is not the win. The win is fusion plus reuse, and separating
// the two is the only way to know which one paid.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cfloat>

#include "flash_attention.h"

namespace cuda_op_lab::flash_attention {

namespace {

constexpr int kBlockSize = 128;

// Key/value rows resident in shared memory at once. Fixed by capacity, not by
// taste: both tiles are held in fp32, so the cost is 2 * kTileN * HeadDim * 4
// bytes and the default per-block limit is 48 KB. Holding the element count
// constant at 4096 puts that at 32 KB for every supported HeadDim, which leaves
// room for the scratch arrays and keeps the tile the same size in bytes whether
// head_dim is 64 or 128 -- so a comparison between the two is not also a
// comparison of shared memory footprints.
//
// The consequence is that head_dim=128 walks the sequence in twice as many tiles
// as head_dim=64, which is the honest tradeoff: more barriers per row, same
// working set.
template <int HeadDim>
constexpr int tile_rows_for()
{
    return 4096 / HeadDim;
}

// Pad the K tile's row stride by one float. Without it the score loop is a 32-way
// bank conflict: thread t reads k_tile[t][dim], the row stride is HeadDim floats,
// HeadDim is a multiple of 32, so bank = (t * HeadDim + dim) % 32 = dim % 32 and
// all 32 lanes of a warp hit the same bank on every one of the HeadDim iterations.
// An odd stride makes bank = (t + dim) % 32, which is distinct per lane.
//
// V needs no padding: the accumulate loop reads v_tile[row][dim] with dim varying
// across lanes and row fixed, so consecutive lanes already hit consecutive banks.
// Padding it anyway would cost shared memory for nothing.
constexpr int kPad = 1;

// Warp-level max/sum via xor shuffle, then a 4-element cross-warp combine through
// shared memory. Two barriers per tile total, which at kTileN = 64 is amortized
// over 64 scores and 64*d multiply-adds.
__inline__ __device__ float warp_reduce_max(float value)
{
    for (int offset = 16; offset > 0; offset >>= 1) value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, offset));
    return value;
}

__inline__ __device__ float warp_reduce_sum(float value)
{
    for (int offset = 16; offset > 0; offset >>= 1) value += __shfl_xor_sync(0xffffffffu, value, offset);
    return value;
}

// HeadDim is a template parameter because the output accumulator is a local array
// and a local array only stays in registers when every index into it is a
// compile-time constant. The same lesson as rms_norm v2: a runtime head_dim would
// spill the accumulator to local memory and turn the fastest part of this kernel
// into the slowest.
//
// The accumulator is split across the 32 lanes of a warp, each lane owning
// HeadDim / 32 dimensions. All four warps hold the same layout, so the final
// combine is a cross-warp sum over matching dimension slices. At HeadDim = 64 that
// is 2 floats per lane, at 128 it is 4.
template <int HeadDim>
__global__ void fused_row_kernel(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
                                 __nv_bfloat16* __restrict__ o, int seq_len, float scale, bool causal)
{
    constexpr int kWarps = kBlockSize / 32;
    constexpr int kDimsPerLane = HeadDim / 32;
    constexpr int kTileN = tile_rows_for<HeadDim>();

    // K and V tiles, plus scratch for the cross-warp reductions. All shared
    // allocations are declared here rather than at point of use, so the block's
    // total footprint is readable in one place.
    __shared__ float k_tile[kTileN][HeadDim + kPad];
    __shared__ float v_tile[kTileN][HeadDim];
    __shared__ float probabilities[kTileN];
    __shared__ float scratch[kWarps];
    __shared__ float q_shared[HeadDim];
    __shared__ float output[HeadDim];

    const int query = blockIdx.x;
    const int bh = blockIdx.y;
    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int warp = tid / 32;

    if (query >= seq_len) return;

    const size_t slab = static_cast<size_t>(bh) * seq_len * HeadDim;

    // The query row is read by every thread on every tile, so it goes to shared
    // memory once. Broadcast reads from shared memory are conflict-free.
    for (int index = tid; index < HeadDim; index += kBlockSize) q_shared[index] = __bfloat162float(q[slab + static_cast<size_t>(query) * HeadDim + index]);
    __syncthreads();

    // Running softmax state and output accumulator. m starts at -FLT_MAX rather
    // than -inf so the first rescale computes exp(-FLT_MAX - m_new), which
    // underflows to 0 cleanly, instead of exp(-inf + inf) = NaN.
    float running_max = -FLT_MAX;
    float running_sum = 0.0f;
    float accumulator[kDimsPerLane];
    for (int index = 0; index < kDimsPerLane; ++index) accumulator[index] = 0.0f;

    // Under a causal mask this row only attends to keys at or before its own
    // index, so the loop stops at the tile containing the diagonal. Roughly half
    // the tiles are never loaded, which is where the causal speedup comes from --
    // v0 loads all of them and multiplies by zero.
    const int key_limit = causal ? query + 1 : seq_len;

    for (int tile_start = 0; tile_start < key_limit; tile_start += kTileN)
    {
        const int tile_rows = min(kTileN, key_limit - tile_start);

        // Cooperative tile load. Converting to fp32 on the way in trades shared
        // memory capacity for not repeating the conversion in the inner loops:
        // each K element is used once but each V element is used by kDimsPerLane
        // reads, and the fp32 tile keeps the accumulate loop free of conversions.
        for (int index = tid; index < tile_rows * HeadDim; index += kBlockSize)
        {
            const int row = index / HeadDim;
            const int dim = index % HeadDim;
            k_tile[row][dim] = __bfloat162float(k[slab + static_cast<size_t>(tile_start + row) * HeadDim + dim]);
            v_tile[row][dim] = __bfloat162float(v[slab + static_cast<size_t>(tile_start + row) * HeadDim + dim]);
        }
        __syncthreads();

        // Scores for this tile. One thread per tile row, computing the full
        // HeadDim dot product, so the scores live in registers spread across the
        // block. kTileN is at most 64 and kBlockSize is 128, so a thread holds at
        // most one score and the surplus threads only participate in the
        // reductions and the V accumulation.
        float score = -FLT_MAX;
        const int score_row = tid;
        if (score_row < tile_rows)
        {
            float dot = 0.0f;
            for (int dim = 0; dim < HeadDim; ++dim) dot += q_shared[dim] * k_tile[score_row][dim];
            score = dot * scale;
        }

        // Tile max, then the online rescale. Both reductions are over the whole
        // block because the scores are spread across it.
        float tile_max = warp_reduce_max(score);
        if (lane == 0) scratch[warp] = tile_max;
        __syncthreads();
        tile_max = scratch[0];
        for (int index = 1; index < kWarps; ++index) tile_max = fmaxf(tile_max, scratch[index]);

        const float new_max = fmaxf(running_max, tile_max);
        const float rescale = __expf(running_max - new_max);

        const float probability = score_row < tile_rows ? __expf(score - new_max) : 0.0f;

        float tile_sum = warp_reduce_sum(probability);
        __syncthreads();  // scratch is reused, so the previous read must be done
        if (lane == 0) scratch[warp] = tile_sum;
        __syncthreads();
        tile_sum = 0.0f;
        for (int index = 0; index < kWarps; ++index) tile_sum += scratch[index];

        running_sum = running_sum * rescale + tile_sum;
        running_max = new_max;

        // Output accumulation. Each thread holds the probability for one tile row
        // and needs to add probability * V[row][:] into the accumulator, but the
        // accumulator is split by dimension across lanes, not by row. So every
        // thread has to see every probability, which means routing them through
        // shared memory.
        //
        // This is the one place where the one-row-per-block layout costs
        // something: the score is produced row-parallel and consumed
        // dimension-parallel, which forces a shared-memory transpose of a
        // kTileN-element vector plus a barrier. v2's query tiling makes this a
        // proper tile-by-tile GEMM instead.
        if (score_row < tile_rows) probabilities[score_row] = probability;
        __syncthreads();

        for (int index = 0; index < kDimsPerLane; ++index)
        {
            const int dim = index * 32 + lane;
            float partial = 0.0f;
            // Each warp handles a strided subset of the tile rows, so the four
            // warps split the row loop and their partials are summed below.
            for (int row = warp; row < tile_rows; row += kWarps) partial += probabilities[row] * v_tile[row][dim];
            accumulator[index] = accumulator[index] * rescale + partial;
        }
        __syncthreads();
    }

    // Cross-warp combine of the accumulator, then the single division by l.
    //
    // Deferring the division to the end is what makes the streaming loop cheap:
    // dividing per tile would be correct too, but it would put a division on the
    // per-tile path for no benefit.
    if (warp == 0)
        for (int index = 0; index < kDimsPerLane; ++index) output[index * 32 + lane] = 0.0f;
    __syncthreads();

    for (int source = 0; source < kWarps; ++source)
    {
        if (warp == source)
            for (int index = 0; index < kDimsPerLane; ++index) output[index * 32 + lane] += accumulator[index];
        __syncthreads();
    }

    const float inverse = running_sum > 0.0f ? 1.0f / running_sum : 0.0f;
    for (int index = tid; index < HeadDim; index += kBlockSize)
        o[slab + static_cast<size_t>(query) * HeadDim + index] = __float2bfloat16(output[index] * inverse);
}

}  // namespace

bool fused_row_supported(int head_dim)
{
    return head_dim == 64 || head_dim == 128;
}

void launch_attention_fused_row(const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, __nv_bfloat16* o,
                                const AttentionShape& shape, bool causal, cudaStream_t stream)
{
    if (!fused_row_supported(shape.head_dim)) return;

    const dim3 grid(static_cast<unsigned>(shape.seq_len), static_cast<unsigned>(shape.batch * shape.heads));
    const float scale = default_scale(shape.head_dim);

    if (shape.head_dim == 64)
        fused_row_kernel<64><<<grid, kBlockSize, 0, stream>>>(q, k, v, o, shape.seq_len, scale, causal);
    else
        fused_row_kernel<128><<<grid, kBlockSize, 0, stream>>>(q, k, v, o, shape.seq_len, scale, causal);
}

FusedRowInfo query_fused_row(int head_dim)
{
    FusedRowInfo info{};
    info.supported = fused_row_supported(head_dim);
    if (!info.supported) return info;

    info.block_size = kBlockSize;
    info.tile_n = head_dim == 64 ? tile_rows_for<64>() : tile_rows_for<128>();

    cudaFuncAttributes attributes{};
    if (head_dim == 64)
        cudaFuncGetAttributes(&attributes, reinterpret_cast<const void*>(fused_row_kernel<64>));
    else
        cudaFuncGetAttributes(&attributes, reinterpret_cast<const void*>(fused_row_kernel<128>));

    info.shared_bytes_per_block = static_cast<int>(attributes.sharedSizeBytes);
    info.registers_per_thread = attributes.numRegs;
    info.local_bytes_per_thread = static_cast<int>(attributes.localSizeBytes);

    int blocks = 0;
    if (head_dim == 64)
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, reinterpret_cast<const void*>(fused_row_kernel<64>), kBlockSize, 0);
    else
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, reinterpret_cast<const void*>(fused_row_kernel<128>), kBlockSize, 0);
    info.max_blocks_per_sm = blocks;

    return info;
}

}  // namespace cuda_op_lab::flash_attention
