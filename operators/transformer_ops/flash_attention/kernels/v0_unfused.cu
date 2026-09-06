// v0: unfused attention. Three kernels, S materialized in global memory.
//
// This is the shape of the computation before flash attention, kept as the
// baseline because it is what the fused version has to beat. The score matrix is
// written to a workspace, read back for the softmax, written again, and read a
// third time by the PV product.
//
// The traffic accounting per (batch, head), in fp32 words:
//
//   scores kernel   write N*N            reads Q, K (N*d each)
//   softmax kernel  read N*N, write N*N
//   PV kernel       read N*N             reads V, writes O
//
// So 5 * N^2 words of score traffic against 4 * N * d words of Q/K/V/O traffic.
// At N=2048, d=64 that ratio is 5*4.2M vs 4*131K, about 40x. The score matrix is
// the entire cost, which is why the inner loops here are left naive: making the
// GEMMs three times faster would move the total by a few percent.
//
// One deliberate choice worth naming: the workspace is fp32, not bf16. Storing S
// in bf16 would halve the dominant traffic term and make the baseline look better
// than the unfused implementations people actually write, and it would also lose
// precision in exactly the place the fused version keeps full fp32. Keeping fp32
// makes the comparison against v1 a comparison of algorithms rather than of score
// dtypes.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cfloat>

#include "flash_attention.h"

namespace cuda_op_lab::flash_attention {

namespace {

constexpr int kScoreBlockDim = 16;   // 16x16 = 256 threads for the two GEMMs
constexpr int kSoftmaxBlockSize = 256;

// S[bh] = Q[bh] @ K[bh]^T * scale, with masked entries set to -inf.
//
// One thread per score element, no tiling. Each thread reads a full d-element row
// of Q and of K, so Q and K are re-read N times over the grid. That is wasteful in
// isolation and irrelevant in context: d is 64 or 128 while the write of S is N*N,
// and the L2 absorbs most of the re-reads anyway.
//
// -FLT_MAX rather than -inf for the mask. exp(x - m) with both operands -inf
// produces NaN rather than 0, and a row whose max is -inf can only arise from a
// fully masked row, so keeping the sentinel finite makes the softmax below
// arithmetic rather than special-cased.
__global__ void scores_kernel(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k, float* __restrict__ s, int seq_len,
                              int head_dim, float scale, bool causal)
{
    const int query = blockIdx.y * kScoreBlockDim + threadIdx.y;
    const int key = blockIdx.x * kScoreBlockDim + threadIdx.x;
    const int bh = blockIdx.z;

    if (query >= seq_len || key >= seq_len) return;

    const size_t slab = static_cast<size_t>(bh) * seq_len * head_dim;
    const size_t score_slab = static_cast<size_t>(bh) * seq_len * seq_len;

    if (causal && key > query)
    {
        s[score_slab + static_cast<size_t>(query) * seq_len + key] = -FLT_MAX;
        return;
    }

    const __nv_bfloat16* q_row = q + slab + static_cast<size_t>(query) * head_dim;
    const __nv_bfloat16* k_row = k + slab + static_cast<size_t>(key) * head_dim;

    // fp32 accumulation: d is only 64 or 128, but a bf16 accumulator would lose
    // significance well before that, and the reference this is checked against
    // accumulates in double.
    float dot = 0.0f;
    for (int dim = 0; dim < head_dim; ++dim) dot += __bfloat162float(q_row[dim]) * __bfloat162float(k_row[dim]);

    s[score_slab + static_cast<size_t>(query) * seq_len + key] = dot * scale;
}

// Row-wise softmax in place, one block per row, shared-memory tree reduction.
//
// This is the v0-grade softmax from stable_softmax rather than the optimized one,
// on purpose: v0 exists to show the cost of materializing S, and swapping in a
// faster softmax would shift the baseline without changing the conclusion, while
// making the baseline harder to explain.
__global__ void softmax_rows_kernel(float* __restrict__ s, int cols)
{
    extern __shared__ float scratch[];

    float* row = s + static_cast<size_t>(blockIdx.x) * cols;
    const int tid = threadIdx.x;

    float local_max = -FLT_MAX;
    for (int col = tid; col < cols; col += blockDim.x) local_max = fmaxf(local_max, row[col]);

    scratch[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride) scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
        __syncthreads();
    }
    const float row_max = scratch[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x)
    {
        const float value = __expf(row[col] - row_max);
        row[col] = value;
        local_sum += value;
    }

    scratch[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride) scratch[tid] += scratch[tid + stride];
        __syncthreads();
    }

    // A fully masked row has every entry at -FLT_MAX, so row_max is -FLT_MAX,
    // every difference is 0, and the sum is cols rather than 0 -- the guard below
    // never fires for that case. It fires only if the sum genuinely underflowed,
    // which the max subtraction is supposed to prevent. Keeping it means a
    // pathological input produces zeros instead of NaNs propagating into O.
    const float denominator = scratch[0] > 0.0f ? scratch[0] : 1.0f;
    const float inverse = 1.0f / denominator;
    __syncthreads();

    for (int col = tid; col < cols; col += blockDim.x) row[col] *= inverse;
}

// O[bh] = P[bh] @ V[bh]. One thread per output element, summing over seq_len.
__global__ void pv_kernel(const float* __restrict__ p, const __nv_bfloat16* __restrict__ v, __nv_bfloat16* __restrict__ o, int seq_len, int head_dim)
{
    const int query = blockIdx.y * kScoreBlockDim + threadIdx.y;
    const int dim = blockIdx.x * kScoreBlockDim + threadIdx.x;
    const int bh = blockIdx.z;

    if (query >= seq_len || dim >= head_dim) return;

    const size_t slab = static_cast<size_t>(bh) * seq_len * head_dim;
    const size_t score_slab = static_cast<size_t>(bh) * seq_len * seq_len;

    const float* p_row = p + score_slab + static_cast<size_t>(query) * seq_len;

    float accumulator = 0.0f;
    for (int key = 0; key < seq_len; ++key) accumulator += p_row[key] * __bfloat162float(v[slab + static_cast<size_t>(key) * head_dim + dim]);

    o[slab + static_cast<size_t>(query) * head_dim + dim] = __float2bfloat16(accumulator);
}

}  // namespace

double attention_flops(const AttentionShape& shape, bool causal)
{
    const double n = static_cast<double>(shape.seq_len);
    const double pairs = causal ? n * (n + 1.0) / 2.0 : n * n;

    // Two GEMMs, each 2 FLOP (multiply + add) per (query, key, dim) triple.
    return 4.0 * pairs * static_cast<double>(shape.head_dim) * static_cast<double>(shape.batch) * static_cast<double>(shape.heads);
}

size_t unfused_workspace_bytes(const AttentionShape& shape)
{
    return static_cast<size_t>(shape.batch) * shape.heads * shape.seq_len * shape.seq_len * sizeof(float);
}

void launch_softmax_rows_inplace(float* s, int rows, int cols, cudaStream_t stream)
{
    const dim3 grid(static_cast<unsigned>(rows));
    const dim3 block(kSoftmaxBlockSize);
    softmax_rows_kernel<<<grid, block, kSoftmaxBlockSize * sizeof(float), stream>>>(s, cols);
}

void launch_attention_unfused(const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, __nv_bfloat16* o, float* workspace,
                              const AttentionShape& shape, bool causal, cudaStream_t stream)
{
    const int bh_count = shape.batch * shape.heads;
    const float scale = default_scale(shape.head_dim);

    const dim3 score_block(kScoreBlockDim, kScoreBlockDim);
    const dim3 score_grid((shape.seq_len + kScoreBlockDim - 1) / kScoreBlockDim, (shape.seq_len + kScoreBlockDim - 1) / kScoreBlockDim,
                          static_cast<unsigned>(bh_count));
    scores_kernel<<<score_grid, score_block, 0, stream>>>(q, k, workspace, shape.seq_len, shape.head_dim, scale, causal);

    // The softmax treats the whole workspace as one [bh*N, N] matrix. Rows never
    // cross a (batch, head) boundary because each slab is exactly N rows of N,
    // so no per-slab launch is needed.
    launch_softmax_rows_inplace(workspace, bh_count * shape.seq_len, shape.seq_len, stream);

    const dim3 pv_block(kScoreBlockDim, kScoreBlockDim);
    const dim3 pv_grid((shape.head_dim + kScoreBlockDim - 1) / kScoreBlockDim, (shape.seq_len + kScoreBlockDim - 1) / kScoreBlockDim,
                       static_cast<unsigned>(bh_count));
    pv_kernel<<<pv_grid, pv_block, 0, stream>>>(workspace, v, o, shape.seq_len, shape.head_dim);
}

}  // namespace cuda_op_lab::flash_attention
