#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>

namespace cuda_op_lab::flash_attention {

// Scaled dot-product attention, single head group, no dropout:
//
//     S = Q K^T / sqrt(d)          [N, N]
//     P = softmax_rowwise(S)       [N, N]
//     O = P V                      [N, d]
//
// Q, K, V, O are device pointers to [batch, heads, seq_len, head_dim] row-major
// bf16 buffers. head_dim is contiguous, so a (batch, head) pair addresses a
// contiguous [seq_len, head_dim] slab and the batch/head dimensions collapse into
// a single index bh = batch * heads + head everywhere below.
//
// Storage is bf16 for the same reasons as rms_norm: it is what an inference
// activation actually is, it halves the traffic on a memory-bound path, and it is
// the format the tensor core MMA instructions want when that version arrives.
// All accumulation happens in fp32. A bf16 accumulator is not an option here: the
// PV product sums seq_len terms, and bf16 cannot represent integers above 256
// exactly, so a 4096-long sum would lose most of its significance.
//
// What makes this operator different from softmax and rms_norm: the intermediate
// S is [N, N] while the inputs and output are [N, d]. At N=2048, d=64 that is a
// 32x blow-up. The entire point of flash attention is that S never reaches HBM,
// so the interesting metric is not "how close to the copy ceiling" but "how much
// traffic exists at all".

struct AttentionShape
{
    int batch;
    int heads;
    int seq_len;   // used for both the query and the key/value length
    int head_dim;  // 64 or 128 in practice
};

// Multiplicative factor applied to the scores. Separated out because the "which
// sqrt" mistake is easy to make and impossible to see in the output.
inline float default_scale(int head_dim)
{
    return 1.0f / std::sqrt(static_cast<float>(head_dim));
}

// FLOPs of the two GEMMs: 2 * N * N * d each, so 4 * N * N * d per (batch, head).
//
// causal is not cosmetic here. With a causal mask only the lower triangle of S is
// ever needed, which is N*(N+1)/2 of the N*N pairs, so the useful FLOP count is
// almost exactly halved. Reporting the dense count for a causal run would
// understate a kernel that skips masked tiles and overstate one that computes
// them and throws the result away, which is precisely the difference between v0
// and the fused versions.
double attention_flops(const AttentionShape& shape, bool causal);

// --------------------------------------------------------------------------
// v0: unfused, three kernels, S materialized in HBM.
// --------------------------------------------------------------------------
//
// This is the implementation flash attention exists to replace, so it is the
// baseline rather than a strawman: scores kernel, row softmax, PV kernel, with
// the [batch, heads, N, N] score matrix written to and read back from global
// memory between each pair.
//
// It is deliberately naive inside each kernel too (one thread per output element,
// no tiling, no vectorization). That is defensible only because the traffic term
// dominates: at N=2048, d=64 the score matrix is 32x the size of Q, K, V and O
// combined, so no amount of work on the GEMM inner loops changes the order of
// magnitude. The fused versions must beat this by removing traffic, not by
// arithmetic, and keeping v0 simple keeps that comparison honest.
//
// workspace must point to at least unfused_workspace_bytes(shape) bytes. That is
// batch * heads * seq_len^2 fp32 values, which grows quadratically and is the
// reason this version cannot be run at every shape the fused ones can. Callers
// are expected to check the size and skip the shape rather than allocate blindly.
size_t unfused_workspace_bytes(const AttentionShape& shape);

void launch_attention_unfused(const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, __nv_bfloat16* o, float* workspace,
                              const AttentionShape& shape, bool causal, cudaStream_t stream = nullptr);

// Row-wise softmax over a [rows, cols] fp32 matrix, in place. Part of v0's
// pipeline, exposed because the cuBLAS-based baseline needs the same step and
// there is no reason for two copies of it.
//
// Rows that are entirely -inf would produce 0/0. That cannot happen under a
// causal mask, since the diagonal entry is always unmasked, but the kernel guards
// it anyway rather than relying on the caller's mask being well formed.
void launch_softmax_rows_inplace(float* s, int rows, int cols, cudaStream_t stream = nullptr);

// --------------------------------------------------------------------------
// v1: fused, one block per query row, online softmax.
// --------------------------------------------------------------------------
//
// S never leaves registers and shared memory. Each block owns one query row, keeps
// the running (m, l) softmax state and the d-element output accumulator, and walks
// the key/value sequence one tile at a time:
//
//     m_new = max(m, max(s_tile))
//     c     = exp(m - m_new)
//     l     = l * c + sum(exp(s_tile - m_new))
//     o     = o * c + exp(s_tile - m_new) @ V_tile
//
// The rescale factor c is what makes a single pass correct: the partial result is
// always a valid softmax over the tiles seen so far, so no second pass over S is
// needed. This is the same combine operator as the online softmax version in
// stable_softmax, applied to the output accumulator as well as to the denominator.
//
// Traffic drops from O(N^2) to O(N*d) per (batch, head) for the query side, and
// the K/V tiles are read once per query row. That last part is the weakness of
// this version and the reason v2 exists: with one query row per block there is no
// reuse of a K/V tile across queries, so K and V get read N times in total. The
// point of v1 is to establish that removing the S traffic alone is worth a large
// factor; fixing the K/V re-read is a separate change with a separate measurement.
//
// Under a causal mask a block stops as soon as the key tile starts past its own
// query index, so roughly half the tiles are never touched. v0 computes them and
// multiplies by zero.
//
// Supported head_dim values are fixed at compile time (64 and 128), because the
// per-thread accumulator and the shared tile sizes are template parameters. Query
// fused_row_supported before calling; an unsupported head_dim is a no-op launch
// rather than a silent wrong answer.
bool fused_row_supported(int head_dim);

void launch_attention_fused_row(const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, __nv_bfloat16* o,
                                const AttentionShape& shape, bool causal, cudaStream_t stream = nullptr);

// What v1 compiled to for a given head_dim, straight from the occupancy API.
//
// shared_bytes_per_block is the number to watch: the K and V tiles are the whole
// working set, and at head_dim=128 they already take 32 KB of the 48 KB default
// limit, which caps blocks per SM at 1 and is a hard constraint on how large the
// tile can grow in v2.
struct FusedRowInfo
{
    bool supported;
    int block_size;
    int tile_n;  // key/value rows resident in shared memory at once
    int shared_bytes_per_block;
    int registers_per_thread;
    int local_bytes_per_thread;  // must be 0, or the output accumulator spilled
    int max_blocks_per_sm;
};

FusedRowInfo query_fused_row(int head_dim);

}  // namespace cuda_op_lab::flash_attention
