#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>

namespace cuda_op_lab::flash_attention {

// define the shape of the memory space
struct AttentionShape
{
    int batch;
    int heads;
    int seq_len;   // query and the key/value length
    int head_dim;  // 64/128
};

// scale func
inline float default_scale(int head_dim)
{
    return 1.0f / std::sqrt(static_cast<float>(head_dim));
}

// caculate flops
double attention_flops(const AttentionShape& shape, bool causal);

// caculate memory storage size
size_t unfused_workspace_bytes(const AttentionShape& shape);

// naive implmentation
void launch_attention_unfused(const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, __nv_bfloat16* o, float* workspace, const AttentionShape& shape, bool causal, cudaStream_t stream = nullptr);

// naive stable softmax
void launch_softmax_rows_inplace(float* s, int rows, int cols, cudaStream_t stream = nullptr);

// each query a block, add online softmax, reg, sharedmem
bool fused_row_supported(int head_dim);

// V1 implmentation
void launch_attention_fused_row(const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, __nv_bfloat16* o, const AttentionShape& shape, bool causal, cudaStream_t stream = nullptr);

// v1 compiled param, for a given head_dim, from the occupancy API.
struct FusedRowInfo
{
    bool supported;
    int block_size;
    int tile_n;  // key/value rows resident in shared memory at once
    int shared_bytes_per_block; // key info
    int registers_per_thread;
    int local_bytes_per_thread;  // must be 0, or the output accumulator spilled
    int max_blocks_per_sm;
};

FusedRowInfo query_fused_row(int head_dim);

}  // namespace cuda_op_lab::flash_attention
