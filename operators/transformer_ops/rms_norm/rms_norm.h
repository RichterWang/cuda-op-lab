#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace cuda_op_lab::rms_norm {

// conduct RMS norm per row 
// memory bound operator
// a block per row, shared mem tree shffle, 2x read and write(without register resident)
void launch_rmsnorm_naive(const __nv_bfloat16* x, const __nv_bfloat16* gamma, __nv_bfloat16* y, int rows, int cols, float eps, cudaStream_t stream = nullptr);

// v1: warp shuffle reduction and 128-bit vectorized access, together. (ideas from stable online softmax)
void launch_rmsnorm_warp_vec(const __nv_bfloat16* x, const __nv_bfloat16* gamma, __nv_bfloat16* y, int rows, int cols, float eps, cudaStream_t stream = nullptr);
// expand the dim of the original input(3 dim expansion)
void launch_rmsnorm_3d(const __nv_bfloat16* x, const __nv_bfloat16* gamma, __nv_bfloat16* y, int batch, int seq, int hidden, float eps, cudaStream_t stream = nullptr);

// v2: add row resiednce registers between the two passes
// may hit L1 cache as not expected(done by compiler)
void launch_rmsnorm_resident(const __nv_bfloat16* x, const __nv_bfloat16* gamma, __nv_bfloat16* y, int rows, int cols, float eps, cudaStream_t stream = nullptr);
// struct to prove the optimization done of v2
struct ResidentInfo
{
    bool supported;  // false means this cols falls back to v1
    int block_size;
    int vec_per_thread;
    int expected_cache_registers;  // 4 per cached vector
    int registers_per_thread;
    int local_bytes_per_thread;  // check if cache spilled
    int max_blocks_per_sm;
    int resident_threads_per_sm;
};
// the experiment conducted on if v2 use the so-called data
ResidentInfo query_rmsnorm_resident(int cols);

// function to get the ceiling bandwidth
void launch_copy_baseline(const __nv_bfloat16* x, __nv_bfloat16* y, int rows, int cols, cudaStream_t stream = nullptr);

// Occupancy probe. Diagnostic only
// seperately check block_size and shared mem melloc influence on the kernel
void launch_rmsnorm_probe(const __nv_bfloat16* x, const __nv_bfloat16* gamma, __nv_bfloat16* y, int rows, int cols, float eps, int block_size, size_t dynamic_shared_bytes, cudaStream_t stream = nullptr);

// hardware based information struct (from probe function)
struct ProbeInfo
{
    int registers_per_thread;
    int static_shared_bytes;
    int max_blocks_per_sm;
    int resident_threads_per_sm;  // max_blocks_per_sm * block_size
};

ProbeInfo query_rmsnorm_probe(int block_size, size_t dynamic_shared_bytes);

}  // namespace cuda_op_lab::rms_norm
