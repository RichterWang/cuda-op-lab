#pragma once

#include <cuda_runtime.h>

namespace cuda_op_lab::sgemm {

void launch_sgemm_naive(
    const float* a,
    const float* b,
    float* c,
    int m,
    int n,
    int k,
    cudaStream_t stream = nullptr);

// kernel for shared memory
void launch_sgemm_shared_mem(
    const float* a,
    const float* b, 
    float* c,
    int m, 
    int n, 
    int k,
    cudaStream_t stream = nullptr);
    
// kernel for shared memory and tile register optimize
void launch_sgemm_register_tiled(
    const float* a,
    const float* b,
    float* c,
    int m,
    int n,
    int k,
    cudaStream_t stream = nullptr); 
}  // namespace cuda_op_lab::sgemm