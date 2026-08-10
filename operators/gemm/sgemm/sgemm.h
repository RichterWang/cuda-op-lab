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
}  // namespace cuda_op_lab::sgemm