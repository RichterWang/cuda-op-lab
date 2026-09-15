#include "../include/cpu_matmul.h"

namespace cpu_cache {

// naive imply
void matmul_naive(const float* A, const float* B, float* C, size_t M, size_t N, size_t K) {
    // init matrix C
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            C[i * N + j] = 0.0f;
        }
    }
    
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) sum += A[i * K + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
    }
}

} // namespace cpu_cache_study
