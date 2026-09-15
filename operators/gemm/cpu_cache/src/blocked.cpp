#include "../include/cpu_matmul.h"
#include <algorithm>

namespace cpu_cache {

// Cache-aware tiling matrix mutiply
void matmul_blocked(const float* A, const float* B, float* C, size_t M, size_t N, size_t K, size_t block_size) {
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            C[i * N + j] = 0.0f;
        }
    }
    
    // 3 layer tiling
    for (size_t i0 = 0; i0 < M; i0 += block_size) {
        size_t i_end = std::min(i0 + block_size, M);
        
        for (size_t j0 = 0; j0 < N; j0 += block_size) {
            size_t j_end = std::min(j0 + block_size, N);
            
            for (size_t k0 = 0; k0 < K; k0 += block_size) {
                size_t k_end = std::min(k0 + block_size, K);
                
                // standard matmul in current tile
                for (size_t i = i0; i < i_end; ++i) {
                    for (size_t j = j0; j < j_end; ++j) {
                        float sum = C[i * N + j];
                        for (size_t k = k0; k < k_end; ++k) {
                            sum += A[i * K + k] * B[k * N + j];
                        }
                        C[i * N + j] = sum;
                    }
                }
            }
        }
    }
}

} // namespace cpu_cache_study
