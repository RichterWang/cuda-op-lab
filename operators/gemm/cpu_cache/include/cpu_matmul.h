#ifndef CPU_MATMUL_H
#define CPU_MATMUL_H

#include <cstddef>

namespace cpu_cache {

// naive implementation(also treat as reference)
// C = A * B, where A is M×K, B is K×N, C is M×N
void matmul_naive(const float* A, const float* B, float* C, size_t M, size_t N, size_t K);

// Cache-aware tiling
void matmul_blocked(const float* A, const float* B, float* C, size_t M, size_t N, size_t K, size_t block_size);

// Cache-oblivious
void matmul_recursive(const float* A, const float* B, float* C, size_t M, size_t N, size_t K, size_t threshold);

bool verify_result(const float* C_ref, const float* C_test, size_t M, size_t N, float tolerance = 1e-4f);

} // namespace cpu_cache_study

#endif // CPU_MATMUL_H
