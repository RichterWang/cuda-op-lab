#include "../include/cpu_matmul.h"
#include <algorithm>
#include <cstring>

namespace cpu_cache {

namespace {

// recursive imply:
// C[row_c:row_c+m, col_c:col_c+n] += A[row_a:row_a+m, col_a:col_a+k] * B[row_b:row_b+k, col_b:col_b+n]
void matmul_recursive_helper(const float* A, const float* B, float* C, size_t row_a, size_t col_a, size_t lda, size_t row_b, size_t col_b, size_t ldb, size_t row_c, size_t col_c, size_t ldc, size_t m, size_t n, size_t k, size_t threshold)        
{
    // basic tile matmul
    if (m <= threshold && n <= threshold && k <= threshold) {
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                float sum = 0.0f;
                for (size_t p = 0; p < k; ++p) {
                    sum += A[(row_a + i) * lda + (col_a + p)] * 
                           B[(row_b + p) * ldb + (col_b + j)];
                }
                C[(row_c + i) * ldc + (col_c + j)] += sum;
            }
        }
        return;
    }
    
    // choose the max dim to divide
    if (m >= n && m >= k) {
        size_t m1 = m / 2;
        size_t m2 = m - m1;
        
        // C[0:m1, :] += A[0:m1, :] * B
        matmul_recursive_helper(A, B, C, row_a, col_a, lda, row_b, col_b, ldb, row_c, col_c, ldc, m1, n, k, threshold);
        
        // C[m1:m, :] += A[m1:m, :] * B
        matmul_recursive_helper(A, B, C, row_a + m1, col_a, lda, row_b, col_b, ldb, row_c + m1, col_c, ldc, m2, n, k, threshold);
    }
    else if (n >= m && n >= k) {
        size_t n1 = n / 2;
        size_t n2 = n - n1;
        
        // C[:, 0:n1] += A * B[:, 0:n1]
        matmul_recursive_helper(A, B, C, row_a, col_a, lda, row_b, col_b, ldb, row_c, col_c, ldc, m, n1, k, threshold);
        
        // C[:, n1:n] += A * B[:, n1:n]
        matmul_recursive_helper(A, B, C, row_a, col_a, lda, row_b, col_b + n1, ldb, row_c, col_c + n1, ldc, m, n2, k, threshold);
    }
    else {
        size_t k1 = k / 2;
        size_t k2 = k - k1;
        
        // C += A[:, 0:k1] * B[0:k1, :]
        matmul_recursive_helper(A, B, C, row_a, col_a, lda, row_b, col_b, ldb, row_c, col_c, ldc, m, n, k1, threshold);
        
        // C += A[:, k1:k] * B[k1:k, :]
        matmul_recursive_helper(A, B, C, row_a, col_a + k1, lda, row_b + k1, col_b, ldb, row_c, col_c, ldc, m, n, k2, threshold);
    }
}

}

// Cache-oblivious:
void matmul_recursive(const float* A, const float* B, float* C, size_t M, size_t N, size_t K, size_t threshold) {
    std::memset(C, 0, sizeof(float) * M * N);
    
    matmul_recursive_helper(A, B, C, 0, 0, K, 0, 0, N, 0, 0, N, M, N, K, threshold);
}

} // namespace cpu_cache_study
