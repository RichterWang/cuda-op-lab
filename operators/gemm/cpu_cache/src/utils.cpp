#include "../include/cpu_matmul.h"
#include <cmath>
#include <algorithm>

namespace cpu_cache {

// check the result
bool verify_result(const float* C_ref, const float* C_test, size_t M, size_t N, float tolerance) {
    float max_abs_error = 0.0f;
    float max_rel_error = 0.0f;
    
    for (size_t i = 0; i < M * N; ++i) {
        float abs_error = std::fabs(C_ref[i] - C_test[i]);
        float denominator = std::max(1.0f, std::fabs(C_ref[i]));
        float rel_error = abs_error / denominator;
        
        max_abs_error = std::max(max_abs_error, abs_error);
        max_rel_error = std::max(max_rel_error, rel_error);
    }
    
    return max_rel_error < tolerance;
}

} // namespace cpu_cache_study
