#include "../include/cpu_matmul.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <chrono>
#include <cstring>
#include <algorithm>
#include <functional>

// performance counter struct (perf)
struct PerfCounters {
    double cache_misses = 0.0;
    double cache_references = 0.0;
    double instructions = 0.0;
    double cycles = 0.0;
    double miss_rate = 0.0;
};

// test config
struct TestConfig {
    size_t M, N, K;
    std::string description;
};

// timer class
class Timer {
public:
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
    }
    
    double stop() {
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end_time - start_time;
        return elapsed.count();
    }
    
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
};

// generate random matrix for computing
void generate_random_matrix(float* matrix, size_t size, unsigned seed = 5206) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < size; ++i) {
        matrix[i] = dist(gen);
    }
}

// cacluate GFLOPS
double calculate_gflops(size_t M, size_t N, size_t K, double time_ms) {
    double flops = 2.0 * M * N * K;
    return flops / (time_ms * 1e6);
}

// printing part
void print_separator() {
    std::cout << std::string(100, '=') << std::endl;
}

// print test config infomation
void print_test_info(const TestConfig& config) {
    print_separator();
    std::cout << "test config: " << config.description << std::endl;
    std::cout << "matrix.size: M=" << config.M << ", N=" << config.N << ", K=" << config.K << std::endl;
    std::cout << "if sepcial size: ";
    if ((config.M & (config.M - 1)) == 0 && 
        (config.N & (config.N - 1)) == 0 && 
        (config.K & (config.K - 1)) == 0) {
        std::cout << "2n" << std::endl;
    } else {
        std::cout << "not 2n" << std::endl;
    }
    print_separator();
}

// test part:
// run single test
void run_algorithm_test(const std::string& name, std::function<void(const float*, const float*, float*, size_t, size_t, size_t)> func, const float* A, const float* B, float* C, size_t M, size_t N, size_t K, const float* C_ref, int warmup_runs = 3, int test_runs = 10)
{
    Timer timer;
    
    // warmup
    for (int i = 0; i < warmup_runs; ++i) {
        func(A, B, C, M, N, K);
    }
    
    // formal running:
    std::vector<double> times;
    times.reserve(test_runs);
    
    for (int i = 0; i < test_runs; ++i) {
        std::memset(C, 0, sizeof(float) * M * N);
        timer.start();
        func(A, B, C, M, N, K);
        times.push_back(timer.stop());
    }
    
    // caculate time
    double avg_time = 0.0;
    for (double t : times) avg_time += t;
    avg_time /= test_runs;
    
    // double min_time = *std::min_element(times.begin(), times.end());
    // double max_time = *std::max_element(times.begin(), times.end());
    
    // vertify
    bool correct = cpu_cache::verify_result(C_ref, C, M, N);
    
    // caculate performance
    double gflops = calculate_gflops(M, N, K, avg_time);
    
    // print result
    std::cout << std::left << std::setw(25) << name 
              << std::right << std::fixed << std::setprecision(4)
              << " | time: " << std::setw(10) << avg_time << " ms"
              << " | GFLOPS: " << std::setw(8) << gflops
              << " | accurancy: " << (correct ? "pass" : "fail")
              << std::endl;
}

// tiling
void run_blocked_tests(
    const float* A, const float* B, float* C,
    size_t M, size_t N, size_t K,
    const float* C_ref,
    const std::vector<size_t>& block_sizes)
{
    std::cout << "\n--- Cache-Aware tiling (different block size) ---" << std::endl;
    
    for (size_t bs : block_sizes) {
        auto func = [bs](const float* a, const float* b, float* c, size_t m, size_t n, size_t k) {
            cpu_cache::matmul_blocked(a, b, c, m, n, k, bs);
        };
        
        std::string name = "Blocked (b=" + std::to_string(bs) + ")";
        run_algorithm_test(name, func, A, B, C, M, N, K, C_ref);
    }
}

// recursive:
void run_recursive_tests(
    const float* A, const float* B, float* C,
    size_t M, size_t N, size_t K,
    const float* C_ref,
    const std::vector<size_t>& thresholds)
{
    std::cout << "\n--- Cache-Oblivious (different threshold) ---" << std::endl;
    
    for (size_t t : thresholds) {
        auto func = [t](const float* a, const float* b, float* c, size_t m, size_t n, size_t k) {
            cpu_cache::matmul_recursive(a, b, c, m, n, k, t);
        };
        
        std::string name = "Recursive (t=" + std::to_string(t) + ")";
        run_algorithm_test(name, func, A, B, C, M, N, K, C_ref);
    }
}

int main() {
    // test config
    std::vector<TestConfig> test_configs = {
        {128, 128, 128, "small 2n"},
        {256, 256, 256, "mid 2n"},
        {512, 512, 512, "max 2n"},
        {1024, 1024, 1024, "supermax 2n"},
        {300, 300, 300, "mid not 2n"},
        {500, 500, 500, "max not 2n"},
        {768, 768, 768, "NN usual"}
    };
    
    // test param:
    std::vector<size_t> block_sizes = {16, 32, 64, 128};
    std::vector<size_t> thresholds = {16, 32, 64, 128};
    
    std::cout << "  Naive Blocked Recursive\n";
    std::cout << std::endl;
    
    // run test for each test config
    for (const auto& config : test_configs) {
        size_t M = config.M;
        size_t N = config.N;
        size_t K = config.K;
        
        std::vector<float> A(M * K);
        std::vector<float> B(K * N);
        std::vector<float> C(M * N);
        std::vector<float> C_ref(M * N);
        
        generate_random_matrix(A.data(), M * K);
        generate_random_matrix(B.data(), K * N);
        
        // use naive as reference
        cpu_cache::matmul_naive(A.data(), B.data(), C_ref.data(), M, N, K);
        
        print_test_info(config);
        
        std::cout << "\n--- formal test ---" << std::endl;
        run_algorithm_test("Naive (baseline)", cpu_cache::matmul_naive, A.data(), B.data(), C.data(), M, N, K, C_ref.data());
        
        run_blocked_tests(A.data(), B.data(), C.data(), M, N, K, C_ref.data(), block_sizes);
        
        run_recursive_tests(A.data(), B.data(), C.data(), M, N, K, C_ref.data(), thresholds);
        
        std::cout << std::endl;
    }

    return 0;
}