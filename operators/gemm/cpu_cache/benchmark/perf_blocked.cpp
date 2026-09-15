// Performance profiling for blocked (cache-aware) matrix multiplication
// Usage: ./perf_blocked <matrix_size> <block_size>
// Example: perf stat -e cache-misses,cache-references ./perf_blocked 1024 64

#include "../include/cpu_matmul.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <random>

using namespace std;

// simple timer
class Timer {
public:
    void start() {
        start_time = chrono::high_resolution_clock::now();
    }
    
    double stop() {
        auto end_time = chrono::high_resolution_clock::now();
        chrono::duration<double, milli> elapsed = end_time - start_time;
        return elapsed.count();
    }
    
private:
    chrono::time_point<chrono::high_resolution_clock> start_time;
};

// generate random matrix
void generate_matrix(float* matrix, size_t size) {
    mt19937 gen(42);
    uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < size; ++i) {
        matrix[i] = dist(gen);
    }
}

// calculate GFLOPS
double calculate_gflops(size_t M, size_t N, size_t K, double time_ms) {
    double flops = 2.0 * M * N * K;
    return flops / (time_ms * 1e6);
}

int main(int argc, char** argv) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <matrix_size> <block_size>" << endl;
        cerr << "Example: " << argv[0] << " 1024 64" << endl;
        return 1;
    }

    const size_t size = atoi(argv[1]);
    const size_t block_size = atoi(argv[2]);
    
    if (size <= 0 || size > 4096) {
        cerr << "Error: matrix size must be in range (0, 4096]" << endl;
        return 1;
    }
    
    if (block_size <= 0 || block_size > 512) {
        cerr << "Error: block size must be in range (0, 512]" << endl;
        return 1;
    }

    cout << "=== Blocked (Cache-Aware) Algorithm Performance Test ===" << endl;
    cout << "Matrix size: " << size << "x" << size << endl;
    cout << "Block size:  " << block_size << endl;
    cout << "Algorithm:   Cache-aware tiling" << endl;
    cout << endl;

    // allocate matrices
    vector<float> A(size * size);
    vector<float> B(size * size);
    vector<float> C(size * size, 0.0f);

    // initialize
    generate_matrix(A.data(), size * size);
    generate_matrix(B.data(), size * size);

    // warmup (3 runs)
    cout << "Warming up..." << endl;
    for (int i = 0; i < 3; ++i) {
        cpu_cache::matmul_blocked(A.data(), B.data(), C.data(), size, size, size, block_size);
    }

    // performance measurement (10 runs)
    cout << "Running performance test (10 iterations)..." << endl;
    const int test_runs = 10;
    vector<double> times;
    times.reserve(test_runs);
    
    Timer timer;
    for (int i = 0; i < test_runs; ++i) {
        fill(C.begin(), C.end(), 0.0f);
        timer.start();
        cpu_cache::matmul_blocked(A.data(), B.data(), C.data(), size, size, size, block_size);
        times.push_back(timer.stop());
    }

    // calculate statistics
    double avg_time = 0.0;
    for (double t : times) avg_time += t;
    avg_time /= test_runs;
    
    double gflops = calculate_gflops(size, size, size, avg_time);

    // prevent compiler optimization
    volatile double checksum = 0.0;
    for (size_t i = 0; i < min(size_t(1000), size * size); ++i) {
        checksum += C[i];
    }

    // print results
    cout << endl;
    cout << "=== Results ===" << endl;
    cout << "Average time: " << avg_time << " ms" << endl;
    cout << "Performance:  " << gflops << " GFLOPS" << endl;
    cout << "Checksum:     " << checksum << " (prevent optimization)" << endl;
    cout << endl;
    cout << "Note: Use with 'perf stat' for detailed cache analysis" << endl;
    cout << "Example: perf stat -e L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses " 
         << argv[0] << " " << size << " " << block_size << endl;

    return 0;
}
