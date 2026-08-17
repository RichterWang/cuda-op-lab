#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <vector>

#include "../sgemm.h"

namespace
{
void check_cuda(cudaError_t status, const char *where)
{
    if (status != cudaSuccess) throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(status));
}

void check_cublas(cublasStatus_t status, const char *where)
{
    if (status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::string(where) + " failed");
}

void sgemm_cpu_double(const float *a, const float *b, float *c, int m, int n, int k)
{
    for (int row = 0; row < m; ++row)
    {
        for (int col = 0; col < n; ++col)
        {
            double sum = 0.0;
            for (int inner = 0; inner < k; ++inner) sum += static_cast<double>(a[row * k + inner]) * static_cast<double>(b[inner * n + col]);
            c[row * n + col] = static_cast<float>(sum);
        }
    }
}

void cublas_sgemm_row_major(cublasHandle_t handle, const float *a, const float *b, float *c, int m, int n, int k)
{
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // cuBLAS is col major；caculate C^T=B^T*A^T，equals to row major C=A*B
    check_cublas(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, b, n, a, k, &beta, c, n), "cublasSgemm");
}
} // namespace

// work flow: define -> check answer -> warmup -> get the time -> output -> release
int main()
{
    // matrix shape define
    // matrix size(256,384,320)
    constexpr int m = 1024;  
    constexpr int n = 512;
    constexpr int k = 256;
    constexpr int warmup = 5;      // warmup iterations
    constexpr int iterations = 30; // timed iterations

    // input matrices, host memory
    std::vector<float> h_a(m * k);
    std::vector<float> h_b(k * n);
    // result for cpu,cublas, naive and shared memory kernel
    std::vector<float> h_cpu(m * n);
    std::vector<float> h_cublas(m * n);
    std::vector<float> h_naive(m * n);
    std::vector<float> h_shared_mem(m * n);
    std::vector<float> h_tile_reg(m * n);

    // random generation of input matrix
    std::mt19937 generator(42);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    for (float &value : h_a) value = distribution(generator);
    for (float &value : h_b) value = distribution(generator);

    sgemm_cpu_double(h_a.data(), h_b.data(), h_cpu.data(), m, n, k);

    // state of dvice ptr
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    cublasHandle_t handle = nullptr;
    cudaEvent_t start = nullptr, stop = nullptr;

    try
    { // cpy data from host to device
        check_cuda(cudaMalloc(&d_a, sizeof(float) * h_a.size()), "cudaMalloc(a)");
        check_cuda(cudaMalloc(&d_b, sizeof(float) * h_b.size()), "cudaMalloc(b)");
        check_cuda(cudaMalloc(&d_c, sizeof(float) * h_cublas.size()), "cudaMalloc(c)");

        check_cuda(cudaMemcpy(d_a, h_a.data(), sizeof(float) * h_a.size(), cudaMemcpyHostToDevice), "copy a");
        check_cuda(cudaMemcpy(d_b, h_b.data(), sizeof(float) * h_b.size(), cudaMemcpyHostToDevice), "copy b");

        check_cublas(cublasCreate(&handle), "cublasCreate");

        cublas_sgemm_row_major(handle, d_a, d_b, d_c, m, n, k);
        check_cuda(cudaMemcpy(h_cublas.data(), d_c, sizeof(float) * h_cublas.size(), cudaMemcpyDeviceToHost), "copy c");

        // check the answer between cubals & cpu result
        float max_abs_error = 0.0f;
        float max_rel_error = 0.0f;
        for (size_t index = 0; index < h_cpu.size(); ++index)
        {
            const float abs_error = std::fabs(h_cpu[index] - h_cublas[index]); // fabs func get absolute value of float
            const float denominator = std::max(1.0f, std::fabs(h_cpu[index])); // prepare for relative error
            max_abs_error = std::max(max_abs_error, abs_error);
            max_rel_error = std::max(max_rel_error, abs_error / denominator);
        }

        // naive result check
        check_cuda(cudaMemset(d_c, 0, sizeof(float) * h_naive.size()), "clear naive output");

        cuda_op_lab::sgemm::launch_sgemm_naive(d_a, d_b, d_c, m, n, k); // execute naive kernel

        check_cuda(cudaGetLastError(), "launch naive kernel");    // add sting followed to the front of error output
        check_cuda(cudaDeviceSynchronize(), "wait naive kernel"); // let cpu wait for gpu

        check_cuda(cudaMemcpy(h_naive.data(), d_c, sizeof(float) * h_naive.size(), cudaMemcpyDeviceToHost), "copy naive output");

        float naive_max_abs_error = 0.0f;
        float naive_max_rel_error = 0.0f;

        for (size_t index = 0; index < h_cpu.size(); ++index)
        {
            const float abs_error = std::fabs(h_cpu[index] - h_naive[index]);
            const float denominator = std::max(1.0f, std::fabs(h_cpu[index]));
            naive_max_abs_error = std::max(naive_max_abs_error, abs_error);
            naive_max_rel_error = std::max(naive_max_rel_error, abs_error / denominator);
        }
        // end naive result check

        // shared mem result check
        check_cuda(cudaMemset(d_c, 0, sizeof(float) * h_shared_mem.size()), "clear shared-memory c");

        cuda_op_lab::sgemm::launch_sgemm_shared_mem(d_a, d_b, d_c, m, n, k);

        check_cuda(cudaGetLastError(), "shared-memory kernel launch");
        check_cuda(cudaDeviceSynchronize(), "shared-memory kernel synchronize");

        check_cuda(cudaMemcpy(h_shared_mem.data(), d_c, sizeof(float) * h_shared_mem.size(), cudaMemcpyDeviceToHost), "copy shared-memory c");

        float shared_mem_max_abs_error = 0.0f;
        float shared_mem_max_rel_error = 0.0f;
        for (size_t index = 0; index < h_cpu.size(); ++index)
        {
            const float abs_error = std::fabs(h_cpu[index] - h_shared_mem[index]);
            const float denominator = std::max(1.0f, std::fabs(h_cpu[index]));
            shared_mem_max_abs_error = std::max(shared_mem_max_abs_error, abs_error);
            shared_mem_max_rel_error = std::max(shared_mem_max_rel_error, abs_error / denominator);
        }
        // end shared mem result check

        // shared memory add tile reg result check
        check_cuda(cudaMemset(d_c, 0, sizeof(float) * h_tile_reg.size()), "clear tiled reg c");

        cuda_op_lab::sgemm::launch_sgemm_register_tiled(d_a, d_b, d_c, m, n, k);

        check_cuda(cudaGetLastError(), "shared-memory kernel launch");
        check_cuda(cudaDeviceSynchronize(), "shared-memory kernel synchronize");

        check_cuda(cudaMemcpy(h_tile_reg.data(), d_c, sizeof(float) * h_tile_reg.size(), cudaMemcpyDeviceToHost), "copy shared-memory c");

        float reg_tile_max_abs_error = 0.0f;
        float reg_tile_max_rel_error = 0.0f;
        for (size_t index = 0; index < h_cpu.size(); ++index)
        {
            const float abs_error = std::fabs(h_cpu[index] - h_tile_reg[index]);
            const float denominator = std::max(1.0f, std::fabs(h_cpu[index]));
            reg_tile_max_abs_error = std::max(reg_tile_max_abs_error, abs_error);
            reg_tile_max_rel_error = std::max(reg_tile_max_rel_error, abs_error / denominator);
        }
        // end shared memory add tile reg result check

        // warmup ==========================================================================================================
        // cublas kernel warmup
        for (int index = 0; index < warmup; ++index) cublas_sgemm_row_major(handle, d_a, d_b, d_c, m, n, k);
        
        check_cuda(cudaDeviceSynchronize(), "warmup synchronize");
        // end cublas kernel warmup

        // naive kernel warmup
        for (int index = 0; index < warmup; ++index) cuda_op_lab::sgemm::launch_sgemm_naive(d_a, d_b, d_c, m, n, k);
        
        check_cuda(cudaGetLastError(), "naive warmup launch");
        check_cuda(cudaDeviceSynchronize(), "naive warmup synchronize");
        // end naive kernel warmup

        // shared mem kernel warmup
        for (int index = 0; index < warmup; ++index) cuda_op_lab::sgemm::launch_sgemm_shared_mem(d_a, d_b, d_c, m, n, k);

        check_cuda(cudaGetLastError(), "shared mem warmup launch");
        check_cuda(cudaDeviceSynchronize(), "shared mem warmup synchronize");
        // end shared mem kernel warmup

        // reg tile kernel warmup
        for (int index = 0; index < warmup; ++index) cuda_op_lab::sgemm::launch_sgemm_register_tiled(d_a, d_b, d_c, m, n, k);

        check_cuda(cudaGetLastError(), "reg tile warmup launch");
        check_cuda(cudaDeviceSynchronize(), "reg tile warmup synchronize");
        // end reg tile kernel warmup

        // time recoding ====================================================================================================
        // use cuda event to record time
        check_cuda(cudaEventCreate(&start), "create start event");
        check_cuda(cudaEventCreate(&stop), "create stop event");

        // naive timer
        check_cuda(cudaEventRecord(start), "record naive start");

        for (int index = 0; index < iterations; ++index) cuda_op_lab::sgemm::launch_sgemm_naive(d_a, d_b, d_c, m, n, k);

        check_cuda(cudaGetLastError(), "naive timed launch");
        check_cuda(cudaEventRecord(stop), "record naive stop");
        check_cuda(cudaEventSynchronize(stop), "wait naive stop");
        // standard process: check launch error, get end point, wait for stop to finish

        float naive_elapsed_ms = 0.0f;
        check_cuda(cudaEventElapsedTime(&naive_elapsed_ms, start, stop), "naive elapsed time");
        // caculate time from start to stop and store it to location ptr point to.

        const float naive_average_ms = naive_elapsed_ms / iterations;
        const double naive_gflops = 2.0 * static_cast<double>(m) * n * k / (naive_average_ms * 1.0e6);
        // end naive timer

        // shared_mem timer
        check_cuda(cudaEventRecord(start), "record shared mem start");

        for(int index = 0; index < iterations; ++index) cuda_op_lab::sgemm::launch_sgemm_shared_mem(d_a, d_b, d_c, m, n, k);

        check_cuda(cudaGetLastError(), "shared-memory kernel launch");
        check_cuda(cudaEventRecord(stop), "record shared-memory stop");
        check_cuda(cudaEventSynchronize(stop), "wait shared-memory stop");

        float shared_mem_elapsed_ms = 0.0f;
        check_cuda(cudaEventElapsedTime(&shared_mem_elapsed_ms, start, stop), "shared mem elapesd time");

        const float shared_mem_average_ms = shared_mem_elapsed_ms / iterations;
        const double shared_mem_gflops = 2.0 * static_cast<double>(m) * n * k / (shared_mem_average_ms * 1.0e6);
        // end shared_mem timer

        // reg tile timer
        check_cuda(cudaEventRecord(start), "record reg tile start");

        for(int index = 0; index < iterations; ++index) cuda_op_lab::sgemm::launch_sgemm_register_tiled(d_a, d_b, d_c, m, n, k);

        check_cuda(cudaGetLastError(), "reg tile kernel launch");
        check_cuda(cudaEventRecord(stop), "record reg tile stop");
        check_cuda(cudaEventSynchronize(stop), "wait reg tile stop");

        float reg_tile_elapsed_ms = 0.0f;
        check_cuda(cudaEventElapsedTime(&reg_tile_elapsed_ms, start, stop), "reg tile elapesd time");

        const float reg_tile_average_ms = reg_tile_elapsed_ms / iterations;
        const double reg_tile_gflops = 2.0 * static_cast<double>(m) * n * k / (reg_tile_average_ms * 1.0e6);
        // end reg tile timer

        // cublas timer
        check_cuda(cudaEventRecord(start), "record cuBLAS start");

        for (int index = 0; index < iterations; ++index) cublas_sgemm_row_major(handle, d_a, d_b, d_c, m, n, k);

        check_cuda(cudaEventRecord(stop), "record cuBLAS stop");
        check_cuda(cudaEventSynchronize(stop), "wait cuBLAS stop");

        float cublas_elapsed_ms = 0.0f;
        check_cuda(cudaEventElapsedTime(&cublas_elapsed_ms, start, stop), "cuBLAS elapsed time");

        const float cublas_average_ms = cublas_elapsed_ms / iterations;
        const double cublas_gflops = 2.0 * static_cast<double>(m) * n * k / (cublas_average_ms * 1.0e6);
        // end cubals timer

        // cal relative percent
        const double naive_relative_percent = 100.0 * naive_gflops / cublas_gflops;
        const double shared_mem_relative_percent = 100.0 * shared_mem_gflops / cublas_gflops;
        const double reg_tile_relative_percent = 100.0 * reg_tile_gflops / cublas_gflops;

        // result output ========================================================================================
        // basic
        std::printf("shape: M=%d N=%d K=%d\n", m, n, k);
        std::printf("max abs error: %.8e\n", max_abs_error);
        std::printf("max rel error: %.8e\n", max_rel_error);
        std::printf("cuBLAS: %.4f ms, %.2f GFLOPS\n", cublas_average_ms, cublas_gflops);

        // naive
        std::printf("naive:  %.4f ms, %.2f GFLOPS\n", naive_average_ms, naive_gflops);
        std::printf("naive / cuBLAS: %.2f%%\n", naive_relative_percent);
        std::printf("naive max abs error: %.8e\n", naive_max_abs_error);
        std::printf("naive max rel error: %.8e\n", naive_max_rel_error);

        // shared mem
        std::printf("shared-memory: %.4f ms, %.2f GFLOPS\n", shared_mem_average_ms, shared_mem_gflops);
        std::printf("shared-memory / cuBLAS: %.2f%%\n", shared_mem_relative_percent);
        std::printf("shared-memory max abs error: %.8e\n", shared_mem_max_abs_error);
        std::printf("shared-memory max rel error: %.8e\n", shared_mem_max_rel_error);

        // reg tile
        std::printf("tiled-register: %.4f ms, %.2f GFLOPS\n", reg_tile_average_ms, reg_tile_gflops);
        std::printf("tiled-register / cuBLAS: %.2f%%\n", reg_tile_relative_percent);
        std::printf("tiled-register max abs error: %.8e\n", reg_tile_max_abs_error);
        std::printf("tiled-register max rel error: %.8e\n", reg_tile_max_rel_error);

        // release the resource ======================================================================================
        // release cuda event
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        // release cublas handle
        cublasDestroy(handle);
        // sdandard cuda library design, create content -> use -> destory content

        // free cuda memory(while ptr will destory automatically with the function)
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);

        // if success, exit normally
        return max_rel_error < 1.0e-4f && naive_max_rel_error < 1.0e-4f && shared_mem_max_rel_error < 1.0e-4f && reg_tile_max_rel_error < 1.0e-4f ? 0 : 1;
    }
    catch (const std::exception &error) // catch all the error under cpp std
    {
        // print error message
        std::fprintf(stderr, "error: %s\n", error.what());

        // clean resource
        if (start) cudaEventDestroy(start);
        if (stop) cudaEventDestroy(stop);
        if (handle) cublasDestroy(handle);
        if (d_a) cudaFree(d_a);
        if (d_b) cudaFree(d_b);
        if (d_c) cudaFree(d_c);

        // exit abnormally
        return 1;
    }
}