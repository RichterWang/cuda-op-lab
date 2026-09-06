// Occupancy experiment. Separate executable from bench_rms_norm, because this is
// a one-off diagnostic whose output is a decision, not a regression metric.
//
// The finding it has to explain. v1 reaches 99.9% of the copy ceiling at
// cols=4096 and 5120, but only 87.7% at 8192 and 83.8% at 11008. select_block_size
// crosses from 512 to 1024 threads at exactly that boundary, which leaves two
// candidate causes confounded: occupancy, and threads idling in the final loop
// iteration.
//
// Experiment A sweeps block size per shape. Not conclusive on its own, since
// moving the block size moves occupancy, tail shape and per-thread element count
// together, but it shows which configuration wins.
//
// Experiment B is the controlled one. Block size is held at 512 while dynamic
// shared memory is requested in increasing amounts to force blocks/SM down. The
// kernel never reads that memory; it is reserved at launch out of a fixed per-SM
// budget, so blocks/SM is the only thing that changes.
//
// Both candidates were wrong. The numbers, on an RTX 3070 Ti Laptop (sm_86):
//
//   Occupancy is not it. The occupancy API grants 2 blocks/SM at 512 threads and
//   1 at 1024, both 66.7% resident, yet 512 is 77% faster at cols=4096 (0.1620 vs
//   0.2860 ms). At 42 registers per thread the register file, not the 1536-thread
//   limit, is what caps blocks/SM. Experiment B then halves occupancy directly at
//   fixed block size and cols=8192 moves only 96.9% -> 95.9%: a bandwidth-bound
//   kernel does not need occupancy to hide latency.
//
//   Tail waste is not it either. cols=8192 at 1024 threads has vec_cols exactly
//   1024, so there is no tail at all, and it still measures 91.0%. cols=5120 at
//   512 threads leaves three quarters of the block idle in the last iteration and
//   reaches 99.6%.
//
//   L1 capacity is what tracks it. Shared memory and L1 share one 100 KB per-SM
//   allocation, so experiment B's shared request also shrinks L1. At unchanged
//   occupancy, 0 -> 40 KB of shared costs cols=8192 96.9% -> 89.4% and cols=11008
//   90.7% -> 80.8%, while cols=4096 and 5120 do not move. The shapes that degrade
//   are the ones whose rows stop fitting in L1, which points at pass 2 re-reads
//   escaping to L2 as the real cause of the original gap.
//
// Experiment B's shared request doubles as an L1 capacity sweep, which was not
// the intent but is the more informative of the two axes it moves.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "rms_norm.h"

namespace {

using cuda_op_lab::rms_norm::launch_copy_baseline;
using cuda_op_lab::rms_norm::launch_rmsnorm_probe;
using cuda_op_lab::rms_norm::ProbeInfo;
using cuda_op_lab::rms_norm::query_rmsnorm_probe;

constexpr int kWarmupIters = 20;
constexpr int kTimedIters = 100;

#define CUDA_CHECK(expr)                                                                                      \
    do                                                                                                        \
    {                                                                                                         \
        const cudaError_t status = (expr);                                                                     \
        if (status != cudaSuccess)                                                                             \
        {                                                                                                     \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(status), __FILE__, __LINE__);  \
            std::exit(EXIT_FAILURE);                                                                           \
        }                                                                                                     \
    } while (0)

struct Shape
{
    int rows;
    int cols;
    const char* note;
};

// Only shapes divisible by 8, since the probe covers the vectorized path. 4096
// and 5120 are the two that already reach the ceiling and act as controls; 8192
// and 11008 are the ones being explained.
const std::vector<Shape> kShapes = {
    {4096, 4096, "reaches ceiling in v1; row is 8 KB"},
    {4096, 5120, "reaches ceiling in v1; row is 10 KB"},
    {2048, 8192, "87.7% in v1; row is 16 KB, and no tail at 1024 threads"},
    {2048, 11008, "83.8% in v1; row is 21.5 KB"},
};

const std::vector<int> kBlockSizes = {128, 256, 512, 1024};

void fill_input(std::vector<__nv_bfloat16>& host, size_t count)
{
    std::mt19937 engine(5678u);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (size_t index = 0; index < count; ++index) host[index] = __float2bfloat16(dist(engine));
}

void fill_gamma(std::vector<__nv_bfloat16>& host, int cols)
{
    std::mt19937 engine(1234u);
    std::normal_distribution<float> dist(1.0f, 0.1f);
    for (int col = 0; col < cols; ++col) host[col] = __float2bfloat16(dist(engine));
}

float time_probe(const __nv_bfloat16* x, const __nv_bfloat16* gamma, __nv_bfloat16* y, int rows, int cols, float eps, int block_size,
                 size_t dynamic_shared_bytes)
{
    for (int iter = 0; iter < kWarmupIters; ++iter) launch_rmsnorm_probe(x, gamma, y, rows, cols, eps, block_size, dynamic_shared_bytes, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int iter = 0; iter < kTimedIters; ++iter) launch_rmsnorm_probe(x, gamma, y, rows, cols, eps, block_size, dynamic_shared_bytes, nullptr);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return elapsed_ms / static_cast<float>(kTimedIters);
}

float time_copy(const __nv_bfloat16* x, __nv_bfloat16* y, int rows, int cols)
{
    for (int iter = 0; iter < kWarmupIters; ++iter) launch_copy_baseline(x, y, rows, cols, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int iter = 0; iter < kTimedIters; ++iter) launch_copy_baseline(x, y, rows, cols, nullptr);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return elapsed_ms / static_cast<float>(kTimedIters);
}

double effective_gb(int rows, int cols)
{
    return static_cast<double>(rows) * cols * sizeof(__nv_bfloat16) * 2.0 / 1.0e9;
}

// How many of a block's threads are active in the final loop iteration. 100%
// means the row divides evenly and there is no tail.
//
// When vec_cols < block_size the loop body never runs for the surplus threads at
// all, which is a different situation from a partial final iteration, so it is
// reported separately rather than folded into the same percentage.
double tail_efficiency(int cols, int block_size)
{
    const int vec_cols = cols / 8;
    if (vec_cols <= block_size) return 100.0 * static_cast<double>(vec_cols) / static_cast<double>(block_size);

    const int iterations = (vec_cols + block_size - 1) / block_size;
    const int active_in_last = vec_cols - (iterations - 1) * block_size;
    return 100.0 * static_cast<double>(active_in_last) / static_cast<double>(block_size);
}

// Vectors each thread handles on average. Below 1.0 means some threads get no
// work at all: cols=4096 over 1024 threads gives 512 vectors for 1024 threads,
// so half the block exits at the first loop test.
double vectors_per_thread(int cols, int block_size)
{
    return static_cast<double>(cols / 8) / static_cast<double>(block_size);
}

}  // namespace

int main()
{
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));

    cudaDeviceProp props{};
    CUDA_CHECK(cudaGetDeviceProperties(&props, device));

    const int max_threads_per_sm = props.maxThreadsPerMultiProcessor;
    const int shared_per_sm = static_cast<int>(props.sharedMemPerMultiprocessor);

    std::printf("device: %s (sm_%d%d), %d SMs\n", props.name, props.major, props.minor, props.multiProcessorCount);
    std::printf("max threads per SM: %d\n", max_threads_per_sm);
    std::printf("max shared per SM: %d KB, max shared per block: %d KB\n", shared_per_sm / 1024, static_cast<int>(props.sharedMemPerBlock) / 1024);
    std::printf("\n");

    // Static picture first: what the occupancy API grants per block size, with no
    // extra shared memory requested. This is a property of the kernel and the
    // hardware, independent of shape.
    std::printf("=== occupancy by block size (no extra shared memory) ===\n");
    std::printf("%10s %8s %10s %12s %12s %10s\n", "block", "regs", "static_sh", "blocks/SM", "threads/SM", "occupancy");
    for (int block_size : kBlockSizes)
    {
        const ProbeInfo info = query_rmsnorm_probe(block_size, 0);
        std::printf("%10d %8d %9dB %12d %12d %9.1f%%\n", block_size, info.registers_per_thread, info.static_shared_bytes, info.max_blocks_per_sm,
                    info.resident_threads_per_sm, 100.0 * info.resident_threads_per_sm / max_threads_per_sm);
    }
    std::printf("\n");

    const float eps = 1.0e-6f;

    // Experiment A: block size sweep. Confounded by design, but it shows which
    // configuration wins and whether select_block_size picks it.
    std::printf("=== experiment A: block size sweep ===\n");
    std::printf("tail%% is the fraction of a block still active in the final loop iteration;\n");
    std::printf("vec/thread below 1.0 means some threads never enter the loop at all.\n\n");

    for (const Shape& shape : kShapes)
    {
        const size_t count = static_cast<size_t>(shape.rows) * shape.cols;
        const size_t bytes = count * sizeof(__nv_bfloat16);

        std::vector<__nv_bfloat16> host_x(count);
        std::vector<__nv_bfloat16> host_gamma(shape.cols);
        fill_input(host_x, count);
        fill_gamma(host_gamma, shape.cols);

        __nv_bfloat16* device_x = nullptr;
        __nv_bfloat16* device_gamma = nullptr;
        __nv_bfloat16* device_y = nullptr;
        CUDA_CHECK(cudaMalloc(&device_x, bytes));
        CUDA_CHECK(cudaMalloc(&device_gamma, static_cast<size_t>(shape.cols) * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&device_y, bytes));
        CUDA_CHECK(cudaMemcpy(device_x, host_x.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(device_gamma, host_gamma.data(), static_cast<size_t>(shape.cols) * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

        const double gb = effective_gb(shape.rows, shape.cols);
        const float copy_ms = time_copy(device_x, device_y, shape.rows, shape.cols);
        const double copy_gb_per_s = gb / (copy_ms / 1.0e3);

        std::printf("rows=%d cols=%d (%s)\n", shape.rows, shape.cols, shape.note);
        std::printf("  copy ceiling: %.2f GB/s\n", copy_gb_per_s);
        std::printf("  %8s %10s %10s %8s %10s %12s %8s %10s\n", "block", "ms", "GB/s", "%copy", "blocks/SM", "occupancy", "tail%", "vec/thread");

        for (int block_size : kBlockSizes)
        {
            const ProbeInfo info = query_rmsnorm_probe(block_size, 0);
            const float ms = time_probe(device_x, device_gamma, device_y, shape.rows, shape.cols, eps, block_size, 0);
            const double gb_per_s = gb / (ms / 1.0e3);

            std::printf("  %8d %10.4f %10.2f %7.1f%% %10d %11.1f%% %7.1f%% %10.2f\n", block_size, ms, gb_per_s, 100.0 * gb_per_s / copy_gb_per_s,
                        info.max_blocks_per_sm, 100.0 * info.resident_threads_per_sm / max_threads_per_sm,
                        tail_efficiency(shape.cols, block_size), vectors_per_thread(shape.cols, block_size));
        }
        std::printf("\n");

        CUDA_CHECK(cudaFree(device_x));
        CUDA_CHECK(cudaFree(device_gamma));
        CUDA_CHECK(cudaFree(device_y));
    }

    // Experiment B: block size fixed at 512, occupancy forced down by requesting
    // shared memory the kernel never touches. This is the controlled comparison:
    // instruction mix, loop structure and tail shape are all identical across
    // rows, so any difference in time is attributable to blocks/SM.
    std::printf("=== experiment B: occupancy forced down at fixed block size 512 ===\n");
    std::printf("The requested shared memory is never read. It only consumes the per-SM budget,\n");
    std::printf("so block size, loop structure and tail shape stay fixed while blocks/SM drops.\n");
    std::printf("Shared and L1 come out of the same 100 KB, so this is also an L1 capacity sweep:\n");
    std::printf("watch the rows where occupancy holds steady but throughput still falls.\n\n");

    // Shared memory per SM on sm_86 is 100 KB. Requesting 60 KB per block is what
    // finally cuts blocks/SM from 2 to 1 here; the smaller requests leave
    // occupancy untouched and vary only how much of that 100 KB is left for L1.
    const std::vector<size_t> shared_requests = {0, 20 * 1024, 40 * 1024, 60 * 1024};

    for (const Shape& shape : kShapes)
    {
        const size_t count = static_cast<size_t>(shape.rows) * shape.cols;
        const size_t bytes = count * sizeof(__nv_bfloat16);

        std::vector<__nv_bfloat16> host_x(count);
        std::vector<__nv_bfloat16> host_gamma(shape.cols);
        fill_input(host_x, count);
        fill_gamma(host_gamma, shape.cols);

        __nv_bfloat16* device_x = nullptr;
        __nv_bfloat16* device_gamma = nullptr;
        __nv_bfloat16* device_y = nullptr;
        CUDA_CHECK(cudaMalloc(&device_x, bytes));
        CUDA_CHECK(cudaMalloc(&device_gamma, static_cast<size_t>(shape.cols) * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&device_y, bytes));
        CUDA_CHECK(cudaMemcpy(device_x, host_x.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(device_gamma, host_gamma.data(), static_cast<size_t>(shape.cols) * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

        const double gb = effective_gb(shape.rows, shape.cols);
        const float copy_ms = time_copy(device_x, device_y, shape.rows, shape.cols);
        const double copy_gb_per_s = gb / (copy_ms / 1.0e3);

        std::printf("rows=%d cols=%d\n", shape.rows, shape.cols);
        std::printf("  %10s %12s %12s %10s %10s %8s\n", "shared/blk", "blocks/SM", "occupancy", "ms", "GB/s", "%copy");

        for (size_t request : shared_requests)
        {
            const ProbeInfo info = query_rmsnorm_probe(512, request);
            const float ms = time_probe(device_x, device_gamma, device_y, shape.rows, shape.cols, eps, 512, request);
            const double gb_per_s = gb / (ms / 1.0e3);

            std::printf("  %9zuK %12d %11.1f%% %10.4f %10.2f %7.1f%%\n", request / 1024, info.max_blocks_per_sm,
                        100.0 * info.resident_threads_per_sm / max_threads_per_sm, ms, gb_per_s, 100.0 * gb_per_s / copy_gb_per_s);
        }
        std::printf("\n");

        CUDA_CHECK(cudaFree(device_x));
        CUDA_CHECK(cudaFree(device_gamma));
        CUDA_CHECK(cudaFree(device_y));
    }

    return EXIT_SUCCESS;
}
