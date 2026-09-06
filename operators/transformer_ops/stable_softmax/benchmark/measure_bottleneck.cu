// Turns two inferences from the README into direct measurements.
//
// Both were derived by arithmetic on the benchmark's bandwidth numbers, which
// makes them plausible but not verified. Nsight Compute is not available in this
// environment, so instead of reading hardware counters this program measures the
// same two things using only the CUDA runtime API.
//
// Claim A: "the three passes over a row hit DRAM at cols=8192, but stay in L1 at
//          cols=1024."
//   Measured by timing two probe kernels that differ only in how many times they
//   read the row (1 vs 3), with the exp() removed from both. If the extra reads
//   really go to DRAM, the 3-read probe should take about 2x as long, because it
//   moves 4N bytes instead of 2N. If they hit in L1, the ratio stays near 1.
//   Also reported: the same ratio for the real v1 kernel, which should track the
//   probe ratio if traffic is what separates the shapes.
//
// Claim A2: "v2 is faster on narrow rows because it shortened the critical path,
//           not because it moved fewer bytes."
//   The 256-thread probes show ratios above 2.0 on narrow rows, which no amount
//   of traffic can explain (4N/2N = 2 is the ceiling). So the cost must be the
//   serial dependency between passes. If that reading is right, v2 should land
//   near the 1-read probe even though it still makes three passes: same traffic,
//   shorter critical path. Reported as the v2/1read column.
//
// Claim A3: "a two-pass online algorithm has little left to win, because the
//           adaptive launcher already turned the extra passes into L1 hits."
//   The 256-thread probe cannot answer this: it measures traffic under v1's
//   original block size, not the 1024 threads the adaptive launcher actually
//   uses on long rows. Running the same probe pair at 1024 threads measures the
//   remaining DRAM traffic directly. A ratio near 1.0 means the extra passes are
//   already cached and v3 can only remove instructions, not bytes.
//
// Claim B: "a 1024-thread block collapses at cols=1024 because only one block
//          fits per SM, so there is nothing left to hide barrier latency."
//   Measured with cudaOccupancyMaxActiveBlocksPerMultiprocessor, which reports
//   resident blocks per SM directly. Resident warps and blocks are what decide
//   whether a stalled barrier can be overlapped with other work.

#include <cuda_runtime.h>

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "stable_softmax.h"

namespace {

void check_cuda(cudaError_t status, const char *where)
{
    if (status != cudaSuccess) throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(status));
}

using LaunchFn = void (*)(const float *, float *, int, int, cudaStream_t);

float time_kernel(LaunchFn fn, const float *d_x, float *d_y, int rows, int cols, int warmup, int iterations)
{
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    check_cuda(cudaEventCreate(&start), "create start event");
    check_cuda(cudaEventCreate(&stop), "create stop event");

    for (int index = 0; index < warmup; ++index) fn(d_x, d_y, rows, cols, nullptr);
    check_cuda(cudaGetLastError(), "warmup launch");
    check_cuda(cudaDeviceSynchronize(), "warmup synchronize");

    check_cuda(cudaEventRecord(start), "record start");
    for (int index = 0; index < iterations; ++index) fn(d_x, d_y, rows, cols, nullptr);
    check_cuda(cudaGetLastError(), "timed launch");
    check_cuda(cudaEventRecord(stop), "record stop");
    check_cuda(cudaEventSynchronize(stop), "wait stop");

    float elapsed_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "elapsed time");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return elapsed_ms / static_cast<float>(iterations);
}

struct Shape
{
    int rows;
    int cols;
    const char *note;
};

const Shape kShapes[] = {
    {8192, 128, "short rows"},
    {4096, 1024, "medium rows"},
    {1024, 8192, "long rows"},
    {8192, 32, "cols < warp"},
    // A row of 128 KB cannot stay in a 128 KB L1 alongside anything else, so the
    // large-block trick has to break down somewhere. This shape is here to find
    // out whether it already has.
    {256, 32768, "row exceeds L1"},
};

// ---------------------------------------------------------------------------
// Claim B needs a kernel whose resource usage matches the real one. Rather than
// exposing v1's internals, reproduce its shape here: two shared-memory float
// arrays of BlockSize/32 entries each, and a comparable register footprint.
// Occupancy depends on block size, shared memory per block, and registers per
// thread, all of which this mirrors.
// ---------------------------------------------------------------------------

template <int BlockSize>
__global__ void occupancy_probe_kernel(const float *__restrict__ x, float *__restrict__ y, int cols)
{
    constexpr int kWarpsPerBlock = BlockSize / 32;
    __shared__ float warp_max[kWarpsPerBlock];
    __shared__ float warp_sum[kWarpsPerBlock];

    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;

    float accumulator = 0.0f;
    for (int col = threadIdx.x; col < cols; col += BlockSize) accumulator += x[col];

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) accumulator += __shfl_xor_sync(0xffffffffu, accumulator, offset);

    if (lane == 0)
    {
        warp_max[warp_id] = accumulator;
        warp_sum[warp_id] = accumulator;
    }
    __syncthreads();

    if (threadIdx.x == 0) y[0] = warp_max[0] + warp_sum[0];
}

template <int BlockSize>
void report_occupancy(int max_threads_per_sm)
{
    int max_active_blocks = 0;
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks, occupancy_probe_kernel<BlockSize>, BlockSize, 0),
               "query max active blocks");

    cudaFuncAttributes attributes{};
    check_cuda(cudaFuncGetAttributes(&attributes, occupancy_probe_kernel<BlockSize>), "query func attributes");

    const int resident_threads = max_active_blocks * BlockSize;
    const double occupancy_percent = 100.0 * resident_threads / static_cast<double>(max_threads_per_sm);

    std::printf("  %10d %14d %16d %14.1f%% %12d %14zu\n", BlockSize, max_active_blocks, resident_threads / 32, occupancy_percent, attributes.numRegs,
                attributes.sharedSizeBytes);
}

}  // namespace

int main()
{
    constexpr int warmup = 5;
    constexpr int iterations = 30;

    float *d_x = nullptr;
    float *d_y = nullptr;

    try
    {
        cudaDeviceProp properties{};
        check_cuda(cudaGetDeviceProperties(&properties, 0), "get device properties");

        std::printf("device: %s\n", properties.name);
        std::printf("SMs: %d, max threads/SM: %d, max blocks/SM: %d\n", properties.multiProcessorCount, properties.maxThreadsPerMultiProcessor,
                    properties.maxBlocksPerMultiProcessor);
        std::printf("shared memory/SM: %zu KB, L2: %d KB\n\n", properties.sharedMemPerMultiprocessor / 1024, properties.l2CacheSize / 1024);

        // -------------------------------------------------------------------
        // Claim A: how much of the extra read traffic actually reaches DRAM
        // -------------------------------------------------------------------
        std::printf("== claim A: does the extra read traffic reach DRAM? ==\n");
        std::printf("probe_1read moves 2N bytes, probe_3read moves 4N bytes, both without exp().\n");
        std::printf("ratio near 2.0 means the extra reads went to DRAM; near 1.0 means they hit in L1.\n\n");

        std::printf("  %-16s %10s %10s %7s %10s %10s %7s %7s %11s\n", "shape", "1read ms", "3read ms", "ratio", "copy ms", "v1 ms", "v1/1r", "v2/1r",
                    "row bytes");

        for (const Shape &shape : kShapes)
        {
            const size_t element_count = static_cast<size_t>(shape.rows) * shape.cols;
            const size_t byte_count = element_count * sizeof(float);

            check_cuda(cudaMalloc(&d_x, byte_count), "cudaMalloc(x)");
            check_cuda(cudaMalloc(&d_y, byte_count), "cudaMalloc(y)");
            check_cuda(cudaMemset(d_x, 0x3f, byte_count), "init x");

            const float copy_ms = time_kernel(cuda_op_lab::stable_softmax::launch_copy_baseline, d_x, d_y, shape.rows, shape.cols, warmup, iterations);
            const float probe1_ms = time_kernel(cuda_op_lab::stable_softmax::launch_probe_1read, d_x, d_y, shape.rows, shape.cols, warmup, iterations);
            const float probe3_ms = time_kernel(cuda_op_lab::stable_softmax::launch_probe_3read, d_x, d_y, shape.rows, shape.cols, warmup, iterations);
            const float v1_ms =
                time_kernel(cuda_op_lab::stable_softmax::launch_softmax_warp_reduce, d_x, d_y, shape.rows, shape.cols, warmup, iterations);
            const float v2_ms =
                time_kernel(cuda_op_lab::stable_softmax::launch_softmax_subwarp_register, d_x, d_y, shape.rows, shape.cols, warmup, iterations);

            char label[32];
            std::snprintf(label, sizeof(label), "%dx%d", shape.rows, shape.cols);

            std::printf("  %-16s %10.4f %10.4f %7.2f %10.4f %10.4f %7.2f %7.2f %9zu B\n", label, probe1_ms, probe3_ms, probe3_ms / probe1_ms, copy_ms,
                        v1_ms, v1_ms / probe1_ms, v2_ms / probe1_ms, shape.cols * sizeof(float));

            cudaFree(d_x);
            cudaFree(d_y);
            d_x = nullptr;
            d_y = nullptr;
        }

        std::printf("\n  v2/1r near 1.00 means v2 reached the speed of a single-pass kernel while still\n");
        std::printf("  making three passes, i.e. the win came from the critical path, not from traffic.\n");

        // -------------------------------------------------------------------
        // Claim A3: how much DRAM traffic is left once a large block is used
        // -------------------------------------------------------------------
        std::printf("\n== claim A3: is there traffic left for a two-pass algorithm? ==\n");
        std::printf("Same probes at 1024 threads, the block size the adaptive launcher picks for long\n");
        std::printf("rows. A ratio near 1.00 means the extra passes already hit L1, so a two-pass\n");
        std::printf("algorithm can only remove instructions, not bytes.\n\n");

        std::printf("  %-16s %12s %12s %7s %14s %11s\n", "shape", "1read@1024", "3read@1024", "ratio", "ratio@256", "row bytes");

        for (const Shape &shape : kShapes)
        {
            // Below 1024 columns most of a 1024-thread block sits idle, which
            // measures idle lanes rather than traffic.
            if (shape.cols < 1024) continue;

            const size_t byte_count = static_cast<size_t>(shape.rows) * shape.cols * sizeof(float);

            check_cuda(cudaMalloc(&d_x, byte_count), "cudaMalloc(x)");
            check_cuda(cudaMalloc(&d_y, byte_count), "cudaMalloc(y)");
            check_cuda(cudaMemset(d_x, 0x3f, byte_count), "init x");

            const float big1_ms =
                time_kernel(cuda_op_lab::stable_softmax::launch_probe_1read_big_block, d_x, d_y, shape.rows, shape.cols, warmup, iterations);
            const float big3_ms =
                time_kernel(cuda_op_lab::stable_softmax::launch_probe_3read_big_block, d_x, d_y, shape.rows, shape.cols, warmup, iterations);
            const float small1_ms =
                time_kernel(cuda_op_lab::stable_softmax::launch_probe_1read, d_x, d_y, shape.rows, shape.cols, warmup, iterations);
            const float small3_ms =
                time_kernel(cuda_op_lab::stable_softmax::launch_probe_3read, d_x, d_y, shape.rows, shape.cols, warmup, iterations);

            char label[32];
            std::snprintf(label, sizeof(label), "%dx%d", shape.rows, shape.cols);

            std::printf("  %-16s %12.4f %12.4f %7.2f %14.2f %9zu B\n", label, big1_ms, big3_ms, big3_ms / big1_ms, small3_ms / small1_ms,
                        shape.cols * sizeof(float));

            cudaFree(d_x);
            cudaFree(d_y);
            d_x = nullptr;
            d_y = nullptr;
        }

        // -------------------------------------------------------------------
        // Claim B: resident blocks per SM as a function of block size
        // -------------------------------------------------------------------
        std::printf("\n== claim B: how many blocks fit per SM? ==\n");
        std::printf("A stalled barrier can only be hidden by other resident blocks, since every\n");
        std::printf("warp inside one block waits at the same barrier.\n\n");

        std::printf("  %10s %14s %16s %15s %12s %14s\n", "block size", "blocks/SM", "warps resident", "occupancy", "registers", "shared bytes");

        report_occupancy<128>(properties.maxThreadsPerMultiProcessor);
        report_occupancy<256>(properties.maxThreadsPerMultiProcessor);
        report_occupancy<512>(properties.maxThreadsPerMultiProcessor);
        report_occupancy<1024>(properties.maxThreadsPerMultiProcessor);

        return 0;
    }
    catch (const std::exception &error)
    {
        std::fprintf(stderr, "error: %s\n", error.what());

        if (d_x) cudaFree(d_x);
        if (d_y) cudaFree(d_y);

        return 1;
    }
}
