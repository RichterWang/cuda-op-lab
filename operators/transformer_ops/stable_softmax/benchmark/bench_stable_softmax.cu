// Benchmark + correctness harness for row-wise stable softmax.
//
// Softmax is memory bound, not compute bound, so the headline metric here is
// effective bandwidth (GB/s), not GFLOPS. The theoretical minimum traffic is
// one read plus one write per element: rows * cols * 4 * 2 bytes.
//
// The ratio that matters is %copy, not %peak. Theoretical peak comes from clock
// times bus width and is not reachable in practice; a plain y = x copy at the
// same shape is, so it is measured per shape and used as the real ceiling.
// %peak is still printed to show how much of the gap is the hardware's own.
//
// Correctness is checked against a double-precision CPU reference under three
// input patterns, including two that make a non-stable implementation produce
// inf / nan. Beyond abs/rel error we also check the softmax-specific invariant
// that every row sums to 1.
//
// Exit code is non-zero if any configuration exceeds the accuracy tolerance,
// which lets ctest use this binary directly as a regression test.

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "stable_softmax.h"

namespace {

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

void check_cuda(cudaError_t status, const char *where)
{
    if (status != cudaSuccess) throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(status));
}

using LaunchFn = void (*)(const float *, float *, int, int, cudaStream_t);

struct KernelEntry
{
    const char *name;
    LaunchFn fn;
};

// Adding a new optimization version means adding one line here.
//
// Note when reading the output: both v2 entries only own the narrow-row regime.
// Above cols=256 they forward to launch_softmax_adaptive, so their numbers there
// are identical to v1_adaptive by construction, not by coincidence. v2b further
// requires cols == LanesPerRow * 4 and falls back to v2a otherwise, so it only
// differs from v2a at cols in {32, 64, 128}.
const KernelEntry kKernels[] = {
    {"v0_naive", cuda_op_lab::stable_softmax::launch_softmax_naive},
    {"v1_warp_reduce", cuda_op_lab::stable_softmax::launch_softmax_warp_reduce},
    {"v1_block1024", cuda_op_lab::stable_softmax::launch_softmax_warp_reduce_big_block},
    {"v1_adaptive", cuda_op_lab::stable_softmax::launch_softmax_adaptive},
    {"v2a_subwarp", cuda_op_lab::stable_softmax::launch_softmax_subwarp},
    {"v2b_register", cuda_op_lab::stable_softmax::launch_softmax_subwarp_register},
    {"v3_online", cuda_op_lab::stable_softmax::launch_softmax_online},
    {"v3b_online_ilp", cuda_op_lab::stable_softmax::launch_softmax_online_ilp},
};

struct Shape
{
    int rows;
    int cols;
    const char *note;
};

// The first three cover the three row-length regimes, because the right
// reduction strategy depends on how the row maps onto warps and blocks.
// The last two are deliberately awkward sizes: they catch the float4 tail and
// warp-mapping bugs that vectorized versions tend to introduce.
const Shape kShapes[] = {
    {8192, 128, "short rows"},
    {4096, 1024, "medium rows"},
    {1024, 8192, "long rows"},
    {4096, 1023, "cols % 4 != 0"},
    {8192, 32, "cols < warp"},
    // A 128 KB row cannot stay in a 128 KB L1, so this is the one shape where
    // the extra pass of a three-pass kernel actually reaches DRAM. The traffic
    // probes measure 3read/1read = 1.83 here versus 1.04 at cols=8192, so it is
    // also the only shape where v3 has real traffic to remove.
    {256, 32768, "row exceeds L1"},
};

enum class Pattern
{
    normal,          // U(-1, 1), the ordinary accuracy case
    large_shift,     // row biased by +1e4: naive exp() overflows to inf
    extreme_negative // row biased by -1e4: naive exp() underflows to 0, denominator dies
};

const char *pattern_name(Pattern pattern)
{
    switch (pattern)
    {
        case Pattern::normal: return "normal";
        case Pattern::large_shift: return "large_shift";
        case Pattern::extreme_negative: return "extreme_negative";
    }
    return "unknown";
}

void fill_input(std::vector<float> &host_x, int rows, int cols, Pattern pattern, unsigned seed)
{
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

    float bias = 0.0f;
    if (pattern == Pattern::large_shift) bias = 1.0e4f;
    if (pattern == Pattern::extreme_negative) bias = -1.0e4f;

    for (int row = 0; row < rows; ++row)
    {
        float *x_row = host_x.data() + static_cast<size_t>(row) * cols;
        for (int col = 0; col < cols; ++col) x_row[col] = distribution(generator) + bias;
    }
}

// Reference implementation in double precision, same max-subtraction trick.
void softmax_cpu_double(const float *x, float *y, int rows, int cols)
{
    for (int row = 0; row < rows; ++row)
    {
        const float *x_row = x + static_cast<size_t>(row) * cols;
        float *y_row = y + static_cast<size_t>(row) * cols;

        double row_max = -std::numeric_limits<double>::infinity();
        for (int col = 0; col < cols; ++col) row_max = std::max(row_max, static_cast<double>(x_row[col]));

        double row_sum = 0.0;
        for (int col = 0; col < cols; ++col) row_sum += std::exp(static_cast<double>(x_row[col]) - row_max);

        for (int col = 0; col < cols; ++col)
            y_row[col] = static_cast<float>(std::exp(static_cast<double>(x_row[col]) - row_max) / row_sum);
    }
}

struct ErrorStat
{
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    float max_row_sum_error = 0.0f;
    bool has_non_finite = false;
};

ErrorStat check_result(const std::vector<float> &reference, const std::vector<float> &actual, int rows, int cols)
{
    ErrorStat stat;

    for (int row = 0; row < rows; ++row)
    {
        const float *ref_row = reference.data() + static_cast<size_t>(row) * cols;
        const float *got_row = actual.data() + static_cast<size_t>(row) * cols;

        double row_sum = 0.0;
        for (int col = 0; col < cols; ++col)
        {
            const float got = got_row[col];
            if (!std::isfinite(got)) stat.has_non_finite = true;

            const float abs_error = std::fabs(ref_row[col] - got);
            // Outputs live in (0, 1); guard the denominator so tiny reference
            // values do not inflate the relative error into noise.
            const float denominator = std::max(1.0e-6f, std::fabs(ref_row[col]));

            stat.max_abs = std::max(stat.max_abs, abs_error);
            stat.max_rel = std::max(stat.max_rel, abs_error / denominator);

            row_sum += static_cast<double>(got);
        }

        // Conservation check: a correct softmax row always sums to 1.
        stat.max_row_sum_error = std::max(stat.max_row_sum_error, static_cast<float>(std::fabs(row_sum - 1.0)));
    }

    return stat;
}

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

// Theoretical peak from the memory clock and bus width, used as the roofline
// substitute since cuBLAS offers no softmax to compare against.
double theoretical_peak_gb_per_s()
{
    // cudaDeviceProp::memoryClockRate was removed in newer CUDA versions, so
    // query the attributes directly instead.
    int memory_clock_khz = 0;
    int memory_bus_width_bits = 0;

    check_cuda(cudaDeviceGetAttribute(&memory_clock_khz, cudaDevAttrMemoryClockRate, 0), "query memory clock rate");
    check_cuda(cudaDeviceGetAttribute(&memory_bus_width_bits, cudaDevAttrGlobalMemoryBusWidth, 0), "query memory bus width");

    // Clock is in kHz, bus width in bits, DDR transfers twice per clock.
    return 2.0 * static_cast<double>(memory_clock_khz) * (memory_bus_width_bits / 8.0) / 1.0e6;
}

}  // namespace

int main()
{
    constexpr int warmup = 5;
    constexpr int iterations = 30;
    constexpr float abs_error_tolerance = 1.0e-6f;
    constexpr float row_sum_tolerance = 1.0e-4f;

    float *d_x = nullptr;
    float *d_y = nullptr;

    try
    {
        cudaDeviceProp properties{};
        check_cuda(cudaGetDeviceProperties(&properties, 0), "get device properties");

        const double peak_gb_per_s = theoretical_peak_gb_per_s();

        std::printf("device: %s\n", properties.name);
        std::printf("theoretical peak bandwidth: %.2f GB/s\n\n", peak_gb_per_s);

        bool all_passed = true;

        for (const Shape &shape : kShapes)
        {
            const size_t element_count = static_cast<size_t>(shape.rows) * shape.cols;
            const size_t byte_count = element_count * sizeof(float);

            std::vector<float> h_x(element_count);
            std::vector<float> h_reference(element_count);
            std::vector<float> h_actual(element_count);

            check_cuda(cudaMalloc(&d_x, byte_count), "cudaMalloc(x)");
            check_cuda(cudaMalloc(&d_y, byte_count), "cudaMalloc(y)");

            const Pattern patterns[] = {Pattern::normal, Pattern::large_shift, Pattern::extreme_negative};

            for (Pattern pattern : patterns)
            {
                fill_input(h_x, shape.rows, shape.cols, pattern, 42);
                softmax_cpu_double(h_x.data(), h_reference.data(), shape.rows, shape.cols);

                check_cuda(cudaMemcpy(d_x, h_x.data(), byte_count, cudaMemcpyHostToDevice), "copy x");

                std::printf("rows=%d cols=%d (%s)  pattern=%s\n", shape.rows, shape.cols, shape.note, pattern_name(pattern));
                std::printf("  %-14s %10s %10s %8s %8s %14s %14s %14s\n", "kernel", "ms", "GB/s", "%peak", "%copy", "max_abs_err", "max_rel_err",
                            "row_sum_err");

                // Roofline for this shape. Not accuracy-checked: it is a copy,
                // not a softmax, so it has no reference to compare against.
                const float copy_ms =
                    time_kernel(cuda_op_lab::stable_softmax::launch_copy_baseline, d_x, d_y, shape.rows, shape.cols, warmup, iterations);
                const double copy_gb_per_s = 2.0 * static_cast<double>(byte_count) / (static_cast<double>(copy_ms) * 1.0e6);

                std::printf("  %-14s %10.4f %10.2f %7.1f%% %7s %14s %14s %14s\n", "copy (ceiling)", copy_ms, copy_gb_per_s,
                            100.0 * copy_gb_per_s / peak_gb_per_s, "-", "-", "-", "-");

                for (const KernelEntry &entry : kKernels)
                {
                    check_cuda(cudaMemset(d_y, 0, byte_count), "clear y");

                    entry.fn(d_x, d_y, shape.rows, shape.cols, nullptr);
                    check_cuda(cudaGetLastError(), "kernel launch");
                    check_cuda(cudaDeviceSynchronize(), "kernel synchronize");

                    check_cuda(cudaMemcpy(h_actual.data(), d_y, byte_count, cudaMemcpyDeviceToHost), "copy y");

                    const ErrorStat stat = check_result(h_reference, h_actual, shape.rows, shape.cols);

                    const float average_ms = time_kernel(entry.fn, d_x, d_y, shape.rows, shape.cols, warmup, iterations);

                    // Minimum traffic: one read + one write per element.
                    const double moved_bytes = 2.0 * static_cast<double>(byte_count);
                    const double effective_gb_per_s = moved_bytes / (static_cast<double>(average_ms) * 1.0e6);
                    const double peak_percent = 100.0 * effective_gb_per_s / peak_gb_per_s;
                    const double copy_percent = 100.0 * effective_gb_per_s / copy_gb_per_s;

                    std::printf("  %-14s %10.4f %10.2f %7.1f%% %7.1f%% %14.6e %14.6e %14.6e%s\n", entry.name, average_ms, effective_gb_per_s,
                                peak_percent, copy_percent, stat.max_abs, stat.max_rel, stat.max_row_sum_error,
                                stat.has_non_finite ? "  [NON-FINITE]" : "");

                    const bool passed = !stat.has_non_finite && stat.max_abs < abs_error_tolerance && stat.max_row_sum_error < row_sum_tolerance;
                    if (!passed)
                    {
                        all_passed = false;
                        std::printf("  -> FAIL: %s on rows=%d cols=%d pattern=%s\n", entry.name, shape.rows, shape.cols, pattern_name(pattern));
                    }
                }

                std::printf("\n");
            }

            cudaFree(d_x);
            cudaFree(d_y);
            d_x = nullptr;
            d_y = nullptr;
        }

        std::printf("%s\n", all_passed ? "all checks passed" : "some checks failed");
        return all_passed ? 0 : 1;
    }
    catch (const std::exception &error)
    {
        std::fprintf(stderr, "error: %s\n", error.what());

        if (d_x) cudaFree(d_x);
        if (d_y) cudaFree(d_y);

        return 1;
    }
}
