// RMSNorm benchmark: bandwidth against the copy ceiling, plus a two-level
// accuracy check.
//
// Independent of the softmax benchmark on purpose. The shapes are real llama
// hidden sizes rather than a power-of-two sweep, the traffic accounting is bf16
// rather than fp32, and the accuracy check has to answer a different question, so
// sharing code would have meant a shared harness that fits neither well.
//
// Why accuracy needs two levels here. bf16 carries 7 mantissa bits, so its
// machine epsilon is 2^-8 = 3.9e-3. A correct kernel writing a bf16 output will
// differ from a double-precision reference by roughly that much, purely from the
// final rounding. A single tolerance therefore has to be set near 4e-3, and a
// threshold that loose will not catch a kernel that is genuinely wrong by a
// percent or two. So two numbers get reported:
//
//   vs_double  the raw error against a double reference. Expected around 4e-3.
//              This measures the format, and is printed for context, not judged.
//
//   ulp        the error against that same reference after rounding it to bf16,
//              expressed in units in the last place. A kernel doing the right
//              arithmetic lands within a couple of ULP, because the only
//              remaining differences are fp32 rounding in the reduction and the
//              order in which partial sums were combined. This is the number the
//              pass/fail decision uses, and it stays tight regardless of how
//              coarse bf16 is.
//
// The exit code is non-zero when any shape fails, so this doubles as the
// regression test under ctest.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "rms_norm.h"

namespace {

using cuda_op_lab::rms_norm::launch_copy_baseline;
using cuda_op_lab::rms_norm::launch_rmsnorm_naive;
using cuda_op_lab::rms_norm::launch_rmsnorm_resident;
using cuda_op_lab::rms_norm::launch_rmsnorm_warp_vec;
using cuda_op_lab::rms_norm::query_rmsnorm_resident;
using cuda_op_lab::rms_norm::ResidentInfo;

constexpr int kWarmupIters = 20;
constexpr int kTimedIters = 100;

// bf16 stores 7 mantissa bits plus an implicit leading 1, so it carries 8
// significand bits and its machine epsilon is 2^-8. Used only to report the
// expected scale of vs_double, not to size a ULP.
constexpr float kBF16Epsilon = 1.0f / 256.0f;

// Correctly rounding a real number to bf16 cannot be off by more than half a
// ULP. A kernel doing the right arithmetic exceeds that only by the fp32
// rounding in its reduction and by summation order, which moves the result by at
// most one more step. One ULP is therefore the real bound, and anything above it
// points at the arithmetic rather than the format.
//
// The earlier version of this check estimated ULP size as magnitude * epsilon,
// which is only correct when the significand is exactly 1.0. For a value near
// the top of its binade that underestimates the step by up to 2x and inflates
// the reported error by the same factor, which is why every kernel appeared to
// sit just under a 2.0 threshold. bf16_ulp below computes the step from the
// actual exponent instead.
constexpr float kUlpTolerance = 1.0f;

// Size of one bf16 ULP at the magnitude of value.
//
// frexp writes value as significand * 2^exponent with the significand in
// [0.5, 1), so |value| lies in [2^(exponent-1), 2^exponent). Across that binade
// the spacing is fixed at 2^(exponent-1) * 2^-7, which is 2^(exponent-8).
float bf16_ulp(float value)
{
    if (value == 0.0f || !std::isfinite(value))
    {
        // Smallest normal bf16 is 2^-126; use its ULP as the floor so a zero
        // reference does not divide by zero.
        return std::ldexp(1.0f, -126 - 7);
    }

    int exponent = 0;
    std::frexp(std::abs(value), &exponent);
    return std::ldexp(1.0f, exponent - 8);
}

#define CUDA_CHECK(expr)                                                                             \
    do                                                                                               \
    {                                                                                                \
        const cudaError_t status = (expr);                                                           \
        if (status != cudaSuccess)                                                                    \
        {                                                                                            \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(status), __FILE__, __LINE__); \
            std::exit(EXIT_FAILURE);                                                                 \
        }                                                                                            \
    } while (0)

struct Shape
{
    int rows;
    int cols;
    const char* note;
};

// Real llama hidden sizes, not a power-of-two sweep. Each entry is here for a
// reason:
//   4096   llama-7b hidden. The shape that matters most.
//   5120   llama-13b hidden. Not a power of two, still a multiple of 8.
//   8192   llama-70b hidden. Large enough that a row no longer fits in L1.
//   11008  llama-7b FFN intermediate. Divisible by 8 but by nothing convenient.
//   4095   deliberately breaks 16-byte row alignment, so the scalar fallback in
//          v1 gets exercised instead of silently never running.
const std::vector<Shape> kShapes = {
    {4096, 4096, "llama-7b hidden"},
    {4096, 5120, "llama-13b hidden"},
    {2048, 8192, "llama-70b hidden"},
    {2048, 11008, "llama-7b FFN intermediate"},
    {4096, 4095, "unaligned, exercises scalar fallback"},
};

enum class Pattern
{
    kUnitScale,   // N(0,1), the normal case
    kLargeScale,  // N(0,1) * 1e3, sum of squares reaches ~1e10
    kTinyScale,   // N(0,1) * 1e-4, mean near zero so eps decides the result
};

const char* pattern_name(Pattern pattern)
{
    switch (pattern)
    {
        case Pattern::kUnitScale:
            return "unit_scale";
        case Pattern::kLargeScale:
            return "large_scale";
        default:
            return "tiny_scale";
    }
}

// Fill in fp32 first, then round to bf16, so the host reference starts from the
// exact same bits the device will read. Generating bf16 directly and then
// widening would be equivalent, but this makes the rounding step explicit.
void fill_input(std::vector<__nv_bfloat16>& host, int rows, int cols, Pattern pattern, unsigned seed)
{
    std::mt19937 engine(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    float scale = 1.0f;
    if (pattern == Pattern::kLargeScale) scale = 1.0e3f;
    if (pattern == Pattern::kTinyScale) scale = 1.0e-4f;

    for (size_t index = 0; index < static_cast<size_t>(rows) * cols; ++index) host[index] = __float2bfloat16(dist(engine) * scale);
}

void fill_gamma(std::vector<__nv_bfloat16>& host, int cols, unsigned seed)
{
    std::mt19937 engine(seed);
    // Centered on 1 because that is how a trained RMSNorm weight actually looks.
    // A zero-centered gamma would let sign cancellation hide errors.
    std::normal_distribution<float> dist(1.0f, 0.1f);

    for (int col = 0; col < cols; ++col) host[col] = __float2bfloat16(dist(engine));
}

// Double-precision reference. Accumulating in double keeps the reference free of
// the fp32 rounding the kernel has, which is what makes the ULP comparison
// meaningful: any difference beyond a couple of ULP came from the kernel.
void reference_rmsnorm(const std::vector<__nv_bfloat16>& x, const std::vector<__nv_bfloat16>& gamma, std::vector<double>& out, int rows, int cols,
                       float eps)
{
    for (int row = 0; row < rows; ++row)
    {
        const size_t base = static_cast<size_t>(row) * cols;

        double sum_squares = 0.0;
        for (int col = 0; col < cols; ++col)
        {
            const double value = static_cast<double>(__bfloat162float(x[base + col]));
            sum_squares += value * value;
        }

        // eps goes on the mean, matching the kernels. Putting it on the sqrt
        // result instead would change the answer at tiny_scale by orders of
        // magnitude, which is exactly why that pattern is in the list.
        const double scale = 1.0 / std::sqrt(sum_squares / static_cast<double>(cols) + static_cast<double>(eps));

        for (int col = 0; col < cols; ++col)
        {
            const double value = static_cast<double>(__bfloat162float(x[base + col]));
            const double gain = static_cast<double>(__bfloat162float(gamma[col]));
            out[base + col] = value * scale * gain;
        }
    }
}

struct Accuracy
{
    float vs_double;  // raw relative error against the double reference
    float ulp;        // error against the bf16-rounded reference, in bf16 ULP
};

Accuracy compare(const std::vector<__nv_bfloat16>& device_out, const std::vector<double>& reference, size_t count)
{
    Accuracy result{0.0f, 0.0f};

    for (size_t index = 0; index < count; ++index)
    {
        const float got = __bfloat162float(device_out[index]);
        const double want = reference[index];

        // Level 1: how far the bf16 output is from the true value. Dominated by
        // the output format, so this is reported rather than judged.
        const double denominator = std::max(std::abs(want), 1.0e-6);
        const float raw_error = static_cast<float>(std::abs(got - want) / denominator);
        result.vs_double = std::max(result.vs_double, raw_error);

        // Level 2: how far the output is from the best bf16 can do. Round the
        // reference the same way the kernel had to, then measure the gap in units
        // of the actual bf16 step at that magnitude. This isolates kernel error
        // from format error, so the threshold stays meaningful no matter how
        // coarse bf16 is.
        const float rounded_reference = __bfloat162float(__float2bfloat16(static_cast<float>(want)));
        const float ulp_error = std::abs(got - rounded_reference) / bf16_ulp(rounded_reference);
        result.ulp = std::max(result.ulp, ulp_error);
    }

    return result;
}

using LaunchFn = void (*)(const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, int, int, float, cudaStream_t);

float time_kernel(LaunchFn launch, const __nv_bfloat16* x, const __nv_bfloat16* gamma, __nv_bfloat16* y, int rows, int cols, float eps)
{
    for (int iter = 0; iter < kWarmupIters; ++iter) launch(x, gamma, y, rows, cols, eps, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int iter = 0; iter < kTimedIters; ++iter) launch(x, gamma, y, rows, cols, eps, nullptr);
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

// One read plus one write per element, at 2 bytes each. gamma is excluded: every
// row reads the same vector, so it is cache-resident after the first blocks and
// counting it would report traffic that never reached DRAM.
double effective_gb(int rows, int cols)
{
    const double bytes = static_cast<double>(rows) * cols * sizeof(__nv_bfloat16) * 2.0;
    return bytes / 1.0e9;
}

double theoretical_peak_gb_per_s()
{
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));

    int memory_clock_khz = 0;
    int bus_width_bits = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&memory_clock_khz, cudaDevAttrMemoryClockRate, device));
    CUDA_CHECK(cudaDeviceGetAttribute(&bus_width_bits, cudaDevAttrGlobalMemoryBusWidth, device));

    // Factor of 2 for the double data rate.
    return 2.0 * static_cast<double>(memory_clock_khz) * 1.0e3 * (static_cast<double>(bus_width_bits) / 8.0) / 1.0e9;
}

struct Candidate
{
    const char* name;
    LaunchFn launch;
};

}  // namespace

int main()
{
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));

    cudaDeviceProp props{};
    CUDA_CHECK(cudaGetDeviceProperties(&props, device));

    const double peak_gb_per_s = theoretical_peak_gb_per_s();

    std::printf("device: %s  (sm_%d%d)\n", props.name, props.major, props.minor);
    std::printf("theoretical peak bandwidth: %.1f GB/s\n", peak_gb_per_s);
    std::printf("dtype: bf16 storage, fp32 accumulate\n");
    std::printf("accuracy: vs_double is expected near 4e-3 (bf16 epsilon is %.2e); ulp is the pass/fail number, tolerance %.1f\n\n", kBF16Epsilon,
                kUlpTolerance);

    const std::vector<Candidate> candidates = {
        {"v0_naive", launch_rmsnorm_naive},
        {"v1_warp_vec", launch_rmsnorm_warp_vec},
        {"v2_resident", launch_rmsnorm_resident},
    };

    // What v2 actually compiled to, before any timing. local_bytes must be 0: a
    // non-zero value means the register cache spilled to local memory, in which
    // case v2 is doing strictly more work than v1 and any speedup reported below
    // would have some other cause.
    std::printf("v2 configuration per shape (local_bytes must be 0, or the cache spilled):\n");
    std::printf("  %8s %10s %8s %10s %8s %12s %12s\n", "cols", "block", "vec/thr", "regs", "local", "regs vs v1", "occupancy");
    for (const Shape& shape : kShapes)
    {
        const ResidentInfo info = query_rmsnorm_resident(shape.cols);
        if (!info.supported)
        {
            std::printf("  %8d %10s (falls back to v1)\n", shape.cols, "-");
            continue;
        }
        std::printf("  %8d %10d %8d %8d %7dB %10d/%-2d %11.1f%%\n", shape.cols, info.block_size, info.vec_per_thread, info.registers_per_thread,
                    info.local_bytes_per_thread, info.registers_per_thread - 42, info.expected_cache_registers,
                    100.0 * info.resident_threads_per_sm / props.maxThreadsPerMultiProcessor);
    }
    std::printf("\n");

    const std::vector<Pattern> patterns = {Pattern::kUnitScale, Pattern::kLargeScale, Pattern::kTinyScale};

    // The value llama uses. Small enough not to perturb a normal row, large
    // enough to keep a near-zero row finite.
    const float eps = 1.0e-6f;

    bool all_passed = true;

    for (const Shape& shape : kShapes)
    {
        const size_t count = static_cast<size_t>(shape.rows) * shape.cols;
        const size_t bytes = count * sizeof(__nv_bfloat16);

        std::vector<__nv_bfloat16> host_x(count);
        std::vector<__nv_bfloat16> host_gamma(shape.cols);
        std::vector<__nv_bfloat16> host_y(count);
        std::vector<double> reference(count);

        __nv_bfloat16* device_x = nullptr;
        __nv_bfloat16* device_gamma = nullptr;
        __nv_bfloat16* device_y = nullptr;
        CUDA_CHECK(cudaMalloc(&device_x, bytes));
        CUDA_CHECK(cudaMalloc(&device_gamma, static_cast<size_t>(shape.cols) * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&device_y, bytes));

        fill_gamma(host_gamma, shape.cols, 1234u);
        CUDA_CHECK(cudaMemcpy(device_gamma, host_gamma.data(), static_cast<size_t>(shape.cols) * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

        const double gb = effective_gb(shape.rows, shape.cols);

        for (Pattern pattern : patterns)
        {
            fill_input(host_x, shape.rows, shape.cols, pattern, 5678u);
            CUDA_CHECK(cudaMemcpy(device_x, host_x.data(), bytes, cudaMemcpyHostToDevice));

            reference_rmsnorm(host_x, host_gamma, reference, shape.rows, shape.cols, eps);

            std::printf("rows=%d cols=%d (%s)  pattern=%s\n", shape.rows, shape.cols, shape.note, pattern_name(pattern));
            std::printf("  %-16s %8s %10s %8s %8s %14s %10s\n", "kernel", "ms", "GB/s", "%peak", "%copy", "vs_double", "ulp");

            const float copy_ms = time_copy(device_x, device_y, shape.rows, shape.cols);
            const double copy_gb_per_s = gb / (copy_ms / 1.0e3);
            std::printf("  %-16s %8.4f %10.2f %7.1f%% %8s %14s %10s\n", "copy (ceiling)", copy_ms, copy_gb_per_s,
                        100.0 * copy_gb_per_s / peak_gb_per_s, "-", "-", "-");

            for (const Candidate& candidate : candidates)
            {
                CUDA_CHECK(cudaMemset(device_y, 0, bytes));
                candidate.launch(device_x, device_gamma, device_y, shape.rows, shape.cols, eps, nullptr);
                CUDA_CHECK(cudaDeviceSynchronize());
                CUDA_CHECK(cudaMemcpy(host_y.data(), device_y, bytes, cudaMemcpyDeviceToHost));

                const Accuracy accuracy = compare(host_y, reference, count);

                const float ms = time_kernel(candidate.launch, device_x, device_gamma, device_y, shape.rows, shape.cols, eps);
                const double gb_per_s = gb / (ms / 1.0e3);

                std::printf("  %-16s %8.4f %10.2f %7.1f%% %7.1f%% %14.6e %10.2f", candidate.name, ms, gb_per_s, 100.0 * gb_per_s / peak_gb_per_s,
                            100.0 * gb_per_s / copy_gb_per_s, accuracy.vs_double, accuracy.ulp);

                if (accuracy.ulp > kUlpTolerance)
                {
                    std::printf("   FAIL");
                    all_passed = false;
                }
                std::printf("\n");
            }
            std::printf("\n");
        }

        CUDA_CHECK(cudaFree(device_x));
        CUDA_CHECK(cudaFree(device_gamma));
        CUDA_CHECK(cudaFree(device_y));
    }

    if (!all_passed)
    {
        std::printf("FAILED: at least one kernel exceeded %.1f bf16 ULP against the rounded reference.\n", kUlpTolerance);
        return EXIT_FAILURE;
    }

    std::printf("all kernels within %.1f bf16 ULP.\n", kUlpTolerance);
    return EXIT_SUCCESS;
}
