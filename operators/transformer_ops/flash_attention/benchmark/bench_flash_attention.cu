// Flash attention benchmark: TFLOPS and HBM traffic against the unfused baseline,
// plus a two-level accuracy check.
//
// Why the primary metric is not bandwidth. softmax and rms_norm are memory bound,
// so "percent of the copy ceiling" was the only number that mattered there.
// Attention has O(N^2 * d) arithmetic against O(N * d) of input, so the compute
// intensity grows with N and the operator crosses from memory bound to compute
// bound somewhere in the middle of the shapes below. Reporting one number would
// hide that crossing, so three are reported per kernel:
//
//   TFLOPS       useful arithmetic per second, with the causal triangle counted
//                correctly. This is the number that matters once N is large.
//   HBM GB       bytes that must cross the memory bus, derived from the algorithm
//                rather than measured. For v0 this includes the score matrix; for
//                the fused kernels it does not. The whole thesis of the operator
//                is visible in this column alone.
//   speedup      against v0 at the same shape, which is what the optimization is
//                actually worth.
//
// Accuracy uses the same two-level scheme as rms_norm -- a loose relative figure
// against a double reference for context, and a tight ULP figure against the
// bf16-rounded reference as the pass/fail gate -- but the ULP figure has to be
// normalized differently here, and getting that wrong is easy.
//
// rms_norm's output is x * g / rms: every element is the same order of magnitude as
// its input, so measuring each element's error in ULP *of that element* is the
// right question. Attention's output is a convex combination of zero-mean V rows,
// so it cancels. An output element can land near zero while every term that
// produced it was order 1, and the absolute precision achievable for that element
// is set by the magnitude of the terms, not by the magnitude of the result. Divide
// the error by that near-zero element's own ULP and the ratio blows up on a kernel
// that is doing nothing wrong.
//
// Measured, not assumed: the first version of this benchmark normalized per
// element and reported 7 ULP for the *unfused baseline* at head_dim=128 while the
// fused kernel reported 1. Two implementations sharing no arithmetic do not
// disagree by 7x on a real bug in only one of them; the metric was reading
// cancellation. Normalizing by the row's largest element -- the scale the row's
// arithmetic actually operated at -- puts both at the same place.
//
// So: error is measured in ULP at the row scale. The tolerance stays modest,
// because after that correction the online softmax's seq_len-long rescale chain is
// the only remaining source of drift, and a wrong mask, a missing rescale, or a
// dropped tile all move the answer far more than a few ULP.
//
// The exit code is non-zero when any shape fails, so this doubles as the
// regression test under ctest.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "flash_attention.h"

namespace {

using cuda_op_lab::flash_attention::AttentionShape;
using cuda_op_lab::flash_attention::attention_flops;
using cuda_op_lab::flash_attention::default_scale;
using cuda_op_lab::flash_attention::FusedRowInfo;
using cuda_op_lab::flash_attention::fused_row_supported;
using cuda_op_lab::flash_attention::launch_attention_fused_row;
using cuda_op_lab::flash_attention::launch_attention_unfused;
using cuda_op_lab::flash_attention::query_fused_row;
using cuda_op_lab::flash_attention::unfused_workspace_bytes;

constexpr int kWarmupIters = 3;
constexpr int kTimedIters = 10;

// Accuracy is checked on a small shape only. The host reference is O(N^2 * d) in
// double precision and single threaded, so running it at N=4096 would dominate the
// benchmark's runtime without testing anything the small shape does not. The
// rescale chain is exercised by the same code path at any N.
constexpr int kAccuracyBatch = 1;
constexpr int kAccuracyHeads = 2;
constexpr int kAccuracySeq = 256;

// Row-scale ULP. 2.0 rather than rms_norm's 1.0 because the online rescale chain
// is seq_len steps long and each step multiplies the accumulator by an inexact
// exp(), so a small amount of drift is expected where rms_norm's single-pass
// reduction had none.
constexpr float kUlpTolerance = 2.0f;

// Above this the fp32 score matrix stops being a reasonable allocation. At
// batch*heads=8 and N=4096 the workspace is 8 * 4096^2 * 4 bytes = 512 MB, which
// on an 8 GB card competes with everything else. Shapes past the limit run the
// fused kernels only and report v0 as skipped, rather than either crashing or
// silently shrinking the shape.
constexpr size_t kMaxWorkspaceBytes = 1ull << 30;  // 1 GB

#define CUDA_CHECK(expr)                                                                                      \
    do                                                                                                        \
    {                                                                                                         \
        const cudaError_t status = (expr);                                                                     \
        if (status != cudaSuccess)                                                                             \
        {                                                                                                     \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(status), __FILE__, __LINE__);  \
            std::exit(EXIT_FAILURE);                                                                          \
        }                                                                                                     \
    } while (0)

struct Case
{
    AttentionShape shape;
    bool causal;
    const char* note;
};

// Real inference shapes rather than a sweep. Each entry answers a question:
//   N=512    short prompt; the fixed per-block cost is still visible here.
//   N=2048   llama-2 context. The main shape.
//   N=4096   llama-2 max context. v0's workspace is largest here, so the traffic
//            argument is most visible.
//   d=128    llama-7b head_dim; halves the tile row count, so it separates
//            "head_dim matters" from "tile count matters".
//   causal   the mask a decoder actually uses. The fused kernels skip masked
//            tiles, v0 computes them, so this is where the two diverge most.
const std::vector<Case> kCases = {
    {{4, 8, 512, 64}, false, "short prompt, dense"},
    {{4, 8, 512, 64}, true, "short prompt, causal"},
    {{2, 8, 2048, 64}, false, "llama-2 context, dense"},
    {{2, 8, 2048, 64}, true, "llama-2 context, causal"},
    {{1, 8, 2048, 128}, true, "llama-7b head_dim, causal"},
    {{1, 8, 4096, 64}, true, "llama-2 max context, causal"},
};

void fill_normal(std::vector<__nv_bfloat16>& host, unsigned seed)
{
    std::mt19937 engine(seed);
    // Unit scale. Scores are then O(sqrt(d)) before the 1/sqrt(d) factor, so the
    // softmax sees inputs of order 1, which is the regime a trained model runs in.
    // A large-scale variant would only test the max subtraction, and that is
    // already covered in stable_softmax.
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto& value : host) value = __float2bfloat16(dist(engine));
}

float bf16_ulp(float value)
{
    if (value == 0.0f || !std::isfinite(value)) return std::ldexp(1.0f, -126 - 7);

    int exponent = 0;
    std::frexp(std::abs(value), &exponent);
    return std::ldexp(1.0f, exponent - 8);
}

// Double-precision attention on the host. Straightforward and slow: the point is
// to share no code and no arithmetic order with any device kernel, so agreement
// means something.
void reference_attention(const std::vector<__nv_bfloat16>& q, const std::vector<__nv_bfloat16>& k, const std::vector<__nv_bfloat16>& v,
                         std::vector<double>& out, const AttentionShape& shape, bool causal)
{
    const int n = shape.seq_len;
    const int d = shape.head_dim;
    const double scale = static_cast<double>(default_scale(d));

    std::vector<double> scores(static_cast<size_t>(n));

    for (int bh = 0; bh < shape.batch * shape.heads; ++bh)
    {
        const size_t slab = static_cast<size_t>(bh) * n * d;

        for (int query = 0; query < n; ++query)
        {
            const int limit = causal ? query + 1 : n;

            double row_max = -1.0e300;
            for (int key = 0; key < limit; ++key)
            {
                double dot = 0.0;
                for (int dim = 0; dim < d; ++dim)
                    dot += static_cast<double>(__bfloat162float(q[slab + static_cast<size_t>(query) * d + dim])) *
                           static_cast<double>(__bfloat162float(k[slab + static_cast<size_t>(key) * d + dim]));
                scores[key] = dot * scale;
                row_max = std::max(row_max, scores[key]);
            }

            double sum = 0.0;
            for (int key = 0; key < limit; ++key)
            {
                scores[key] = std::exp(scores[key] - row_max);
                sum += scores[key];
            }

            for (int dim = 0; dim < d; ++dim)
            {
                double accumulator = 0.0;
                for (int key = 0; key < limit; ++key)
                    accumulator += scores[key] * static_cast<double>(__bfloat162float(v[slab + static_cast<size_t>(key) * d + dim]));
                out[slab + static_cast<size_t>(query) * d + dim] = accumulator / sum;
            }
        }
    }
}

struct Accuracy
{
    float vs_double;
    float ulp;
};

// vs_double is the largest per-element relative deviation from the double
// reference. It is reported for context only: bf16 storage caps it at roughly
// 2^-8 = 3.9e-3 regardless of kernel quality, so it cannot separate a correct
// kernel from a slightly wrong one.
//
// ulp is the gate. Error is normalized by one bf16 ULP at the row's largest
// element rather than at each element's own value, for the cancellation reason in
// the file header: the row is what the arithmetic operated on, so the row's scale
// is what bounds the achievable absolute error for every element in it.
Accuracy compare(const std::vector<__nv_bfloat16>& device_out, const std::vector<double>& reference, const AttentionShape& shape)
{
    Accuracy result{0.0f, 0.0f};

    const int rows = shape.batch * shape.heads * shape.seq_len;
    const int d = shape.head_dim;

    for (int row = 0; row < rows; ++row)
    {
        const size_t base = static_cast<size_t>(row) * d;

        double row_scale = 0.0;
        for (int dim = 0; dim < d; ++dim) row_scale = std::max(row_scale, std::abs(reference[base + dim]));

        const float scale_ulp = bf16_ulp(static_cast<float>(row_scale));

        for (int dim = 0; dim < d; ++dim)
        {
            const float got = __bfloat162float(device_out[base + dim]);
            const double want = reference[base + dim];

            const double denominator = std::max(std::abs(want), 1.0e-6);
            result.vs_double = std::max(result.vs_double, static_cast<float>(std::abs(got - want) / denominator));

            const float rounded = __bfloat162float(__float2bfloat16(static_cast<float>(want)));
            result.ulp = std::max(result.ulp, std::abs(got - rounded) / scale_ulp);
        }
    }

    return result;
}

// Bytes that have to cross the memory bus, derived from the algorithm rather than
// measured. Reported as an algorithmic property, not a profiler reading: the L2
// absorbs some of the fused kernels' K/V re-reads, so the real DRAM figure is
// lower than the fused number below and higher than its ideal. What the column
// establishes is the asymptotic difference, which is not sensitive to that.
double qkvo_bytes(const AttentionShape& shape)
{
    // Q, K, V read once and O written once, all bf16.
    return 4.0 * shape.batch * shape.heads * shape.seq_len * shape.head_dim * sizeof(__nv_bfloat16);
}

double unfused_score_bytes(const AttentionShape& shape)
{
    // Written by the scores kernel, read and rewritten by the softmax, read by the
    // PV kernel: five passes over an fp32 [N, N] matrix per (batch, head).
    return 5.0 * shape.batch * shape.heads * static_cast<double>(shape.seq_len) * shape.seq_len * sizeof(float);
}

float time_iterations(void (*body)(void*), void* context)
{
    for (int iter = 0; iter < kWarmupIters; ++iter) body(context);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int iter = 0; iter < kTimedIters; ++iter) body(context);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return elapsed_ms / static_cast<float>(kTimedIters);
}

struct Buffers
{
    const __nv_bfloat16* q;
    const __nv_bfloat16* k;
    const __nv_bfloat16* v;
    __nv_bfloat16* o;
    float* workspace;
    AttentionShape shape;
    bool causal;
};

void run_unfused(void* context)
{
    Buffers& buffers = *static_cast<Buffers*>(context);
    launch_attention_unfused(buffers.q, buffers.k, buffers.v, buffers.o, buffers.workspace, buffers.shape, buffers.causal, nullptr);
}

void run_fused_row(void* context)
{
    Buffers& buffers = *static_cast<Buffers*>(context);
    launch_attention_fused_row(buffers.q, buffers.k, buffers.v, buffers.o, buffers.shape, buffers.causal, nullptr);
}

}  // namespace

int main()
{
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));

    cudaDeviceProp props{};
    CUDA_CHECK(cudaGetDeviceProperties(&props, device));

    std::printf("device: %s  (sm_%d%d)\n", props.name, props.major, props.minor);
    std::printf("dtype: bf16 storage, fp32 accumulate\n");
    std::printf("HBM GB is the algorithmic traffic, not a profiler reading: v0 includes the fp32 score matrix, fused kernels do not.\n");
    std::printf("accuracy: ulp is the pass/fail number, measured at the row scale (not per element -- attention output cancels), tolerance %.1f\n\n",
                kUlpTolerance);

    std::printf("v1 configuration (local_bytes must be 0, or the output accumulator spilled):\n");
    std::printf("  %10s %10s %8s %14s %8s %8s %12s\n", "head_dim", "block", "tile_n", "shared/block", "regs", "local", "blocks/SM");
    for (int head_dim : {64, 128})
    {
        const FusedRowInfo info = query_fused_row(head_dim);
        if (!info.supported)
        {
            std::printf("  %10d   unsupported\n", head_dim);
            continue;
        }
        std::printf("  %10d %10d %8d %12dB %8d %7dB %12d\n", head_dim, info.block_size, info.tile_n, info.shared_bytes_per_block,
                    info.registers_per_thread, info.local_bytes_per_thread, info.max_blocks_per_sm);
    }
    std::printf("\n");

    bool all_passed = true;

    // ---------------------------------------------------------------------
    // Accuracy, on one small shape, both mask settings.
    // ---------------------------------------------------------------------
    {
        std::printf("accuracy check (batch=%d heads=%d seq=%d):\n", kAccuracyBatch, kAccuracyHeads, kAccuracySeq);
        std::printf("  %-16s %8s %8s %14s %10s\n", "kernel", "head_dim", "causal", "vs_double", "ulp");

        for (int head_dim : {64, 128})
        {
            for (bool causal : {false, true})
            {
                const AttentionShape shape{kAccuracyBatch, kAccuracyHeads, kAccuracySeq, head_dim};
                const size_t count = static_cast<size_t>(shape.batch) * shape.heads * shape.seq_len * shape.head_dim;
                const size_t bytes = count * sizeof(__nv_bfloat16);

                std::vector<__nv_bfloat16> host_q(count);
                std::vector<__nv_bfloat16> host_k(count);
                std::vector<__nv_bfloat16> host_v(count);
                std::vector<__nv_bfloat16> host_o(count);
                std::vector<double> reference(count);

                fill_normal(host_q, 11u);
                fill_normal(host_k, 22u);
                fill_normal(host_v, 33u);

                reference_attention(host_q, host_k, host_v, reference, shape, causal);

                __nv_bfloat16* device_q = nullptr;
                __nv_bfloat16* device_k = nullptr;
                __nv_bfloat16* device_v = nullptr;
                __nv_bfloat16* device_o = nullptr;
                float* workspace = nullptr;
                CUDA_CHECK(cudaMalloc(&device_q, bytes));
                CUDA_CHECK(cudaMalloc(&device_k, bytes));
                CUDA_CHECK(cudaMalloc(&device_v, bytes));
                CUDA_CHECK(cudaMalloc(&device_o, bytes));
                CUDA_CHECK(cudaMalloc(&workspace, unfused_workspace_bytes(shape)));

                CUDA_CHECK(cudaMemcpy(device_q, host_q.data(), bytes, cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(device_k, host_k.data(), bytes, cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(device_v, host_v.data(), bytes, cudaMemcpyHostToDevice));

                struct Entry
                {
                    const char* name;
                    bool fused;
                };
                const Entry entries[] = {{"v0_unfused", false}, {"v1_fused_row", true}};

                for (const Entry& entry : entries)
                {
                    if (entry.fused && !fused_row_supported(head_dim)) continue;

                    CUDA_CHECK(cudaMemset(device_o, 0, bytes));
                    if (entry.fused)
                        launch_attention_fused_row(device_q, device_k, device_v, device_o, shape, causal, nullptr);
                    else
                        launch_attention_unfused(device_q, device_k, device_v, device_o, workspace, shape, causal, nullptr);
                    CUDA_CHECK(cudaDeviceSynchronize());
                    CUDA_CHECK(cudaMemcpy(host_o.data(), device_o, bytes, cudaMemcpyDeviceToHost));

                    const Accuracy accuracy = compare(host_o, reference, shape);
                    std::printf("  %-16s %8d %8s %14.6e %10.2f", entry.name, head_dim, causal ? "yes" : "no", accuracy.vs_double, accuracy.ulp);
                    if (accuracy.ulp > kUlpTolerance)
                    {
                        std::printf("   FAIL");
                        all_passed = false;
                    }
                    std::printf("\n");
                }

                CUDA_CHECK(cudaFree(device_q));
                CUDA_CHECK(cudaFree(device_k));
                CUDA_CHECK(cudaFree(device_v));
                CUDA_CHECK(cudaFree(device_o));
                CUDA_CHECK(cudaFree(workspace));
            }
        }
        std::printf("\n");
    }

    // ---------------------------------------------------------------------
    // Performance.
    // ---------------------------------------------------------------------
    for (const Case& item : kCases)
    {
        const AttentionShape& shape = item.shape;
        const size_t count = static_cast<size_t>(shape.batch) * shape.heads * shape.seq_len * shape.head_dim;
        const size_t bytes = count * sizeof(__nv_bfloat16);

        std::vector<__nv_bfloat16> host_input(count);

        __nv_bfloat16* device_q = nullptr;
        __nv_bfloat16* device_k = nullptr;
        __nv_bfloat16* device_v = nullptr;
        __nv_bfloat16* device_o = nullptr;
        CUDA_CHECK(cudaMalloc(&device_q, bytes));
        CUDA_CHECK(cudaMalloc(&device_k, bytes));
        CUDA_CHECK(cudaMalloc(&device_v, bytes));
        CUDA_CHECK(cudaMalloc(&device_o, bytes));

        fill_normal(host_input, 11u);
        CUDA_CHECK(cudaMemcpy(device_q, host_input.data(), bytes, cudaMemcpyHostToDevice));
        fill_normal(host_input, 22u);
        CUDA_CHECK(cudaMemcpy(device_k, host_input.data(), bytes, cudaMemcpyHostToDevice));
        fill_normal(host_input, 33u);
        CUDA_CHECK(cudaMemcpy(device_v, host_input.data(), bytes, cudaMemcpyHostToDevice));

        const size_t workspace_bytes = unfused_workspace_bytes(shape);
        const bool run_v0 = workspace_bytes <= kMaxWorkspaceBytes;

        float* workspace = nullptr;
        if (run_v0) CUDA_CHECK(cudaMalloc(&workspace, workspace_bytes));

        Buffers buffers{device_q, device_k, device_v, device_o, workspace, shape, item.causal};

        const double flops = attention_flops(shape, item.causal);

        std::printf("batch=%d heads=%d seq=%d head_dim=%d causal=%s (%s)\n", shape.batch, shape.heads, shape.seq_len, shape.head_dim,
                    item.causal ? "yes" : "no", item.note);
        std::printf("  %-16s %10s %10s %10s %10s\n", "kernel", "ms", "TFLOPS", "HBM GB", "speedup");

        float baseline_ms = 0.0f;

        if (run_v0)
        {
            baseline_ms = time_iterations(run_unfused, &buffers);
            const double gb = (qkvo_bytes(shape) + unfused_score_bytes(shape)) / 1.0e9;
            std::printf("  %-16s %10.3f %10.2f %10.3f %10s\n", "v0_unfused", baseline_ms, flops / (baseline_ms / 1.0e3) / 1.0e12, gb, "1.00x");
        }
        else
        {
            std::printf("  %-16s %10s %10s %10.3f %10s   (workspace %.1f GB exceeds the %.1f GB cap)\n", "v0_unfused", "-", "-",
                        (qkvo_bytes(shape) + unfused_score_bytes(shape)) / 1.0e9, "-", workspace_bytes / 1.0e9, kMaxWorkspaceBytes / 1.0e9);
        }

        if (fused_row_supported(shape.head_dim))
        {
            const float ms = time_iterations(run_fused_row, &buffers);
            // K and V are re-read once per query row, which is the term v2 exists
            // to remove. Counting it here rather than the ideal N*d makes the
            // number comparable to v0's and makes v2's improvement visible in the
            // same column.
            const double kv_reread = 2.0 * shape.batch * shape.heads * static_cast<double>(shape.seq_len) * shape.seq_len * shape.head_dim *
                                     sizeof(__nv_bfloat16) * (item.causal ? 0.5 : 1.0);
            const double gb = (qkvo_bytes(shape) + kv_reread) / 1.0e9;

            std::printf("  %-16s %10.3f %10.2f %10.3f", "v1_fused_row", ms, flops / (ms / 1.0e3) / 1.0e12, gb);
            if (run_v0)
                std::printf(" %9.2fx\n", baseline_ms / ms);
            else
                std::printf(" %10s\n", "-");
        }

        std::printf("\n");

        CUDA_CHECK(cudaFree(device_q));
        CUDA_CHECK(cudaFree(device_k));
        CUDA_CHECK(cudaFree(device_v));
        CUDA_CHECK(cudaFree(device_o));
        if (run_v0) CUDA_CHECK(cudaFree(workspace));
    }

    if (!all_passed)
    {
        std::printf("FAILED: at least one kernel exceeded %.1f bf16 ULP at the row scale.\n", kUlpTolerance);
        return EXIT_FAILURE;
    }

    std::printf("all kernels within %.1f bf16 ULP at the row scale.\n", kUlpTolerance);
    return EXIT_SUCCESS;
}
