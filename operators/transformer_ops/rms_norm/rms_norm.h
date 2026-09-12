#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace cuda_op_lab::rms_norm {

// Row-wise RMS normalization, the form used by llama-style transformers.
//
//     rms(x) = sqrt( mean(x_i^2) + eps )
//     y_i    = x_i / rms(x) * gamma_i
//
// x, y:   device pointers to [rows, cols] row-major bf16 buffers
// gamma:  device pointer to [cols] bf16 weights, shared by every row
// eps:    added to the mean, not to the sqrt result. The two are not equivalent
//         and the wrong one costs an order of magnitude of accuracy when the
//         row is near zero.
//
// Storage is bf16 because that is what an inference activation actually is, and
// because this operator is memory bound: halving the bytes per element halves
// the traffic and is worth more than anything done to the arithmetic.
//
// All accumulation and normalization happen in fp32 regardless. A bf16 sum of
// squares is not an option: with cols=4096 the sum reaches ~4096 for unit-scale
// inputs, and bf16 cannot represent integers above 256 exactly, so the reduction
// would lose most of its significance long before it overflowed. bf16 is only
// ever the load and store format here.
//
// Why bf16 rather than fp16: bf16 keeps all 8 exponent bits of fp32, so the sum
// of squares cannot overflow the storage format and the conversion to fp32 is
// exact. fp16 tops out at 65504, which a sum of squares reaches easily.
//
// Accuracy expectation: bf16 has 7 mantissa bits, so machine epsilon is 2^-8 or
// about 3.9e-3. Output error against a double reference lands in the 1e-3 range
// by construction. That is the format, not the kernel.

// v0: one block per row, shared-memory tree reduction, two passes over the row.
// Two is the floor for a non-resident implementation: one pass to accumulate the
// sum of squares, one to normalize. Unlike softmax there is no second reduction
// to fuse away, so the pass count is not an optimization axis here.
void launch_rmsnorm_naive(const __nv_bfloat16* x, const __nv_bfloat16* gamma, __nv_bfloat16* y, int rows, int cols, float eps,
                          cudaStream_t stream = nullptr);

// v1: warp shuffle reduction and 128-bit vectorized access, together.
//
// These are combined rather than staged because they are independent and both
// are known wins from the softmax work: __shfl_xor_sync removes the shared
// memory round trip and all but one barrier, and a 16-byte access is what the
// memory system wants regardless. Splitting them would produce two versions that
// differ by a result already established elsewhere.
//
// A 16-byte access holds 8 bf16 elements, not the 4 that a float4 of fp32 holds.
// That doubles the per-thread element count and will matter later: hidden=4096
// over 1024 threads gives each thread only 4 elements, so full-width access and
// a 1024-thread block cannot both be had at that shape.
//
// Rows whose length is not a multiple of 8, or whose base pointer is not
// 16-byte aligned, fall back to scalar access.
void launch_rmsnorm_warp_vec(const __nv_bfloat16* x, const __nv_bfloat16* gamma, __nv_bfloat16* y, int rows, int cols, float eps,
                             cudaStream_t stream = nullptr);

// Convenience entry point for the [batch, seq, hidden] tensor an attention block
// actually hands over. Normalization runs along the last dimension, and a
// row-major tensor is already contiguous along it, so flattening is a pure
// reinterpretation of the shape: rows = batch * seq, cols = hidden. No data
// moves.
//
// This assumes the last dimension is contiguous. A strided view (a slice along
// hidden, for instance) would be read incorrectly with no error reported.
void launch_rmsnorm_3d(const __nv_bfloat16* x, const __nv_bfloat16* gamma, __nv_bfloat16* y, int batch, int seq, int hidden, float eps,
                       cudaStream_t stream = nullptr);

// v2: the row stays in registers between the two passes, so pass 2 loads no x.
//
// The traffic argument for this is not the one it looks like. Removing pass 2's
// read does not halve DRAM traffic, because v1's pass 2 already mostly hits L1 --
// that is why v1 can sit at 99.9% of a ceiling defined as one read plus one write.
// What v2 removes is the fraction of pass 2 that misses L1, which the occupancy
// experiment showed is confined to long rows: shrinking L1 hurts cols=8192 and
// 11008 and leaves cols=4096 and 5120 untouched.
//
// So the expectation is deliberately narrow. Long rows should gain, short rows
// should not, and if short rows gain anyway then the explanation above is wrong.
//
// Restrictions come from the mechanism, not from convenience. The per-thread
// cache is a local array, and a local array only stays in registers when every
// index into it is a compile-time constant, so the vectors-per-thread count is a
// template parameter and only a fixed set of instantiations exists. Rows longer
// than that set covers, and rows not divisible by 8, fall back to v1. A generic
// version would spill the cache to local memory and be slower than v1 while
// still looking like an optimization.
void launch_rmsnorm_resident(const __nv_bfloat16* x, const __nv_bfloat16* gamma, __nv_bfloat16* y, int rows, int cols, float eps,
                             cudaStream_t stream = nullptr);

// What v2 does with a given cols, and what the hardware grants for it.
//
// local_bytes_per_thread is the one to check first. It should be zero. Anything
// above zero means the register cache spilled to local memory, and a spilled v2
// is strictly worse than v1: it pays the same read twice and adds address
// arithmetic. registers_per_thread should exceed v1's 42 by roughly
// expected_cache_registers; a much larger gap means the compiler kept extra live
// values, a much smaller one means it did not keep the cache at all.
struct ResidentInfo
{
    bool supported;  // false means this cols falls back to v1
    int block_size;
    int vec_per_thread;
    int expected_cache_registers;  // 4 per cached vector
    int registers_per_thread;
    int local_bytes_per_thread;  // must be 0, or the cache spilled
    int max_blocks_per_sm;
    int resident_threads_per_sm;
};

ResidentInfo query_rmsnorm_resident(int cols);

// Not a normalization: plain y = x copy in bf16, used as the achievable
// bandwidth ceiling. It moves exactly the traffic an ideal RMSNorm would, one
// read and one write per element, so its GB/s is the real target for the kernels
// above. Measured per shape, since the ceiling itself moves with shape.
//
// gamma is deliberately not part of this: every row reads the same cols-element
// vector, so after the first block it lives in cache and contributes almost no
// DRAM traffic. Counting it would inflate the reported bandwidth without any
// traffic behind it.
void launch_copy_baseline(const __nv_bfloat16* x, __nv_bfloat16* y, int rows, int cols, cudaStream_t stream = nullptr);

// --------------------------------------------------------------------------
// Occupancy probe. Diagnostic only, not part of the operator.
// --------------------------------------------------------------------------
//
// v1 sits at 99.9% of the copy ceiling for cols=4096 and 5120, but 87.7% at 8192
// and 83.8% at 11008. The probe exposes block_size and a dynamic shared memory
// request as independent knobs so the candidate causes can be varied one at a
// time. The shared memory is never read: it is reserved per block out of a fixed
// per-SM budget, so requesting a large amount lowers blocks per SM while leaving
// block size, loop structure and instruction mix untouched.
//
// What it found, on sm_86 with this kernel at 42 registers per thread:
//
//   Occupancy is not the cause. The occupancy API grants 2 blocks per SM at 512
//   threads and 1 block at 1024, so both configurations sit at 66.7%, yet 512 is
//   77% faster at cols=4096. The 1536-thread per-SM limit never binds here; the
//   register budget does, and it caps 512-thread blocks at 2 rather than the 3
//   that 1536/512 would suggest.
//
//   Tail waste is not the cause either. cols=8192 at 1024 threads has vec_cols
//   exactly 1024, so every thread loads one vector and there is no tail, and it
//   still measures 91.0%. Meanwhile cols=5120 at 512 threads leaves three
//   quarters of the block idle in the final iteration and reaches 99.6%.
//
//   Forcing occupancy down directly confirms the first point: at fixed block size
//   512, halving blocks per SM from 2 to 1 moves cols=8192 from 96.9% to 95.9%.
//   A bandwidth-bound kernel does not need occupancy to hide latency.
//
//   The variable that does track performance is L1 capacity. At fixed occupancy,
//   growing the shared memory request from 0 to 40 KB drops cols=8192 from 96.9%
//   to 89.4% and cols=11008 from 90.7% to 80.8%, while cols=4096 and 5120 do not
//   move at all. Shared memory and L1 share one 100 KB per-SM allocation, so
//   taking 40 KB for shared leaves 40 KB less for L1. The shapes that degrade are
//   exactly the ones whose rows are too large to keep pass 2 resident once L1
//   shrinks; the small ones fit regardless.
//
// So the missing bandwidth at 8192 and 11008 is pass 2 re-reads that L1 does not
// fully absorb, which is a different cause from either candidate above.
void launch_rmsnorm_probe(const __nv_bfloat16* x, const __nv_bfloat16* gamma, __nv_bfloat16* y, int rows, int cols, float eps, int block_size,
                          size_t dynamic_shared_bytes, cudaStream_t stream = nullptr);


// hardware based information struct (from probe function)
struct ProbeInfo
{
    int registers_per_thread;
    int static_shared_bytes;
    int max_blocks_per_sm;
    int resident_threads_per_sm;  // max_blocks_per_sm * block_size
};

ProbeInfo query_rmsnorm_probe(int block_size, size_t dynamic_shared_bytes);

}  // namespace cuda_op_lab::rms_norm
