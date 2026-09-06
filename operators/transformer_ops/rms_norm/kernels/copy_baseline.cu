// Not an RMSNorm: y = x, used as the achievable-bandwidth ceiling.
//
// RMSNorm is memory bound, so the useful question is not "how many GB/s" but
// "how close to the most this hardware can move". There is no cuBLAS RMSNorm to
// compare against, so the reference gets built here instead: a kernel that moves
// exactly the traffic an ideal RMSNorm would, one read and one write per element,
// and does nothing else. Nothing that computes a normalization can beat it.
//
// It is measured per shape because the ceiling itself moves with shape. Small
// arrays fit in L2 and report above the DRAM peak; large ones settle at what DRAM
// actually delivers.
//
// gamma is deliberately excluded. All rows read the same cols-element vector, so
// after the first blocks it is cache-resident and generates almost no DRAM
// traffic. Including it in the byte count would inflate the reported bandwidth
// with traffic that never happened.
//
// The copy is 16-byte vectorized so the ceiling reflects what the memory system
// can do at full access width, matching how v1 loads. A scalar copy would report
// a lower ceiling and make the kernels look better than they are.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "rms_norm.h"

namespace cuda_op_lab::rms_norm {
namespace {

constexpr int kBlockSize = 256;
constexpr int kVecWidth = 8;  // bf16 elements per 16 bytes

__global__ void copy_vec_kernel(const float4 *__restrict__ x, float4 *__restrict__ y, size_t vec_count)
{
    const size_t index = static_cast<size_t>(blockIdx.x) * kBlockSize + threadIdx.x;
    if (index < vec_count) y[index] = x[index];
}

__global__ void copy_scalar_kernel(const __nv_bfloat16 *__restrict__ x, __nv_bfloat16 *__restrict__ y, size_t count)
{
    const size_t index = static_cast<size_t>(blockIdx.x) * kBlockSize + threadIdx.x;
    if (index < count) y[index] = x[index];
}

}  // namespace

void launch_copy_baseline(const __nv_bfloat16 *x, __nv_bfloat16 *y, int rows, int cols, cudaStream_t stream)
{
    if (rows <= 0 || cols <= 0) return;

    const size_t count = static_cast<size_t>(rows) * cols;

    // The copy walks the whole buffer as one flat array, so only the total length
    // has to be a multiple of the vector width, not each row.
    if (count % kVecWidth == 0)
    {
        const size_t vec_count = count / kVecWidth;
        const size_t grid = (vec_count + kBlockSize - 1) / kBlockSize;
        copy_vec_kernel<<<dim3(static_cast<unsigned>(grid)), dim3(kBlockSize), 0, stream>>>(reinterpret_cast<const float4 *>(x),
                                                                                           reinterpret_cast<float4 *>(y), vec_count);
        return;
    }

    const size_t grid = (count + kBlockSize - 1) / kBlockSize;
    copy_scalar_kernel<<<dim3(static_cast<unsigned>(grid)), dim3(kBlockSize), 0, stream>>>(x, y, count);
}

}  // namespace cuda_op_lab::rms_norm
