// achievable-bandwidth ceiling.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "rms_norm.h"

namespace cuda_op_lab::rms_norm {
namespace {

constexpr int kBlockSize = 256;
constexpr int kVecWidth = 8;  // bf16: 16 bytes

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

    // treat the buffer as a 1dim array
    if (count % kVecWidth == 0)
    {
        const size_t vec_count = count / kVecWidth;
        const size_t grid = (vec_count + kBlockSize - 1) / kBlockSize;
        copy_vec_kernel<<<dim3(static_cast<unsigned>(grid)), dim3(kBlockSize), 0, stream>>>(reinterpret_cast<const float4 *>(x), reinterpret_cast<float4 *>(y), vec_count);
        return;
    }

    const size_t grid = (count + kBlockSize - 1) / kBlockSize;
    copy_scalar_kernel<<<dim3(static_cast<unsigned>(grid)), dim3(kBlockSize), 0, stream>>>(x, y, count);
}

}  // namespace cuda_op_lab::rms_norm
