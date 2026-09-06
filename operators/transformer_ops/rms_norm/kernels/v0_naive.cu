// v0: baseline RMSNorm. One block per row, shared-memory tree reduction, two
// passes over the row, scalar bf16 access.
//
// This is the reference point every later version is measured against, so it is
// written for clarity rather than speed. The only thing here that is not
// negotiable is where the precision boundary sits.
//
// Two passes is the floor for any version that does not keep the row in
// registers: pass 1 accumulates the sum of squares, pass 2 needs the finished
// scale before it can write anything. Softmax had three passes and a second
// reduction, which is what made fusing passes an optimization axis there. RMSNorm
// has one reduction, so that axis does not exist and the interesting work moves
// to access width and residency instead.
//
// Precision boundary: bf16 is the storage format and nothing more. Every value is
// converted to fp32 the moment it is loaded, all arithmetic including the
// reduction runs in fp32, and the result is converted back only at the store.
// Accumulating in bf16 would be useless, not merely imprecise: bf16 has 7
// mantissa bits and cannot represent integers above 256 exactly, while a
// sum of squares over cols=4096 unit-scale elements lands near 4096. The
// accumulator would stop registering small contributions almost immediately.
//
// eps is added to the mean, not to the sqrt result. rsqrt(mean + eps) keeps the
// result finite when the row is all zeros; 1/(sqrt(mean) + eps) would give a very
// different answer and does not match what the reference implementations do.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "rms_norm.h"

namespace cuda_op_lab::rms_norm {
namespace {

constexpr int kBlockSize = 256;

__global__ void rmsnorm_naive_kernel(const __nv_bfloat16 *__restrict__ x, const __nv_bfloat16 *__restrict__ gamma, __nv_bfloat16 *__restrict__ y,
                                     int rows, int cols, float eps)
{
    const int row = blockIdx.x;
    if (row >= rows) return;

    const __nv_bfloat16 *x_row = x + static_cast<size_t>(row) * cols;
    __nv_bfloat16 *y_row = y + static_cast<size_t>(row) * cols;

    __shared__ float partial[kBlockSize];

    // Pass 1: sum of squares. Grid-stride within the row so any cols works.
    float thread_sum = 0.0f;
    for (int col = threadIdx.x; col < cols; col += kBlockSize)
    {
        const float value = __bfloat162float(x_row[col]);
        thread_sum += value * value;
    }

    partial[threadIdx.x] = thread_sum;
    __syncthreads();

    // Shared-memory tree reduction. Every step halves the active threads and
    // needs a barrier, which is exactly the cost v1 removes with shuffles.
#pragma unroll
    for (int stride = kBlockSize / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }

    // rsqrtf rather than 1.0f / sqrtf: one instruction instead of two, and the
    // accuracy is far beyond what a bf16 output can record.
    const float scale = rsqrtf(partial[0] / static_cast<float>(cols) + eps);

    // Pass 2: normalize and apply the per-column weight. gamma is the same for
    // every row, so after the first few blocks these loads are cache hits.
    for (int col = threadIdx.x; col < cols; col += kBlockSize)
    {
        const float value = __bfloat162float(x_row[col]);
        const float weight = __bfloat162float(gamma[col]);
        y_row[col] = __float2bfloat16(value * scale * weight);
    }
}

}  // namespace

void launch_rmsnorm_naive(const __nv_bfloat16 *x, const __nv_bfloat16 *gamma, __nv_bfloat16 *y, int rows, int cols, float eps, cudaStream_t stream)
{
    if (rows <= 0 || cols <= 0) return;

    rmsnorm_naive_kernel<<<dim3(rows), dim3(kBlockSize), 0, stream>>>(x, gamma, y, rows, cols, eps);
}

void launch_rmsnorm_3d(const __nv_bfloat16 *x, const __nv_bfloat16 *gamma, __nv_bfloat16 *y, int batch, int seq, int hidden, float eps,
                       cudaStream_t stream)
{
    // Flattening [batch, seq, hidden] to [batch * seq, hidden] is free: the
    // tensor is row-major, so it is already contiguous along the normalized
    // dimension. Only the interpretation of the shape changes.
    launch_rmsnorm_warp_vec(x, gamma, y, batch * seq, hidden, eps, stream);
}

}  // namespace cuda_op_lab::rms_norm
