// Bandwidth roofline reference: y = x, nothing else.
//
// This is not a softmax. It exists because 448 GB/s is the *theoretical* peak
// derived from clock times bus width, and real achievable bandwidth is usually
// only 80-90% of that. Without this measurement we cannot tell whether a
// softmax kernel at 348 GB/s has 100 GB/s left on the table or is already
// within a few percent of the hardware limit.
//
// It moves exactly the traffic that an ideal softmax would move (one read plus
// one write per element), so its GB/s number is directly comparable and forms
// the real ceiling for every kernel in the table.
//
// float4 is used so the copy itself is not instruction-issue limited; the tail
// is handled with scalar accesses so odd `cols` still works.

#include <cuda_runtime.h>

#include "stable_softmax.h"

namespace cuda_op_lab::stable_softmax {
namespace {

constexpr int kBlockSize = 256;

__global__ void copy_kernel(const float4 *__restrict__ x4, float4 *__restrict__ y4, size_t vector_count)
{
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < vector_count; index += stride) y4[index] = x4[index];
}

__global__ void copy_tail_kernel(const float *__restrict__ x, float *__restrict__ y, size_t begin, size_t end)
{
    const size_t index = begin + blockIdx.x * blockDim.x + threadIdx.x;
    if (index < end) y[index] = x[index];
}

}  // namespace

void launch_copy_baseline(const float *x, float *y, int rows, int cols, cudaStream_t stream)
{
    if (rows <= 0 || cols <= 0) return;

    const size_t element_count = static_cast<size_t>(rows) * cols;
    const size_t vector_count = element_count / 4;
    const size_t tail_begin = vector_count * 4;

    if (vector_count > 0)
    {
        // Cap the grid so each thread handles several float4s. Persistent-style
        // looping keeps launch overhead out of the measurement.
        const size_t needed_blocks = (vector_count + kBlockSize - 1) / kBlockSize;
        const unsigned blocks = static_cast<unsigned>(needed_blocks < 4096 ? needed_blocks : 4096);

        copy_kernel<<<blocks, kBlockSize, 0, stream>>>(reinterpret_cast<const float4 *>(x), reinterpret_cast<float4 *>(y), vector_count);
    }

    if (tail_begin < element_count)
    {
        const unsigned tail_elements = static_cast<unsigned>(element_count - tail_begin);
        copy_tail_kernel<<<1, tail_elements, 0, stream>>>(x, y, tail_begin, element_count);
    }
}

}  // namespace cuda_op_lab::stable_softmax
