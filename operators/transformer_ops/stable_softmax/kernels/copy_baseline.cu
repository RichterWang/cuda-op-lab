// get the bandwidth roofline reference for a softmax kernel by copying the input to the output
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
