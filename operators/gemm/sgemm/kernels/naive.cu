#include <cuda_runtime.h>
#include "../sgemm.h"

#define BLOCK_SIZE 16

// Naive implementation of SGEMM kernel
// 传入矩阵的存储约定是行优先
__global__ void sgemm_naive_kernel(const float* A, const float* B, float* C, int M, int N, int K)
{
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    if (row < M && col < N)
    {
        float sum = 0.0f;
        for(int k = 0; k < K; k++)
        {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

namespace cuda_op_lab{
namespace sgemm{
    void launch_sgemm_naive(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream)
    {
        dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
        dim3 gridSize((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

        sgemm_naive_kernel<<<gridSize, blockSize, 0, stream>>>(A, B, C, M, N, K);
    }
}
}

