#include <cuda_runtime.h>

#include "../sgemm.h"

#define BLOCK_SIZE 8

// sgemm implement of shared mem
__global__ void sgemm_shared_mem_kernel(const float* A, const float* B, float* C, int M, int N, int K)
{
    // all threads in the same block can see
    __shared__ float tileA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float tileB[BLOCK_SIZE][BLOCK_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // global index(each thread responsible for which element in C)
    int row = blockIdx.y * BLOCK_SIZE + ty; 
    int col = blockIdx.x * BLOCK_SIZE + tx;

    // acclumelaotr 
    float sum = 0.0f;

    // main for loop
    for(int tile_num = 0; tile_num < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; tile_num++)
    {
        // get tileA element
        int a_col = tile_num * BLOCK_SIZE + tx;
        if(row < M && a_col < K) tileA[ty][tx] = A[row * K + a_col];
        else tileA[ty][tx] = 0.0f;
        
        // get tileB element
        int b_row = tile_num * BLOCK_SIZE + ty;
        if(b_row < K && col < N) tileB[ty][tx] = B[b_row * N + col];
        else tileB[ty][tx] = 0.0f;

        __syncthreads();

        for(int k = 0; k < BLOCK_SIZE; k++)
        {
            sum += tileA[ty][k] * tileB[k][tx];
        }

        __syncthreads();
    }

    if(row < M && col < N) C[row * N + col] = sum;
}

namespace cuda_op_lab::sgemm {

void launch_sgemm_shared_mem(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream) {
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    sgemm_shared_mem_kernel<<<gridSize, blockSize, 0, stream>>>(A, B, C, M, N, K);
}

}  // namespace cuda_op_lab::sgemm