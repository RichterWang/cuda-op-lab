#include <cuda_runtime.h>

#include "../sgemm.h"

#define BLOCK_SIZE 16
#define TILE_SIZE 64
#define TILE_M 4
#define TILE_N 4

// optimize info：
//  add vector load from global to shared mem
//  add register access between shared mem and caculate
//  expand process element per thread
__global__ void sgemm_vector_register_kernel(const float* A, const float* B, float* C, int M, int N, int K)
{
    // define base of elements in C
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    // the element index of the result
    int base_row = blockIdx.y * TILE_SIZE + ty * TILE_M;
    int base_col = blockIdx.x * TILE_SIZE + tx * TILE_N;

    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    // register for result to acclumlate
    float sum[TILE_M][TILE_N] = {0.0f};

    // out loop describe the increase of K
    for(int tileId = 0; tileId < (K + TILE_SIZE - 1) / TILE_SIZE; tileId++)
    {
        // write in tileA
        // basic index for each thread
        int a_row_base = base_row;
        int a_col_base = tileId * TILE_SIZE + tx * TILE_N;

        // caculate for each thread
        #pragma unroll
        for(int i = 0; i < TILE_M; i++)
        {
            // define the global row
            int row = a_row_base + i;
            int col = a_col_base;

            // each memory or register access needs edge check
            if(row < M && col < K - 3)
            {
                float4 temp = *reinterpret_cast<const float4*>(&A[row * K + col]);
                tileA[ty * TILE_M + i][tx * TILE_N + 0] = temp.x;
                tileA[ty * TILE_M + i][tx * TILE_N + 1] = temp.y;
                tileA[ty * TILE_M + i][tx * TILE_N + 2] = temp.z;
                tileA[ty * TILE_M + i][tx * TILE_N + 3] = temp.w;
            }
            else
            {
                tileA[ty * TILE_M + i][tx * TILE_N + 0] = 0.0f;
                tileA[ty * TILE_M + i][tx * TILE_N + 1] = 0.0f;
                tileA[ty * TILE_M + i][tx * TILE_N + 2] = 0.0f;
                tileA[ty * TILE_M + i][tx * TILE_N + 3] = 0.0f;   
            }
        }

        // write in tileB
        // basic index for each thread
        int b_row_base = tileId * TILE_SIZE + ty * TILE_M;
        int b_col_base = base_col;

        #pragma unroll
        for(int i = 0; i < TILE_M; i++)
        {
            int row = b_row_base + i;
            int col = b_col_base;


            if(row < K && col < N - 3)
            {
                float4 temp = *reinterpret_cast<const float4*>(&B[row * N + col]);
                tileB[ty * TILE_M + i][tx * TILE_N + 0] = temp.x;
                tileB[ty * TILE_M + i][tx * TILE_N + 1] = temp.y;
                tileB[ty * TILE_M + i][tx * TILE_N + 2] = temp.z;
                tileB[ty * TILE_M + i][tx * TILE_N + 3] = temp.w;
            }
            else
            {
                tileB[ty * TILE_M + i][tx * TILE_N + 0] = 0.0f;
                tileB[ty * TILE_M + i][tx * TILE_N + 1] = 0.0f;
                tileB[ty * TILE_M + i][tx * TILE_N + 2] = 0.0f;
                tileB[ty * TILE_M + i][tx * TILE_N + 3] = 0.0f;   
            }
        }

        __syncthreads();

        #pragma unroll
        for(int k = 0; k < TILE_SIZE; k++)
        {
            float tileA_reg[4];
            float tileB_reg[4];

            #pragma unroll
            for(int i = 0; i < TILE_M; i++) tileA_reg[i] = tileA[TILE_M * ty + i][k];
            
            #pragma unroll
            for(int i = 0; i < TILE_N; i++) tileB_reg[i] = tileB[k][TILE_N* tx + i];

            #pragma unroll
            for(int i = 0; i < TILE_M; i++)
            {
                #pragma unroll
                for(int j = 0; j < TILE_N; j++)
                {
                    sum[i][j] += tileA_reg[i] * tileB_reg[j];
                }
            }
        }

        __syncthreads();

        // reference group: whitout register
        // #pragma unroll
        // for(int k = 0; k < TILE_SIZE; k++)
        // {
        //     for(int i = 0; i < 4; i++)
        //     {
        //         for(int j = 0; j , 4; j++)
        //         {
        //             sum[i][j] += tileA[ty * 4 + i][k] * tileB[k][tx * 4 + j];
        //         }
        //     }
        // }
    }

    // write the result back to C
    #pragma unroll
    for(int i = 0; i < TILE_M; i++)
    {
        #pragma unroll
        for(int j = 0; j < TILE_N; j++)
        {
            int row = base_row + i;
            int col = base_col + j;

            if(row < M && col < N) C[row * N + col] = sum[i][j];
        }
    }
}

namespace cuda_op_lab::sgemm {
// TODO(v3): shared-memory tiling + register tiling.
void launch_sgemm_vec_reg(const float* A, const float* B, float* C, int m, int n, int k, cudaStream_t stream) 
{
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 gridSize((n + TILE_SIZE - 1) / TILE_SIZE, (m + TILE_SIZE - 1) / TILE_SIZE, 1);

    sgemm_vector_register_kernel<<<gridSize, blockSize, 0, stream>>>(A, B, C, m, n, k);
}
}