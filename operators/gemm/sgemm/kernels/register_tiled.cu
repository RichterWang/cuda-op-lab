#include <cuda_runtime.h>

#include "../sgemm.h"

#define BLOCK_SIZE 16
#define TILE_SIZE 32
#define TILE_M 2
#define TILE_N 2

// row major matrix
__global__ void sgemm_tile_reg_kernel(const float* A, const float* B, float* C, int M, int N, int K)
{
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];
    
    // the c element caculate by one thread
    float sum[TILE_M][TILE_N] = {0.0f};

    // cal the index for each block and each thread
    int tx = threadIdx.x; // x define col
    int ty = threadIdx.y; // y define row

    // describe the output location of element
    int row_base = blockIdx.y * (BLOCK_SIZE * TILE_M) + ty * TILE_M;
    int col_base = blockIdx.x * (BLOCK_SIZE * TILE_N) + tx * TILE_N;

    for(int tileIdx = 0; tileIdx < (K + TILE_SIZE - 1) / TILE_SIZE; tileIdx ++) // describe tile extend to k
    {
        // store data into tile A & B
        // initial offset pos of A & B in each tile
        int a_col_base = tileIdx * TILE_SIZE;
        int b_row_base = tileIdx * TILE_SIZE;

        // store data into tile A imply
        // 11subtile:
        {
            int row = blockIdx.y * TILE_SIZE + ty;
            int col = a_col_base + tx;
            if(row < M && col < K) tileA[ty][tx] = A[row * K + col];
            else tileA[ty][tx] = 0.0f;
        }
        // 12subtile
        {
            int row = blockIdx.y * TILE_SIZE + ty;
            int col = a_col_base + 16 + tx;
            if(row < M && col < K) tileA[ty][tx + 16] = A[row * K + col];
            else tileA[ty][tx + 16] = 0.0f;
        }
        // 21subtile
        {
            int row = blockIdx.y * TILE_SIZE + 16 + ty;
            int col = a_col_base + tx;
            if(row < M && col < K) tileA[ty + 16][tx] = A[row * K + col];
            else tileA[ty + 16][tx] = 0.0f;            
        }
        // 22subtile
        {
            int row = blockIdx.y * TILE_SIZE + 16 + ty;
            int col = a_col_base + 16 +tx;
            if(row < M && col < K) tileA[ty + 16][tx + 16] = A[row * K + col];
            else tileA[ty + 16][tx + 16] = 0.0f; 
        }

        // store date into tileB imply
        // 11subtile
        {
            int row = b_row_base + ty;
            int col = blockIdx.x * TILE_SIZE + tx;
            if(row < K && col < N) tileB[ty][tx] = B[row * N + col];
            else tileB[ty][tx] = 0.0f; 
        }
        // 12subtile
        {
            int row = b_row_base + ty;
            int col = blockIdx.x * TILE_SIZE + 16 + tx;
            if(row < K && col < N) tileB[ty][tx + 16] = B[row * N + col];
            else tileB[ty][tx + 16] = 0.0f; 
        }
        // 21subtile
        {
            int row = b_row_base + 16 + ty;
            int col = blockIdx.x * TILE_SIZE + tx;
            if(row < K && col < N) tileB[ty + 16][tx] = B[row * N + col];
            else tileB[ty + 16][tx] = 0.0f; 
        }
        // 22subtile
        {
            int row = b_row_base + 16 + ty;
            int col = blockIdx.x * TILE_SIZE + 16 + tx;
            if(row < K && col < N) tileB[ty + 16][tx + 16] = B[row * N + col];
            else tileB[ty + 16][tx + 16] = 0.0f; 
        }

        __syncthreads();

        for(int k = 0; k < TILE_SIZE; k++)
        {
            sum[0][0] += tileA[2 * ty][k] * tileB[k][tx * 2];
            sum[0][1] += tileA[2 * ty][k] * tileB[k][tx * 2 + 1];
            sum[1][0] += tileA[2 * ty + 1][k] * tileB[k][tx * 2];
            sum[1][1] += tileA[2 * ty + 1][k] * tileB[k][tx * 2 + 1];
        }
        __syncthreads();
    }

    // each write in need safety check
    for(int i = 0; i < TILE_M; i++)
    {
        for(int j = 0; j < TILE_N; j++)
        {
            int row = row_base + i;
            int col = col_base + j;
            if(row < M && col < N) C[row * N + col] = sum[i][j];
        }
    }
}

namespace cuda_op_lab::sgemm {

// TODO(v2): shared-memory tiling + register tiling.
void launch_sgemm_register_tiled(const float* A, const float* B, float* C, int m, int n, int k, cudaStream_t stream) 
{
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 gridSize((n + TILE_SIZE - 1) / TILE_SIZE, (m + TILE_SIZE - 1) / TILE_SIZE, 1);

    sgemm_tile_reg_kernel<<<gridSize, blockSize, 0, stream>>>(A, B, C, m, n, k);
}

}  // namespace cuda_op_lab::sgemm