#include <cuda_runtime.h>

#include "../sgemm.h"

#define BLOCK_SIZE 16
#define TILE_SIZE 32
#define TILE_M 2
#define TILE_N 2

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

        // 11subtile:
        {
            int row = blockIdx.y * TILE_SIZE + ty;
            int col = a_col_base + tx;
            if(row < M && col < K) tileA[tx][ty] = A[row * K + col];
            else tileA[tx][ty] = 0.0f;
        }

    }
}

namespace cuda_op_lab::sgemm {

// TODO(v2): shared-memory tiling + register tiling.
void launch_sgemm_register_tiled(const float* A, const float* B, float* C, int m, int n, int k, cudaStream_t stream) 
{
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 gridSize((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);

    sgemm_tile_reg_kernel<<<gridSize, blockSize, 0, stream>>>(A, B, C, m, n, k);
}

}  // namespace cuda_op_lab::sgemm