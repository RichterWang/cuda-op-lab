#include <cuda_runtime.h>

#include "../sgemm.h"

#define BM 16
#define BN 16
#define TM 64
#define TN 64
#define TK 32

#define tileM 4
#define tileN 4

// goal of this kernel: double buffering, avoid bank conflict, reg write back, repair edge
// optimize idea:
//  fix bank conflict under tileB
//  introduce cp.async to get ture double buffering
__global__ void sgemm_double_buffering_kernel(const float* A, const float* B, float* C, int M, int N, int K)
{
    // double buffering and shared mem padding
    __shared__ float tileA[2][64][32];
    __shared__ float tileB[2][32][64];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // write in the fist gruop of buffer at first
    int tid = ty * 16 + tx; // expand thread index to global, only use for A tile
    int a_init_base_row = blockIdx.y * TM + tid % TM;
    int a_init_base_col = (tid / TM) * 8;

    #pragma unroll
    for(int i = 0; i < 2; i++)
    {
        int global_row = a_init_base_row;
        int global_col = a_init_base_col + 4 * i;
        int tile_row = tid % TM;
        int tile_col = (tid / TM) * 8 + 4 * i;

        if(global_row < M && global_col + 4 <= K)
        {
            float4 temp = *reinterpret_cast<const float4*>(&A[global_row * K + global_col]);
            // remember to use local memory to access
            tileA[0][tile_row][tile_col + 0] = temp.x;
            tileA[0][tile_row][tile_col + 1] = temp.y;
            tileA[0][tile_row][tile_col + 2] = temp.z;
            tileA[0][tile_row][tile_col + 3] = temp.w; 
        }
        else
        {
            if(global_row < M && global_col + 4 > K){
                #pragma unroll
                for(int j = 0; j < K - global_col; j++) tileA[0][tile_row][tile_col + j] = A[global_row * K + global_col +j];
            }
            else{
                #pragma unroll
                for(int j = 0; j < 4; j++) tileA[0][tile_row][tile_col + j] = 0.0f;
            }
        }
    }

    // to ensure cache localty, use tid again
    int b_init_base_row = tid % TK;
    int b_init_base_col = blockIdx.x * TN + (tid / TK) * 8;

    #pragma unroll
    for(int i = 0; i < 2; i++)
    {
        int global_row = b_init_base_row;
        int global_col = b_init_base_col + 4 * i;
        int tile_row = tid % TK;
        int tile_col = (tid / TK) * 8 + 4 * i;

        if(global_row < K && global_col <= N - 4)
        {
            float4 temp = *reinterpret_cast<const float4*>(&B[global_row * N + global_col]);
            tileB[0][tile_row][tile_col + 0] = temp.x;
            tileB[0][tile_row][tile_col + 1] = temp.y;
            tileB[0][tile_row][tile_col + 2] = temp.z;
            tileB[0][tile_row][tile_col + 3] = temp.w;             
        }
        else{
            if(global_row < K && global_col > N - 4){
                #pragma unroll
                for(int j = 0; j < N - global_col; j++) tileB[0][tile_row][tile_col + j] = B[global_row * N + global_col +j];  
            }
            else{
                #pragma unroll
                for(int j = 0; j < 4; j++) tileB[0][tile_row][tile_col + j] = 0.0f;  
            }
        }
    }

    // create sum matrix to store result of each thread
    float sum[4][4] = {0.0f};

    __syncthreads();

    // main loop
    for(int tileId = 1; tileId <= (K + TK - 1)/TK; tileId++)
    {
        // buffer switching
        int curr_buf = (tileId - 1) & 1;
        int next_buf = tileId & 1;

        if(tileId != (K + TK - 1)/TK)
        {
            int a_base_row = blockIdx.y * TM + tid % TM;
            int a_base_col = tileId * TK + (tid / TM) * 8;

            #pragma unroll
            for(int i = 0; i < 2; i++)
            {
                int global_row = a_base_row;
                int global_col = a_base_col + 4 * i;
                int tile_row = tid % TM;
                int tile_col = (tid / TM) * 8 + 4 * i;

                if(global_row < M && global_col + 4 <= K)
                {
                    float4 temp = *reinterpret_cast<const float4*>(&A[global_row * K + global_col]);
                    // remember to use local memory to access
                    tileA[next_buf][tile_row][tile_col + 0] = temp.x;
                    tileA[next_buf][tile_row][tile_col + 1] = temp.y;
                    tileA[next_buf][tile_row][tile_col + 2] = temp.z;
                    tileA[next_buf][tile_row][tile_col + 3] = temp.w; 
                }
                else
                {
                    if(global_row < M && global_col + 4 > K){
                        #pragma unroll
                        for(int j = 0; j < K - global_col; j++) tileA[next_buf][tile_row][tile_col + j] = A[global_row * K + global_col +j];
                    }
                    else{
                        #pragma unroll
                        for(int j = 0; j < 4; j++) tileA[next_buf][tile_row][tile_col + j] = 0.0f;
                    }
                }
            }

            int b_base_row = tileId * TK + tid % TK;
            int b_base_col = blockIdx.x * TN + (tid / TK) * 8;

            #pragma unroll
            for(int i = 0; i < TK / BM; i++)
            {
                int global_row = b_base_row;
                int global_col = b_base_col + 4 * i;
                int tile_row = tid % TK;
                int tile_col = (tid / TK) * 8 + 4 * i;

                if(global_row < K && global_col <= N - 4)
                {
                    float4 temp = *reinterpret_cast<const float4*>(&B[global_row * N + global_col]);
                    tileB[next_buf][tile_row][tile_col + 0] = temp.x;
                    tileB[next_buf][tile_row][tile_col + 1] = temp.y;
                    tileB[next_buf][tile_row][tile_col + 2] = temp.z;
                    tileB[next_buf][tile_row][tile_col + 3] = temp.w;             
                }
                else{
                    if(global_row < K && global_col > N - 4){
                        #pragma unroll
                        for(int j = 0; j < N - global_col; j++) tileB[next_buf][tile_row][tile_col + j] = B[global_row * N + global_col +j];  
                    }
                    else{
                        #pragma unroll
                        for(int j = 0; j < 4; j++) tileB[next_buf][tile_row][tile_col + j] = 0.0f;  
                    }
                }
            }
        }

        // caculate current data register write in
        #pragma unroll
        for(int k = 0; k < TK; k++)
        {
            float regA[4] = {0.0f};
            float regB[4] = {0.0f};

            #pragma unroll
            for(int i = 0; i < tileM; i++) regA[i] = tileA[curr_buf][ty * 4 + i][k];

            #pragma unroll
            for(int i = 0; i < tileN; i++) regB[i] = tileB[curr_buf][k][tx * 4 + i];

            #pragma unroll
            for(int i = 0; i < tileM; i++)
            {
                #pragma unroll
                for(int j = 0; j < tileN; j++)
                {
                    sum[i][j] += regA[i] * regB[j];
                }
            }
        }
        
        __syncthreads();
    }

    // write back to C in global memory
    // int localA_row = ty * 4 + i;
    // int localB_col = tx * 4 + i;
    int row_offset = blockIdx.y * TM + ty * 4;
    int col_offset = blockIdx.x * TN + tx * 4;
    
    #pragma unroll
    for(int i = 0; i < 4; i++)
    {
        if(row_offset + i < M && col_offset + 3 < N)
        {
            float4 temp = *reinterpret_cast<const float4*>(sum[i]);
            *reinterpret_cast<float4*>(&C[(row_offset + i) * N + col_offset]) = temp;
        }
        else{
            for(int j = 0; j < 4; j++) if(row_offset + i < M && col_offset + j < N) C[(row_offset + i) * N + col_offset + j] = sum[i][j];
        }
    }
}

namespace cuda_op_lab::sgemm{

void launch_sgemm_double_buf(const float* A, const float* B, float* C, int m, int n, int k, cudaStream_t stream)
{
    dim3 blockSize(BM, BN, 1);
    dim3 gridSize((n + TN - 1)/TN, (m + TM - 1)/ TM, 1);

    sgemm_double_buffering_kernel<<<gridSize, blockSize, 0, stream>>>(A, B, C, m, n, k);
}
}