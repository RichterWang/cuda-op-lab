#include <cuda_runtime.h>

#include "../sgemm.h"

// define of the baseline, need modify through ncu
#define BM 16
#define BN 16
#define TM 64
#define TN 64
#define TK 32

#define tileM 4
#define tileN 4

// use xor swissle to solve bank conflict
__device__ inline int swz(int row, int col){
    return col ^ ((row & 7) << 2);
} // change the col index to aviod bank conflict

// use asm to realize asynconization
// there is the reference code
__device__ __forceinline__
void cp_async_16(void* shared_ptr, const void* global_ptr, bool pred) {
#if __CUDA_ARCH__ >= 800
    const unsigned shared_address = static_cast<unsigned>(__cvta_generic_to_shared(shared_ptr)); // switch general ptr to shared mem ptr

    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
        :
        : "r"(shared_address), "l"(global_ptr), "r"(pred ? 16 : 0)
    );
#else
    if (pred) *reinterpret_cast<float4*>(shared_ptr) = *reinterpret_cast<const float4*>(global_ptr);
    else *reinterpret_cast<float4*>(shared_ptr) = float4{0, 0, 0, 0};
#endif
}

// archive the commit to a group
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

// wait for the group to finish， n is the number of groups to wait for, 0 means wait for all groups
template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

// ==================================================================================================================
__device__ __forceinline__ void load_tileA(const float* __restrict__ A, float (&tileA)[2][64][32], int base_A_row, int base_A_col, int M, int K, int tid, int current_buffer)
{
    #pragma unroll
    for(int i = 0; i < 2; i++)
    {
        int row = 2 * (tid / 8) + i;
        int col = 4 * (tid % 8);

        int global_row = base_A_row + row;
        int global_col = base_A_col + col;
        bool pred = (global_row < M) && (global_col < K - 3);

        cp_async_16(&tileA[current_buffer][row][swz(row, col)], &A[(global_row) * K + (global_col)], pred);
    }
}

__device__ __forceinline__ void load_tileB(const float* __restrict__ B, float (&tileB)[2][32][64], int base_B_row, int base_B_col, int K, int N, int tid, int current_buffer)
{
    #pragma unroll
    for(int i = 0; i < 2; i++)
    {
        int row = 2 * (tid / 16) + i;
        int col = 4 * (tid % 16);
        
        int global_row = base_B_row + row;
        int global_col = base_B_col + col;
        bool pred = (global_row < K) && (global_col < N - 3);

        cp_async_16(&tileB[current_buffer][row][col], &B[global_row * N + global_col], pred);
    }
}

// compute kernel ==================================================================================================================
__device__ __forceinline__ void compute_tile(const float tileA[2][64][32], const float tileB[2][32][64], float sum[4][4], int current_buffer, int tid)
{
    #pragma unroll
    for(int k = 0; k < TK; k++)
    {
        float regA[4] = {0.0f};
        float regB[4] = {0.0f};

        #pragma unroll
        for(int i = 0; i < tileM; i++) regA[i] = tileA[current_buffer][(tid / 16) * 4 + i][swz((tid / 16) * 4 + i, k)];

        // force compiler to use vectorized load, and avoid bank conflict
        const float4 b4 = *reinterpret_cast<const float4*>(&tileB[current_buffer][k][(tid % 16) * 4]);
        regB[0] = b4.x;
        regB[1] = b4.y;
        regB[2] = b4.z;
        regB[3] = b4.w;

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
}

// main kernel ====================================================================================================
// Requires: K % 4 == 0 && N % 4 == 0
//   - cp.async 16B needs the row stride (K for A, N for B) aligned to 16B
//   - the 16B predicate granularity would silently drop partial tail blocks
//   - the float4 epilogue store needs &C[r*N + col] 16B-aligned
__global__ void sgemm_asynchronous_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K)
{
    // double buffering and shared mem padding
    __shared__ float tileA[2][64][32];
    __shared__ float tileB[2][32][64];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = threadIdx.y * 16 + threadIdx.x;

    int base_A_row = blockIdx.y * TM;
    int base_B_col = blockIdx.x * TN;

    // cp.async write in tileA and tileB of first round
    load_tileA(A, tileA, base_A_row, 0, M, K, tid, 0);
    load_tileB(B, tileB, 0, base_B_col, K, N, tid, 0);
    cp_async_commit();

    float sum[4][4] = {0.0f};

    const int num_tiles = (K + TK - 1) / TK;

    for(int k = 1; k < num_tiles; k++)
    {
        // double buffering
        int current_buffer = (k - 1) % 2;
        int next_buffer = k % 2;

        // load the next tileA and tileB, and commit the load
        load_tileA(A, tileA, base_A_row, k * TK, M, K, tid, next_buffer);
        load_tileB(B, tileB, k * TK, base_B_col, K, N, tid, next_buffer);
        cp_async_commit();

        // wait for the current tileA and tileB to finish loading
        cp_async_wait<1>();
        __syncthreads();

        // compute the current tileA and tileB
        // caculate current data register write in
        compute_tile(tileA, tileB, sum, current_buffer, tid);

        __syncthreads();
    }

    // final stage write in
    // make sure the last tileA and tileB is loaded
    cp_async_wait<0>();
    __syncthreads();
    compute_tile(tileA, tileB, sum, (num_tiles - 1) & 1, tid);

    // write back to C in global memory
    // int localA_row = ty * 4 + i;
    // int localB_col = tx * 4 + i;
    int row_offset = blockIdx.y * TM + ty * 4;
    int col_offset = blockIdx.x * TN + tx * 4;

    const bool if_full = (base_A_row + TM <= M) && (base_B_col + TN <= N);
    
    if(if_full)
    {
        #pragma unroll
        for(int i = 0; i < 4; i++)
        {
            float4 temp = make_float4(sum[i][0], sum[i][1], sum[i][2], sum[i][3]);
            *reinterpret_cast<float4*>(&C[(row_offset + i) * N + col_offset]) = temp;
        }
    }
    else{
        #pragma unroll
        for(int i = 0; i < 4; i++)
        {
            for(int j = 0; j < 4; j++)
            {
                if(row_offset + i < M && col_offset + j < N)
                {
                    C[(row_offset + i) * N + col_offset + j] = sum[i][j];
                }
            }
        }
    }
}

namespace cuda_op_lab::sgemm{
void launch_sgemm_asynchronous(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream)
{
    dim3 blockSize(BM, BN, 1);
    dim3 gridSize((N + TN - 1) / TN, (M + TM - 1) / TM, 1);
    sgemm_asynchronous_kernel<<<gridSize, blockSize, 0, stream>>>(A, B, C, M, N, K);
}
}