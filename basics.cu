#include <stdio.h>
#define N_THREADS 1024
#define N_BLOCKS 1000
//kernel def

__global__ void VecAdd(float* A, float* B, float* C){
    // printf("Thread idx: %d\n", threadIdx.x);

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    C[i] = A[i] + B[i];
}

__global__ void PrintThreadIdx(){
    printf("Thread idx vector: %d, %d, %d\n", threadIdx.x, threadIdx.y, threadIdx.z);
}

int main(){
    // invoke the kernel here
    // int N = 256;
    float *A, *B, *C;
    cudaMallocManaged(&A, N_THREADS * N_BLOCKS * sizeof(float));
    cudaMallocManaged(&B, N_THREADS * N_BLOCKS * sizeof(float));
    cudaMallocManaged(&C, N_THREADS * N_BLOCKS * sizeof(float));

    // fill the arrays A and B on the host
    for (int i = 0; i < N_THREADS * N_BLOCKS; i++) {
        A[i] = static_cast<float>(i);
        B[i] = static_cast<float>(i * 2);
    }
    // runs the VecAdd kernel, with N_THREADS threads in N_BLOCKS blocks
    VecAdd<<<N_BLOCKS, N_THREADS>>>(A, B, C);
    PrintThreadIdx<<<N_BLOCKS, N_THREADS>>>();
    cudaDeviceSynchronize();

    // Print results
    // for (int i = 0; i < N_THREADS * N_BLOCKS; i++) {
    //     printf("C[%d] = %f + %f = %f\n", i, A[i], B[i], C[i]);
    // }

    // const float eps = 1e-8;
    // // verify the result
    // for (int i = 0; i < N_THREADS * N_BLOCKS; i++) {
    //     if (fabs(C[i] - (A[i] + B[i])) > eps) {
    //         printf("Error at index %d: %f + %f != %f\n", i, A[i], B[i], C[i]);
    //         return -1;
    //     }
    // }

    return 0;
}
