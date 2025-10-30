//kernel def

__global__ void VecAdd(float* A, float* B, float* C){
    int i = threadIdx.x
    C[i] = A[i] + B[i];
}

int main(){
    // invoke the kernel here
    int N = 256;
    float *A, *B, *C;
    cudaMallocManaged(&A, N * sizeof(float));
    cudaMallocManaged(&B, N * sizeof(float));
    cudaMallocManaged(&C, N * sizeof(float));
    for (int i = 0; i < N; i++) {
        A[i] = static_cast<float>(i);
        B[i] = static_cast<float>(i * 2);
    }
    VecAdd<<<1, N>>>(A, B, C);
    cudaDeviceSynchronize();
    
}
