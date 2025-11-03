#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include "cuda_runtime.h"
#include "thrust/device_vector.h"

// __host__ means the function runs on the host (CPU)
// __device__ means the function runs on the device (GPU) but cannot be called from the host so must be used within a __global__ function
// __global__ means the function runs on the device (GPU) and is called from the host (CPU) also called a kernel

__host__ __device__ inline double sinsum (double x, uint terms){
    // sin(x) = x - x^3/3! + x^5/5! ... using the taylor series
    //first term of the series
    double term = x;
    // sum at 1
    double sum = term;
    // second term
    double   x2 = x*x;

    for (uint n = 1; n < terms; n++){
        // compute next term
        term *= -x2 / (double) (2*n*(2*n+1));
        sum += term;
    }
    return sum;
}

__global__ void gpu_sin(double *sums, long long steps, uint terms,
double step_size){
    //thread id as step as each thread computes one step
    uint step = blockIdx.x * blockDim.x + threadIdx.x;

    if (step < steps){
        double x = step * step_size;
        sums[step] = sinsum(x, terms); // store result in sums array
    }
}

int main( int argc, char *argv[]){
    long long steps = argc > 1 ? atoll(argv[1]) : 1000000; // atoll for long long
    uint terms = argc > 2 ? atoll(argv[2]) : 10; // atoi for int ie casts char* to int

    uint threads_per_block = argc > 3 ? atoi(argv[3]) : 256;
    uint blocks_per_grid = (steps + threads_per_block -1) / threads_per_block;
    if (blocks_per_grid > 65535) blocks_per_grid = 65535; // max blocks per grid in cuda
    printf("steps %lld blocks %d threads/block %d total threads %d\n", steps, blocks_per_grid, threads_per_block, blocks_per_grid*threads_per_block);

    //allocate device memory using thrust
    thrust::device_vector<double> d_sums(steps); // array of size steps declared on device
    double *d_sums_ptr = thrust::raw_pointer_cast(d_sums.data()); // get raw pointer to device memory has to be used as we cant pass thrust vectors to cuda kernels

    double step_size = M_PI / (double) (steps -1);

    auto start = std::chrono::high_resolution_clock::now();

    double cpu_sum = 0.0;

    // for (int step = 0; step < steps; step++){
    //     float x = step * step_size;
    //     cpu_sum += sinsum(x, terms);
    // }
    // launch kernel
    gpu_sin<<<blocks_per_grid, threads_per_block>>>(d_sums_ptr, steps, terms, step_size);
    cudaDeviceSynchronize();
    cpu_sum = thrust::reduce(d_sums.begin(), d_sums.end(), 0.0, thrust::plus<double>());
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;


    //trapezoidal rule correction
    cpu_sum -= 0.5 *(sinsum(0.0,terms)+sinsum(M_PI, terms));
    cpu_sum *= step_size;

    printf("cpu sum = %.10f,steps %d terms %d time %.3f ms\n", cpu_sum, steps, terms, elapsed.count()*1000.0);

    return 0;


}
