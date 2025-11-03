#include "cx.h"
#include "cxtimers.h"
#include <random>

__global__ void reduce0(double* x, int m){
    int tid = threadIdx.x + blockIdx.x * blockDim.x; // global thread index for 1d grid
    x[tid] += x[tid+m]; //
}

int main(int argc, char** argv) {
    int N = (argc >1) ? atoi(argv[1]) : 1 << 24; //shifts bit 24 to the left ie 2^24

    thrust::host_vector<double> h_x(N);
    thrust::device_vector<double> d_x(N);

    std::default_random_engine gen(1);// random number generator with fixed seed for reproducibility
    std::uniform_real_distribution<double> fran(0.0,1.0);

    for(int k = 0; k<N; k++){
        h_x[k] = fran(gen);
    }
    d_x = h_x; // copy data to device

    auto start = std::chrono::high_resolution_clock::now();

    double host_sum = 0.0;
    for(int k = 0; k<N; k++){
        host_sum += h_x[k];
    } // reduce on host

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;



    // printf("sum of %d random numbers: host %.1f %.3f ms\n", N, host_sum, elapsed.count() * 1000);

    auto start2 = std::chrono::high_resolution_clock::now();

    for (int m = N/2; m>=1; m/=2){
        int threads = std::min(256, m);
        int nblocks = std::max(m/256,1);
        reduce0<<<nblocks, threads>>>(d_x.data().get(), m);

    } // reduce on device
    cudaDeviceSynchronize();
    double gpu_sum = d_x[0]; // copy result back to host

    auto end2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed2 = end2 - start2;

    printf("sum of %d random numbers: host %.3f %.3f ms, GPU %.3f %.3f ms\n", N, host_sum, elapsed.count() * 1000, gpu_sum, elapsed2.count() * 1000);
    printf("Speed Up: %.2fx\n", elapsed.count()/elapsed2.count());
    return 0;

}
