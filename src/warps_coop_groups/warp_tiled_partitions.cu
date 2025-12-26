// tiled partitions in warps come with instrinsic functions that improve efficiency and synchronization btn threads
#include <cooperative_groups.h>
#include "cooperative_groups/reduce.h"
#include <cuda_runtime.h>
#include "../cx.h"
#include "../cxtimers.h"
#include <thrust/random.h>
#include <iostream>

namespace cg = cooperative_groups; // or using

//showcases use of tiled partitions member functions
template <int blockSize> __global__ void
reduce6(r_Ptr<float> sums, cr_Ptr<float> data, int n){ //r_Ptr<float> is a restricted float pointer the c is const
    // This template kernel assumes blockDim.x = blockSize
    // and that blockSize ≤ 1024
    __shared__ float s[blockSize];
    
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    
    int id = block.thread_rank();
    
    s[id] = 0.0f;
    
    for(int tid=grid.thread_rank();tid < n; tid+=grid.size()) s[id] += data[tid]; //thread linear addressing loop
    
    block.sync(); // syncs all threads in block equivalent to __syncthreads()
    
    if(blockSize>512 && id < 512 && id + 512 <blockSize) s[id] += s[id+512];
    block.sync();
    
    if(blockSize>256 && id < 256 && id + 256 <blockSize) s[id] += s[id+256];
    block.sync();
    
    if(blockSize>128 && id < 128 && id + 128 <blockSize) s[id] += s[id+128];
    block.sync();
    
    if(blockSize>64 && id < 64 && id + 64 <blockSize) s[id] += s[id+64];
    block.sync();
    
    //dealing with warp 0
    if(warp.meta_group_rank() == 0){
        s[id] += s[id + 32]; warp.sync();
        s[id] += warp.shfl_down(s[id], 16);
        s[id] += warp.shfl_down(s[id], 8);
        s[id] += warp.shfl_down(s[id], 4);
        s[id] += warp.shfl_down(s[id], 2);
        s[id] += warp.shfl_down(s[id], 1);
        
        if(id == 0) sums[blockIdx.x] = s[0];//copy to global memory
    }
}

//showcases intrawarp member functions
__global__ void reduce7(r_Ptr<float> sums, cr_Ptr<float> data, int n){
    // this kernel reduces data within warps then sums up the warps for the final result
    // works for any block size multiple of 32
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    
    float v = 0.0f;
    for(int tid=grid.thread_rank(); tid<n; tid+=grid.size()) v += data[tid];
    warp.sync();
    //warp level reduce
    // each warp reduces its own data independently and in parallel 
    // atomic add then sums up the final result by summing the output of the individual warps
    v += warp.shfl_down(v,16);
    v += warp.shfl_down(v,8);
    v += warp.shfl_down(v,4);
    v += warp.shfl_down(v,2);
    v += warp.shfl_down(v,1);
    
    if(warp.thread_rank()==0){
        atomicAdd(&sums[block.group_index().x], v); // atomicAdd sums accross warps
        //atomicAdd_block is even fasterthan
    }
}

//cg built in reduce
__global__ void reduce8(r_Ptr<float> sums, cr_Ptr<float> data, int n){
    // this kernel reduces data within warps then sums up the warps for the final result
    // works for any block size multiple of 32
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    
    float v = 0.0f;
    for(int tid=grid.thread_rank(); tid<n; tid+=grid.size()) v += data[tid];
    warp.sync();
    v = cg::reduce( warp, v, cg::plus<float>());
    
    if(warp.thread_rank()==0){
        atomicAdd(&sums[block.group_index().x], v); 
    }
}

int main(int argc, char* argv[]){
    const int N =(argc > 1) ? atoi(argv[1]) : 1<<24;  // Number of elements in the tensor
    const int blockSize = 1024;  // Block size for the kernel
    const int gridSize = (N + blockSize - 1) / blockSize;  // Calculate grid size
    
    // Create thrust vectors for input data and partial sums
    thrust::device_vector<float> d_data(N);
    thrust::host_vector<float> h_data(N);
    thrust::device_vector<float> d_sums(gridSize);
    
    // Create a random number generator
    thrust::default_random_engine rng;
    thrust::uniform_real_distribution<float> dist(0.0f, 1.0f);
    cxxtimer::Timer timer2;
    timer2.start();
    // Fill the input vector with random numbers between 0 and 1
    thrust::generate(h_data.begin(), h_data.end(), [&](){ return dist(rng); });
    thrust::copy(h_data.begin(), h_data.end(), d_data.begin());
    timer2.stop();
    std::cout << "Data generation time: " << timer2.count<cxxtimer::ms>() << " ms" << std::endl;
    
    // Create timers to measure execution time
    cxxtimer::Timer timer;
    cxxtimer::Timer timer_reduce7;
    cxxtimer::Timer timer_reduce8;
    
    // Start the timer
    timer.start();
    
    // Launch the reduce6 kernel
    reduce6<blockSize><<<gridSize, blockSize>>>(
        d_sums.data().get(),
        d_data.data().get(),
        N
    );
    
    // Wait for the kernel to complete
    cudaDeviceSynchronize();
    
    // Stop the timer
    timer.stop();
    
    // Calculate the final sum on the host
    float final_sum_reduce6 = 0.0f;
    for(int i = 0; i < gridSize; i++) {
        final_sum_reduce6 += d_sums[i];
    }
    
    // Print results
    std::cout << "Number of elements: " << N << std::endl;
    std::cout << "Block size: " << blockSize << std::endl;
    std::cout << "Grid size: " << gridSize << std::endl;
    std::cout << "reduce6 final sum: " << final_sum_reduce6 << std::endl;
    std::cout << "Expected sum (approximate): " << N * 0.5f << std::endl;
    std::cout << "reduce6 kernel execution time: " << timer.count<cxxtimer::ms>() << " ms" << std::endl;
    
    // Prepare and time reduce7
    cudaMemset(d_sums.data().get(), 0, gridSize * sizeof(float));
    timer_reduce7.start();
    
    reduce7<<<gridSize, blockSize>>>(
        d_sums.data().get(),
        d_data.data().get(),
        N
    );
    
    cudaDeviceSynchronize();
    timer_reduce7.stop();
    
    float final_sum_reduce7 = 0.0f;
    for(int i = 0; i < gridSize; i++) {
        final_sum_reduce7 += d_sums[i];
    }
    
    std::cout << "reduce7 final sum: " << final_sum_reduce7 << std::endl;
    std::cout << "reduce7 kernel execution time: " << timer_reduce7.count<cxxtimer::ms>() << " ms" << std::endl;
    
    // Prepare and time reduce8
    cudaMemset(d_sums.data().get(), 0, gridSize * sizeof(float));
    timer_reduce8.start();
    
    reduce8<<<gridSize, blockSize>>>(
        d_sums.data().get(),
        d_data.data().get(),
        N
    );
    
    cudaDeviceSynchronize();
    timer_reduce8.stop();
    
    float final_sum_reduce8 = 0.0f;
    for(int i = 0; i < gridSize; i++) {
        final_sum_reduce8 += d_sums[i];
    }
    
    std::cout << "reduce8 final sum: " << final_sum_reduce8 << std::endl;
    std::cout << "reduce8 kernel execution time: " << timer_reduce8.count<cxxtimer::ms>() << " ms" << std::endl;
    
    return 0;
}