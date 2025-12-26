//shows using vector loading for optimised memory access 
#include <cooperative_groups.h>
#include "cooperative_groups/reduce.h"
#include <cuda_runtime.h>
#include "../cx.h"
#include "../cxtimers.h"
#include <thrust/random.h>
#include <iostream>

namespace cg = cooperative_groups; // or using

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

//reduce 7 vl showing the optimisation by vector loading
//  https://devblogs.nvidia.com/cuda-pro-tip-increase-performance-with-vectorized-memory-access/
__global__ void reduce7_vl(r_Ptr<float> sums, cr_Ptr<float> data, int n){
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    
    float4 v4= {0.0f,0.0f,0.0f,0.0f}; //float4 is a struct of 4 floats denoted x,y,z,w
    for(int tid = grid.thread_rank(); tid < n/4; tid += grid.size()) {
        const float4 loaded_v4 = reinterpret_cast<const float4 *>(data)[tid];
        v4.x += loaded_v4.x;
        v4.y += loaded_v4.y;
        v4.z += loaded_v4.z;
        v4.w += loaded_v4.w;
    }
        
    //accumulate thread sums
    float v = v4.x + v4.y + v4.z + v4.w;
    //sync then reduce
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
    
    cxxtimer::Timer timer_reduce8;
    cxxtimer::Timer timer_reduce7_vl;
    
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
    
    //prep for next kernel
    cudaMemset(d_sums.data().get(), 0, gridSize * sizeof(float));
    
    timer_reduce7_vl.start();
    
    reduce7_vl<<<gridSize, blockSize>>>(
        d_sums.data().get(),
        d_data.data().get(),
        N
    );
    
    cudaDeviceSynchronize();
    timer_reduce7_vl.stop();
    
    float final_sum_reduce7_vl = 0.0f;
    for(int i = 0; i < gridSize; i++) {
        final_sum_reduce7_vl += d_sums[i];
    }
    
    std::cout << "reduce7_vl final sum: " << final_sum_reduce7_vl << std::endl;
    std::cout << "reduce7_vl kernel execution time: " << timer_reduce7_vl.count<cxxtimer::ms>() << " ms" << std::endl;
    
    

return 0;
}