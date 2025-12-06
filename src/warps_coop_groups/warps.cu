// function using the reduce kernel methodology showing the improvements in perf by using
// warps

#include <stdio.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <random>

//using a template to define the reduce kernel
template <int BLOCK_SIZE>
__global__ void reduce( float* __restrict__ sums, const float *__restrict__ data, int n){
    //assumses blockdim == BLOCK_SIZE
    __shared__ float s[BLOCK_SIZE];
    int id = threadIdx.x; //rank within block
    s[id] = 0;

    for (int tid = BLOCK_SIZE*blockIdx.x+threadIdx.x; tid < n; tid += BLOCK_SIZE*gridDim.x){
        // for unique thread id and that thread id lower than n, accumulate data
        // then increment by total number of threads in grid ie stride so adjacent threads access data from shared memory reducing memory load overhead
        s[id] += data[tid];
    }

    __syncthreads(); // synchronize threads within block
    // now do reduction in shared memory ie folding in on its self
    if (BLOCK_SIZE>512 && id < 512 && id + 512 < BLOCK_SIZE) s[id] += s[id + 512];
    __syncthreads();
    if (BLOCK_SIZE>256 && id < 256 && id + 256 < BLOCK_SIZE) s[id] += s[id + 256];
    __syncthreads();
    if (BLOCK_SIZE>128 && id < 128 && id + 128 < BLOCK_SIZE) s[id] += s[id + 128];
    __syncthreads();
    if (BLOCK_SIZE>64 && id < 64 && id + 64 < BLOCK_SIZE) s[id] += s[id + 64];
    __syncthreads();
    if (id < 32){
        s[id] += s[id + 32];
        __syncwarp(); // syncwarp for the final warp as all threads are in the same warp so syncthreads will cause unnecessary overhead
        if (id < 16) s[id] += s[id + 16];
        __syncwarp();
        if (id < 8) s[id] += s[id + 8];
        __syncwarp();
        if (id < 4) s[id] += s[id + 4];
        __syncwarp();
        if (id < 2) s[id] += s[id + 2];
        __syncwarp();
        if (id < 1) s[id] += s[id + 1];
        __syncwarp();
    }

    if (id == 0){
        sums[blockIdx.x] = s[0]; // write result for this block to global memory
    }
}

__global__ void reduce_syncthreads(float* __restrict__ y, float* __restrict__ x, int m) {
  // using shared memory
  extern __shared__ float tsum[]; // array of floats in shared memory

  int id = threadIdx.x;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  int stride = gridDim.x * blockDim.x;

  tsum[id] = 0.0f;

  for (int k = tid; k < m; k += stride)
    tsum[id] += x[k];
  __syncthreads();

  if (id < 256 && id + 256 < blockDim.x)
    tsum[id] += tsum[id + 256];
  __syncthreads();
  if (id < 128)
    tsum[id] += tsum[id + 128];
  __syncthreads();
  if (id < 64)
    tsum[id] += tsum[id + 64];
  __syncthreads();
  if (id < 32)
    tsum[id] += tsum[id + 32];
  __syncthreads();

  if (id < 16)
    tsum[id] += tsum[id + 16];
  __syncthreads();
  if (id < 8)
    tsum[id] += tsum[id + 8];
  __syncthreads();
  if (id < 4)
    tsum[id] += tsum[id + 4];
  __syncthreads();
  if (id < 2)
    tsum[id] += tsum[id + 2];
  __syncthreads();
  if (id < 1)
    tsum[id] += tsum[id + 1];
  __syncthreads();

  if (id == 0)
    y[blockIdx.x] = tsum[0] + tsum[1];
}

int main(){
    int n = 1<<29;
    int nblocks = 128;
    int nthreads = 1024;

    size_t shared_mem_size = nthreads * sizeof(float); // for dynamically allocated shared memory in reduce_syncthreads

    // float flops = 2 *

    // allocate host memory
    thrust::host_vector<float> h_data(n);
    thrust::host_vector<float> h_sums(nblocks);

    thrust::device_vector<float> d_sums(nblocks);

    // initialize data with random numbers
    std::default_random_engine generator;
    std::uniform_real_distribution<float> fran(0.0, 1.0);
    // auto start_init = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n; i++){
        h_data[i] = fran(generator);
    }
    //copy data to device
    thrust::device_vector<float> d_data = h_data;
    // auto end_init = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<float, std::milli> duration_init = end_init - start_init;
    // std::cout << "Data initialization time: " << duration_init.count() << " ms" << std::endl;


    // launch kernel
    auto start = std::chrono::high_resolution_clock::now();
    reduce<1024><<<nblocks, nthreads>>>(d_sums.data().get(),
                                       d_data.data().get(), n);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = end - start;
    std::cout << "Kernel execution time (Syncwarps): " << duration.count() << " ms" << std::endl;

    auto start1 = std::chrono::high_resolution_clock::now();
    reduce_syncthreads<<<nblocks, nthreads, shared_mem_size>>>(d_sums.data().get(),
                                       d_data.data().get(), n);
    cudaDeviceSynchronize();
    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration1 = end1 - start1;
    std::cout << "Kernel execution time (Syncthreads): " << duration1.count() << " ms" << std::endl;

    return 0;

};
