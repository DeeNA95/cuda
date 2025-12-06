//displays uses and efficiency  of warps and cooperative groups
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include "cx.h"
#include "random"

namespace cg = cooperative_groups;

__device__ int a[512][512][512];
__device__ float b[512][512][512];

//kernel indicating use of tiled partitions
template <int T> __device__ void show_tile(const char *tag, cg::thread_block_tile<T> p){
    int rank = p.thread_rank(); // thread rank in tile
    int size = p.size(); // number of threads in tile
    int mrank = p.meta_group_rank(); // rank of tile in parent
    int msize = p.meta_group_size(); // number of tiles in parent
    printf("%s: rank %d of %d in tile %d of %d\n", tag, rank, size, mrank, msize);
}

__global__ void cgwarp(int id){
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block(); // definitions
// 32 thread warps
    auto warp32 = cg::tiled_partition<32>(block);
    auto warp16 = cg::tiled_partition<16>(block);
    auto warp8 = cg::tiled_partition< 8>(block);
    auto tile8 = cg::tiled_partition<8>(warp32);
    auto tile4 = cg::tiled_partition<4>(tile8);
    if(grid.thread_rank() == id) {
        printf("warps and subwarps for thread %d:\n",id);
        show_tile<32>("warp32", warp32);
        show_tile<16>("warp16", warp16);
        show_tile<8>("warp8", warp8);
        show_tile<8>("tile8", tile8);
        show_tile<4>("tile4", tile4);
    }
}

//kernel indicating use of cooperative groups
__global__ void coop3D(int nx,int ny,int nz,int id){
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    int x = block.group_index().x*block.group_dim().x+ block.thread_index().x;
    int y = block.group_index().y*block.group_dim().y+ block.thread_index().y;
    int z = block.group_index().z*block.group_dim().z+ block.thread_index().z;
    // if(x<nx && y<ny && z<nz) a[x][y][z] = id;
    if(x >=nx || y >=ny || z >=nz) return;
    int array_size = nx*ny*nz;
    // threads in one block
    int block_size = block.size();
    // blocks in grid
    int grid_size = grid.size()/block.size();
    // threads in whole grid
    int total_threads = grid.size();
    int thread_rank_in_block = block.thread_rank();
    int block_rank_in_grid = grid.thread_rank()/block.size();
    int thread_rank_in_grid = grid.thread_rank();

}

//template for reduce 5 matmul
//kernel indicating use of warp level primitives
template <int blockSize> __global__ void reduce5(r_Ptr<int> sums,cr_Ptr<int> data, int n){
    //requires blockDim.x = blockSize and blockSize is a power of 2 [64, 1024]
    __shared__ int s[blockSize];

    int id = threadIdx.x;
    s[id] = 0;
    for (int tid = blockSize*blockIdx.x+threadIdx.x; tid < n; tid += blockSize*gridDim.x)
        s[id] += data[tid];
    __syncthreads();

    if(blockSize > 512 && id < 512 && id+512 < blockSize)
        s[id] += s[id+512];
    __syncthreads();

    if(blockSize > 256 && id < 256 && id+256 < blockSize)
        s[id] += s[id+256];
    __syncthreads();

    if(blockSize > 128 && id < 128 && id+128 < blockSize)
        s[id] += s[id+128];
    __syncthreads();

    if(blockSize > 64 && id < 64 && id+64 < blockSize)
        s[id] += s[id+64];
    __syncthreads();
    //above 32 can be done as it will require more than one warp to complete
    // below 32 will require sync warp over sync threads as all threads will be in one warp


    if(id <32){
        s[id] += s[id + 32]; __syncwarp();
        if(id < 16) s[id] += s[id + 16]; __syncwarp();
        if(id < 8) s[id] += s[id + 8]; __syncwarp();
        if(id < 4) s[id] += s[id + 4]; __syncwarp();
        if(id < 2) s[id] += s[id + 2]; __syncwarp();
        if(id < 1) s[id] += s[id + 1]; __syncwarp();

        if(id == 0) sums[blockIdx.x] = s[0]; //block sum in index 0 at global mem
    }

}

int main(int argc, char **argv) {
    int id = (argc > 1) ? atoi(argv[1]) : 12345;
    int cgwarpBlocks = (argc > 2) ? atoi(argv[2]) : 28800;
    int cgwarpThreads = (argc > 3) ? atoi(argv[3]) : 256;

    // Default parameters
    int n = 1<<24;  // 16M elements for reduction
    constexpr int blockSize = 1024;

    // 3D grid parameters for coop3D
    int nx = 256, ny = 512, nz = 512;

    printf("=== CUDA Warp and Cooperative Groups Demo ===\n\n");
    printf("Array size for reduction: %d elements (%.2f MB)\n", n, n * sizeof(int) / 1e6);
    printf("3D array size: %d x %d x %d\n\n", nx, ny, nz);

    // Allocate host and device memory for reduction
    thrust::host_vector<int> data(n);
    thrust::device_vector<int> dev_data(n);
    int numBlocks = (n + blockSize - 1) / blockSize;
    thrust::device_vector<int> dev_sums(numBlocks);

    // Initialize data with 1s for easy verification
    for (int i = 0; i < n; i++) data[i] = 1;
    dev_data = data;

    // CUDA events for accurate GPU timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    printf("--- Kernel Comparison ---\n\n");

    // =====================================================
    // Benchmark cgwarp kernel (Tiled Partitions)
    // =====================================================
    printf("[1] cgwarp Kernel (Tiled Partitions with thread_block_tile)\n");

    // Warmup run
    cgwarp<<<cgwarpBlocks, cgwarpThreads>>>(id);
    cudaDeviceSynchronize();

    // Timed run
    cudaEventRecord(start);
    cgwarp<<<cgwarpBlocks, cgwarpThreads>>>(id);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    int totalCgwarpThreads = cgwarpBlocks * cgwarpThreads;
    printf("   Blocks: %d, Threads/block: %d\n", cgwarpBlocks, cgwarpThreads);
    printf("   Total threads: %d\n", totalCgwarpThreads);
    printf("   Time: %.3f ms\n", milliseconds);
    printf("   Demonstrates: tiled_partition<32/16/8/4> for sub-warp grouping\n");
    printf("   Thread %d info printed above ^\n\n", id);

    // =====================================================
    // Benchmark reduce5 kernel
    // =====================================================
    printf("[2] reduce5 Kernel (Warp-level reduction with __syncwarp)\n");

    // Warmup run
    reduce5<blockSize><<<numBlocks, blockSize>>>(
        thrust::raw_pointer_cast(dev_sums.data()),
        thrust::raw_pointer_cast(dev_data.data()), n);
    cudaDeviceSynchronize();

    // Timed run
    cudaEventRecord(start);
    reduce5<blockSize><<<numBlocks, blockSize>>>(
        thrust::raw_pointer_cast(dev_sums.data()),
        thrust::raw_pointer_cast(dev_data.data()), n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy partial sums back and compute final result
    thrust::host_vector<int> h_sums = dev_sums;
    long long total_sum = 0;
    for (int i = 0; i < numBlocks; i++) {
        total_sum += h_sums[i];
    }

    float gbytes = (n * sizeof(int)) / (milliseconds * 1e6);
    float gflops = n / (milliseconds * 1e6);

    printf("   Blocks: %d, Threads/block: %d\n", numBlocks, blockSize);
    printf("   Time: %.3f ms\n", milliseconds);
    printf("   Bandwidth: %.2f GB/s\n", gbytes);
    printf("   GFLOPS: %.2f\n", gflops);
    printf("   Result: %lld (expected: %d) %s\n\n",
           total_sum, n, (total_sum == n) ? "✓ PASS" : "✗ FAIL");

    // =====================================================
    // Benchmark coop3D kernel
    // =====================================================
    printf("[3] coop3D Kernel (Cooperative Groups for 3D indexing)\n");

    // Configure 3D grid and block dimensions
    dim3 threadsPerBlock(8, 8, 8);  // 512 threads per block
    dim3 numBlocks3D(
        (nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (ny + threadsPerBlock.y - 1) / threadsPerBlock.y,
        (nz + threadsPerBlock.z - 1) / threadsPerBlock.z
    );

    // Warmup run
    coop3D<<<numBlocks3D, threadsPerBlock>>>(nx, ny, nz, 42);
    cudaDeviceSynchronize();

    // Timed run
    cudaEventRecord(start);
    coop3D<<<numBlocks3D, threadsPerBlock>>>(nx, ny, nz, 42);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    int total3DElements = nx * ny * nz;
    float gbytes3D = (total3DElements * sizeof(int)) / (milliseconds * 1e6);

    printf("   Grid: (%d, %d, %d), Block: (%d, %d, %d)\n",
           numBlocks3D.x, numBlocks3D.y, numBlocks3D.z,
           threadsPerBlock.x, threadsPerBlock.y, threadsPerBlock.z);
    printf("   Total threads: %d\n",
           numBlocks3D.x * numBlocks3D.y * numBlocks3D.z *
           threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z);
    printf("   Time: %.3f ms\n", milliseconds);
    printf("   Theoretical Bandwidth: %.2f GB/s\n\n", gbytes3D);

    // =====================================================
    // Summary
    // =====================================================
    printf("--- Summary ---\n");
    printf("cgwarp:  Uses tiled_partition<N> to create sub-warp groups,\n");
    printf("         enabling fine-grained thread cooperation at 32/16/8/4\n");
    printf("         thread granularity with meta_group_rank/size info.\n\n");
    printf("reduce5: Uses warp-level primitives (__syncwarp) for efficient\n");
    printf("         final reduction within the last warp, avoiding\n");
    printf("         unnecessary __syncthreads() calls.\n\n");
    printf("coop3D:  Uses cooperative_groups for clean 3D indexing,\n");
    printf("         providing portable access to grid/block/thread\n");
    printf("         hierarchy through cg::this_grid() and cg::this_thread_block().\n\n");

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

