//illustrates the use of tiled partitions in 3D cooperative groups
#include <cooperative_groups.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups; // or using

template <int T> __device__ void show_tile(const char *tag, cg::thread_block_tile<T> p){
    int rank = p.thread_rank();
    int size = p.size();
    int mrank = p.meta_group_rank();
    int msize = p.meta_group_size();
    
    printf("%s: rank=%d, size=%d, mrank=%d, msize=%d\n", tag, rank, size, mrank, msize);
}

__global__ void cgwarp(int id){
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    
    auto warp32 = cg::tiled_partition<32>(block); //normal warp
    auto warp16 = cg::tiled_partition<16>(block); // 16 thread warp 
    auto warp8 = cg::tiled_partition< 8>(block);
    auto tile8 = cg::tiled_partition<8>(warp32); // tile of 8 threads in the warp
    auto tile4 = cg::tiled_partition<4>(tile8);
    
    if(grid.thread_rank() == id) {
        printf("warps and subwarps for thread %d:\n",id);
        show_tile<32>("warp32",warp32);
        show_tile<16>("warp16",warp16);
        show_tile< 8>("warp8 ",warp8);
        show_tile< 8>("tile8 ",tile8);
        show_tile< 4>("tile4 ",tile4);
    }
}

int main(int argc, char *argv[]){
    int id = (argc >1) ? atoi(argv[1]) : 12345;
    int blocks = (argc >2) ? atoi(argv[2]) : 28800;
    int threads = (argc >3) ? atoi(argv[3]) : 256;
    
    cgwarp<<<blocks,threads>>>(id);
    cudaDeviceSynchronize();
    return 0;
}