#include <cooperative_groups.h>
#include <iostream>
#include <random>
#include <chrono>
#include <cuda_runtime.h>

namespace cg = cooperative_groups; // or using

/*
in cgs, the familiar thread, block , warp can be accessed via
cg::this_thread_block()
cg::this_grid()
cg::tiled_partition<32>(block)

explicitly with defining types
cg::thread_block tb = cg::this_thread_block();
cg::grid g = cg::this_grid();
cg::thread_block_tile<32> warp = cg::tiled_partition<32>(tb);

or using auto
auto tb = cg::this_thread_block();
auto g = cg::this_grid();
auto warp = cg::tiled_partition<32>(tb);
*/

__device__ int a[256][512][512];
__device__ float b[256][512][512];

//this kernel uses 3D cooperative groups to acces the 3d grid
__global__ void coop3D(int nx,int ny,int nz,int id){

    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    int x = block.thread_index().x + block.group_dim().x * block.group_index().x;
    int y = block.thread_index().y + block.group_dim().y * block.group_index().y;
    int z = block.thread_index().z + block.group_dim().z * block.group_index().z;

    if(x >=nx || y >=ny || z >=nz) return; //if not in range return

    int array_size = nx*ny*nz;

    //threads in one block
    int block_size = block.size();

    //blocks in grid
    int grid_size = grid.size()/block_size;

    //threads in whole grid
    int total_threads = grid.size();

    int thread_rank_in_block = block.thread_rank();

    int block_rank_in_grid = grid.thread_rank()/block_size;

    int thread_rank_in_grid = grid.thread_rank();

    if (thread_rank_in_grid == id) {
    printf("array size %3d x %3d x %3d = %d\n", nx, ny, nz, array_size);
    printf("thread block %3d x %3d x %3d = %d\n", blockDim.x, blockDim.y,
           blockDim.z, block_size);
    printf("thread grid %3d x %3d x %3d = %d\n", gridDim.x, gridDim.y,
           gridDim.z, grid_size);
    printf("total number of threads in grid %d\n", total_threads);
    printf("a[%d][%d][%d] = %i and b[%d][%d][%d] = %f\n", z, y, x, a[z][y][x],
           z, y, x, b[z][y][x]);
    printf("for thread with 3D-rank: %d \n1D-rank: %d block rank in grid %d\n",
           thread_rank_in_grid, thread_rank_in_block, block_rank_in_grid);
  }
}


int main(int argc, char *argv[]) {
  int id = (argc > 1) ? atoi(argv[1]) : 12345;
  dim3 thread3d(32, 8, 2);   // 32*8*2 = 512
  dim3 block3d(16, 64, 128); // 16*64*128 = 131072
  coop3D<<<block3d, thread3d>>>(512, 512, 256, id);
  cudaDeviceSynchronize();
  return 0;
}
