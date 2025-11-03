#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include "cuda_runtime.h"
#include "thrust/device_vector.h"
#include "device_launch_parameters.h"

__device__ int a[256][512][512]; // a 3d array in device memory NB declared as z,y,x order
__device__ float b[256][512][512]; // file scope

__global__ void grid3D(int nx, int ny, int nz, int id) // array dims and id of thread to print info
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;  // find x using the index of the block * nthreads per block + thread index within block
    int y = blockIdx.y * blockDim.y + threadIdx.y;  // same for y
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= nx || y >= ny || z >= nz) return;  // range check

    int array_size = nx * ny * nz;

    //
    int block_size = blockDim.x * blockDim.y * blockDim.z; // number of threads per block
    int grid_size = gridDim.x * gridDim.y * gridDim.z; // number of blocks in grid

    int total_threads = block_size * grid_size;

    int thread_rank_in_block = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x; // rank of thread in block = x + (y * blockDim.x) + (z * blockDim.x * blockDim.y)
    int block_rank_in_grid = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x; // rank of block in grid = x + (y * gridDim.x) + (z * gridDim.x * gridDim.y)
    int thread_rank_in_grid = block_rank_in_grid * block_size + thread_rank_in_block;

    // do some work here
    a[z][y][x] = thread_rank_in_grid;
    b[z][y][x] = sqrtf((float)a[z][y][x]);

    if (thread_rank_in_grid == id) {
        printf("array size %3d x %3d x %3d = %d\n",
               nx, ny, nz, array_size);
        printf("thread block %3d x %3d x %3d = %d\n",
               blockDim.x, blockDim.y, blockDim.z, block_size);
        printf("thread grid %3d x %3d x %3d = %d\n",
               gridDim.x, gridDim.y, gridDim.z, grid_size);
        printf("total number of threads in grid %d\n",
               total_threads);
        printf("a[%d][%d][%d] = %i and b[%d][%d][%d] = %f\n",
               z, y, x, a[z][y][x], z, y, x, b[z][y][x]);
        printf("for thread with 3D-rank: %d \n1D-rank: %d block rank in grid %d\n",
               thread_rank_in_grid, thread_rank_in_block, block_rank_in_grid);
    }
}

int main(int argc, char *argv[])
{
    int id = (argc > 1) ? atoi(argv[1]) : 12345;
    dim3 thread3d(32, 8, 2);  // 32*8*2 = 512
    dim3 block3d(16, 64, 128);  // 16*64*128 = 131072
    grid3D<<<block3d, thread3d>>>(512, 512, 256, id);
    cudaDeviceSynchronize();
    return 0;
}
