//illustrates the use of tiled partitions in 3D cooperative groups
#include <cooperative_groups.h>
#include <iostream>
#include <random>
#include <chrono>
#include <cuda_runtime.h>

namespace cg = cooperative_groups; // or using

template <int T> __device__ void show_tile(const char *tag, cg::thread_block_tile<T> p){
    int rank = p.thread_rank();
    int size = p.size();
    int mrank = p.meta_group_rank();
    int msize = p.meta_group_size();
}
