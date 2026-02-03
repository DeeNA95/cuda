#include "../cx.h"
#include "../cxtimers.h"
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <istream>
#include "stencils.h"

namespace cg = cooperative_groups;

template <typename T>
__global__ void stencil2d(cr_Ptr<T> a, r_Ptr<T> b, int nx, int ny) {
  auto idx = [&nx](int y, int x) { return y * nx + x; };
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < 1 || y < 1 || x >= nx - 1 || y >= ny - 1)
    return;

  // simple von neumann stencil neighbours
  // consider a matrix of nx rows and ny columns
  // so x is the row and y is the column
  b[idx(y, x)] = 0.25f * (a[idx(y, x + 1)]   // one cell above
                          + a[idx(y, x - 1)] // one cell below
                          + a[idx(y + 1, x)] // one cell to the right
                          + a[idx(y - 1, x)] // one cell to the left
                         );
  // this effective all contibute to the middle cell, and the mutiply by 0.25 is
  // just arbitrary
}

template <int Nx, int Ny>
__global__ void stencil2d_sm(cr_Ptr<float> a, r_Ptr<float> b, int nx, int ny) {

  __shared__ float s[Nx][Ny];

  auto idx = [&nx](int y, int x) { return y * nx + x; };

  // x and y origin
  int x0 = (blockDim.x - 2) * blockIdx.x;
  int y0 = (blockDim.y - 2) * blockIdx.y;

  int xa = x0 + threadIdx.x;
  int ya = y0 + threadIdx.y;

  int xs = threadIdx.x;
  int ys = threadIdx.y;

  if (xa >= nx || ya >= ny)
    return;

  s[ys][xs] = a[idx(ya, xa)];
  __syncthreads();

  if (xa < 1 || ya < 1 || xa >= nx - 1 || ya >= ny - 1)
    return;

  if (xs < 1 || ys < 1 || xs >= nx - 1 || ys >= ny - 1)
    return;

  b[idx(ya, xa)] =
      0.25f * (s[ys][xs + 1] + s[ys][xs - 1] + s[ys + 1][xs] + s[ys - 1][xs]);
}

__global__ void stencil2d9pt(cr_Ptr<float> a, r_Ptr<float> b, int nx, int ny,
                             cr_Ptr<float> c) {
  auto idx = [&nx](int y, int x) { return y * nx + x; };

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < 1 || y < 1 || x >= nx - 1 || y >= ny - 1)
    return;

  // moore's stencil neighbours
  // uses diagonals in addition to von neumann
  b[idx(y, x)] = c[0] * a[idx(y - 1, x - 1)]    // top left
                 + c[1] * a[idx(y - 1, x)]      // top
                 + c[2] * a[idx(y - 1, x + 1)]  // top right
                 + c[3] * a[idx(y, x - 1)]      // left
                 + c[4] * a[idx(y, x)]          // center
                 + c[5] * a[idx(y, x + 1)]      // right
                 + c[6] * a[idx(y + 1, x - 1)]  // bottom left
                 + c[7] * a[idx(y + 1, x)]      // bottom
                 + c[8] * a[idx(y + 1, x + 1)]; // bottom right
}

int stencil2d_host(cr_Ptr<float> a, r_Ptr<float> b, int nx, int ny) {
  auto idx = [&nx](int y, int x) { return y * nx + x; };
  for (int y = 1; y < ny - 1; y++) {
    for (int x = 1; x < nx - 1; x++) {
      b[idx(y, x)] = 0.25f * (a[idx(y, x + 1)] + a[idx(y, x - 1)] +
                              a[idx(y - 1, x)] + a[idx(y + 1, x)]);
    }
  }
  return 0;
}

// phase3 register tiling
__global__ void stencil2d_rt(float *a, float *b, int nx, int ny) {
  __shared__ float s[18][18];
  const int tile_h = 16;

  // current x and y will be the origin plus the current thread value
  int x = (blockDim.x) * blockIdx.x + threadIdx.x;
  int y = (blockDim.y) * blockIdx.y + threadIdx.y;

  // x and y in the shared buffer will just be the current thread value
  int xs = threadIdx.x;
  int ys = threadIdx.y;

  int block_start_y = blockIdx.y * tile_h;

  for (int i = ys; i < 16; i += blockDim.y) {
    int global_y = block_start_y + i - 1;
    if (global_y >= 0 && global_y < ny) {
      // Load central part
      s[i][xs + 1] = a[x + global_y * nx];

      // Load left halo
      if (xs == 0) {
        if (x > 0)
          s[i][0] = a[(x - 1) + global_y * nx];
        // else s[i][0] = 0.0f; // Optional: Handle image boundary
      }

      // Load right halo
      if (xs == blockDim.x - 1) {
        if (x < nx - 1)
          s[i][17] = a[(x + 1) + global_y * nx];
        // else s[i][17] = 0.0f; // Optional: Handle image boundary
      }
    }
  }

  __syncthreads(); // so all copies to shared memory are created before
                   // continuing

  // these will be reused
  int y_offset = ys * 4;
  float top = s[y_offset][xs + 1];
  float center = s[y_offset + 1][xs + 1];
  // float bottom = s[ys+2][xs+1];

  for (int k = 0; k < 4; k++) {

    float bottom = s[y_offset + k + 2][xs + 1];
    // declare top, right, left since we didnt cache
    float left = s[y_offset + k + 1][xs];
    float right = s[y_offset + k + 1][xs + 2];

    // stencil calc to global mem
    int global_write_row = block_start_y + y_offset + k;
    if (global_write_row < ny) {
      b[x + global_write_row * nx] = 0.25f * (left + right + top + bottom);
    }

    // slide for next iter
    top = center;
    center = bottom;
  }
}

// now there are 1 main way to check for stencil convergence
// - compare arrays a & b elementwise
// - if max absolute difference is below a threshold, stop
// NB situations exist in the limit where a and b will just swap values on
// iteration this next kernel tests that

template <typename T>
__global__ void reduce_maxdiff(r_Ptr<T> smax, cr_Ptr<T> a, cr_Ptr<T> b, int n) {
  // based on reduce 6 and uses fixed blocksize 256
  // finds the max abs between arrays a and b using reduction
  auto grid = cg::this_grid();
  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<32>(block);

  // shared mem for 256 fixed block
  __shared__ T s[256];
  int id = block.thread_rank();

  // assign value to id from global mem
  // in this case casting 0 to type t
  s[id] = (T)0;

  for (int tid = grid.thread_rank(); tid < n; tid += grid.size()) {
    // first pass
    if (b != nullptr) // in the first pass , max btn 0 and the absolute diff
      s[id] = fmaxf(s[id], fabs(a[tid] - b[tid]));
    // second pass
    else // in the second pass, max btn abs difference and a current value
      s[id] = fmaxf(s[id], a[tid]);
  } 
  // so now s[id] is the max absolute difference between a and b at that index

  block.sync();
  // reduction algo here folds comparing the max value and the higher move on
  if (id < 128)
    fmaxf(s[id], s[id + 128]);
  block.sync();
  if (id < 64)
    fmaxf(s[id], s[id + 64]);
  block.sync();

  if (warp.meta_group_rank() == 0) { // when in last warp
    s[id] = fmaxf(s[id], s[id + 32]);
    warp.sync();

    s[id] = fmaxf(s[id], warp.shfl_down(s[id], 16));
    s[id] = fmaxf(s[id], warp.shfl_down(s[id], 8));
    s[id] = fmaxf(s[id], warp.shfl_down(s[id], 4));
    s[id] = fmaxf(s[id], warp.shfl_down(s[id], 2));
    s[id] = fmaxf(s[id], warp.shfl_down(s[id], 1));

    if (id == 0)
      smax[blockIdx.x] = s[0];
  }
}

template <typename T>
T array_diff_max(cr_Ptr<T> a, cr_Ptr<T> b, int nx, int ny) {
  // returns the abs max diff between arrays a and b
  thrustDvec<T> c(256); // 256 because of fixed blocksize and dim
  thrustDvec<T> d(1);
  reduce_maxdiff<T><<<256, 256>>>(c.data().get(), a, b, nx * ny);
  reduce_maxdiff<T><<<1, 256>>>(d.data().get(), c.data().get(), nullptr, 256);
  cudaDeviceSynchronize();
  return d[0];
}

int mai(int argc, char *argv[]) {
  int nx = (argc > 1) ? atoi(argv[1]) : 1024;
  int ny = (argc > 2) ? atoi(argv[2]) : 1024;
  int iter_host = (argc > 3) ? atoi(argv[3]) : 1000;
  int iter_gpu = (argc > 4) ? atoi(argv[4]) : 10000;

  int size = nx * ny;

  thrustHvec<float> a(size);
  thrustHvec<float> b(size);
  thrustDvec<float> dev_a(size);
  thrustDvec<float> dev_b(size);
  thrustDvec<float> dev_a_sm(size);
  thrustDvec<float> dev_b_sm(size);
  thrustDvec<float> dev_a_9pt(size);
  thrustDvec<float> dev_b_9pt(size);
  thrustDvec<float> dev_a_rt(size);
  thrustDvec<float> dev_b_rt(size);
  thrustDvec<float> c(9);

  auto idx = [&nx](int y, int x) { return y * nx + x; };
  for (int y = 0; y < ny; y++)
    a[idx(y, 0)] = a[idx(y, nx - 1)] = 1.0f;

  a[idx(0, 0)] = a[idx(0, nx - 1)] = a[idx(ny - 1, 0)] =
      a[idx(ny - 1, nx - 1)] = 0.5f;
  dev_a = a;
  dev_b = a;
  dev_a_sm = a;
  dev_b_sm = a;
  dev_a_9pt = a;
  dev_b_9pt = a;
  dev_a_rt = a;
  dev_b_rt = a;

  cxxtimer::Timer tim;
  tim.start();

  for (int k = 0; k < iter_host / 2; k++) {
    stencil2d_host(a.data(), b.data(), nx, ny);
    stencil2d_host(b.data(), a.data(), nx, ny);
  }
  tim.stop();
  double t1 = tim.count<cxxtimer::ms>();
  double gflops_host = (double)(iter_host * 4) * (double)size / (t1 * 1e6);

  dim3 threads = {16, 16, 1};
  dim3 blocks = {(nx + threads.x - 1) / threads.x,
                 (ny + threads.y - 1) / threads.y, 1};

  tim.reset();

  // base
  tim.start();
  for (int k = 0; k < iter_gpu / 2; k++) {
    stencil2d<<<blocks, threads>>>(dev_a.data().get(), dev_b.data().get(), nx,
                                   ny); // a->b
    stencil2d<<<blocks, threads>>>(dev_b.data().get(), dev_a.data().get(), nx,
                                   ny); // b->a
    if (k > 0 && k % 5000 == 0) {
      cudaDeviceSynchronize();
      float diff =
          array_diff_max<float>(dev_a.data().get(), dev_b.data().get(), nx, ny);
      printf("iter %d maxdiff %f\n", k, diff);
      if (diff < 1e-6) {
        printf("converged at iter %d\n", k);
        break;
      }
    }
  }
  cudaDeviceSynchronize();
  tim.stop();
  a = dev_b;
  printf("device middle value %.3f\n", a[(ny / 2) * nx + nx / 2]);
  double t2 = tim.count<cxxtimer::ms>();

  double gflops_gpu = (double)(iter_gpu * 4) * (double)size / (t2 * 1e6);
  double speedup = gflops_gpu / gflops_host;

  tim.reset();
  // shared mem
  tim.start();
  for (int k = 0; k < iter_gpu / 2; k++) {
    stencil2d_sm<18, 18><<<blocks, threads>>>(dev_a_sm.data().get(),
                                              dev_b_sm.data().get(), nx, ny);
    stencil2d_sm<18, 18><<<blocks, threads>>>(dev_b_sm.data().get(),
                                              dev_a_sm.data().get(), nx, ny);
    if (k > 0 && k % 5000 == 0) {
      cudaDeviceSynchronize();
      float diff = array_diff_max<float>(dev_a_sm.data().get(),
                                         dev_b_sm.data().get(), nx, ny);
      printf("iter %d maxdiff %f\n", k, diff);
      if (diff < 1e-6) {
        printf("converged at iter %d\n", k);
        break;
      }
    }
  }
  cudaDeviceSynchronize();
  tim.stop();
  double t3 = tim.count<cxxtimer::ms>();
  double gflops_gpu2 = (double)(iter_gpu * 4) * (double)size / (t3 * 1e6);
  double speedup2 = gflops_gpu2 / gflops_host;

  tim.reset();
  tim.start();
  // 9pt
  for (int k = 0; k < iter_gpu / 2; k++) {
    stencil2d9pt<<<blocks, threads>>>(
        dev_a_9pt.data().get(), dev_b_9pt.data().get(), nx, ny, c.data().get());
    stencil2d9pt<<<blocks, threads>>>(
        dev_b_9pt.data().get(), dev_a_9pt.data().get(), nx, ny, c.data().get());
    if (k > 0 && k % 5000 == 0) {
      cudaDeviceSynchronize();
      float diff = array_diff_max<float>(dev_a_9pt.data().get(),
                                         dev_b_9pt.data().get(), nx, ny);
      printf("iter %d maxdiff %f\n", k, diff);
      if (diff < 1e-6) {
        printf("converged at iter %d\n", k);
        break;
      }
    }
  }
  cudaDeviceSynchronize();
  tim.stop();
  double t4 = tim.count<cxxtimer::ms>();
  double gflops_gpu3 = (double)(iter_gpu * 4) * (double)size / (t4 * 1e6);
  double speedup3 = gflops_gpu3 / gflops_host;


  tim.reset();
  dim3 threads_rt = {16, 4, 1};
  dim3 blocks_rt = {(nx + threads_rt.x - 1) / threads_rt.x, (ny + 16 - 1) / 16,
                    1};

  tim.start();
  for (int k = 0; k < iter_gpu / 2; k++) {
    stencil2d_rt<<<blocks_rt, threads_rt>>>(dev_a_rt.data().get(),
                                            dev_b_rt.data().get(), nx, ny);
    stencil2d_rt<<<blocks_rt, threads_rt>>>(dev_b_rt.data().get(),
                                            dev_a_rt.data().get(), nx, ny);
    if (k > 0 && k % 5000 == 0) {
      cudaDeviceSynchronize();
      float diff = array_diff_max<float>(dev_a_rt.data().get(),
                                         dev_b_rt.data().get(), nx, ny);
      printf("iter %d maxdiff %f\n", k, diff);
      if (diff < 1e-6) {
        printf("converged at iter %d\n", k);
        break;
      }
    }
  }
  cudaDeviceSynchronize();
  tim.stop();
  a = dev_b_rt;
  printf("device middle value %.8f\n", a[(ny / 2) * nx + nx / 2]);
  double t5 = tim.count<cxxtimer::ms>();
  double gflops_gpu4 = (double)(iter_gpu * 4) * (double)size / (t5 * 1e6);
  double speedup4 = gflops_gpu4 / gflops_host;


  printf("host iter %8d time %9.3fms GFlops %8.3f\n", iter_host, t1,
         gflops_host);
  printf("gpu iter %8d time %9.3fms GFlops %8.3f\n", iter_gpu, t2, gflops_gpu);
  printf("gpu shared iter %8d time %9.3fms GFlops %8.3f\n", iter_gpu, t3,
         gflops_gpu2);
  printf("gpu 9pt iter %8d time %9.3fms GFlops %8.3f\n", iter_gpu, t4,
         gflops_gpu3);
  printf("gpu rt iter %8d time %9.3fms GFlops %8.3f\n", iter_gpu, t5,
         gflops_gpu4);
  printf("Speedup: %.3f\n", speedup);
  printf("Speedup shared: %.3f\n", speedup2);
  printf("Speedup 9pt: %.3f\n", speedup3);
  printf("Speedup rt: %.3f\n", speedup4);

  return 0;
}

//definitions for import to cascade_iterations
template __global__ void stencil2d<float>(cr_Ptr<float>, r_Ptr<float>, int, int);
template __global__ void stencil2d<double>(cr_Ptr<double>, r_Ptr<double>, int, int);
template double array_diff_max<double>(cr_Ptr<double> , cr_Ptr<double>, int , int );
