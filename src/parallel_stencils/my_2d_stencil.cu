#include "../cx.h"
#include "../cxtimers.h"
#include <cuda_runtime.h>

// i will try to implement the stencil to see if i trully understand it

__global__ void stencil2d(float *a, float *b, int nx, int ny) {
  // args:
  /*
  a: first array
  b: second array with same elements
  nx: number of columns
  ny: number of rows
  reason for the sort of confusing order of nx, ny is that matrices are stored
  as an array in row major order ie it is flattened such that all the values of
  the first row are stored first, then the second row, and so on so to move one
  cell we will just move x+1, but to move to the next row we will need to move
  nx cells which will be a unitary increase in y
  */

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  // printf("x %d y %d\n", x, y);
  // simple stencil using only up, down, left right
  if (x == 0 || y == 0 || x >= nx - 1 || y >= ny - 1)
    return; // if at the edges of the array, update is impossible

  b[x + y * nx] = 0.25f * (a[x - 1 + y * nx] + a[x + 1 + y * nx] +
                           a[x + (y - 1) * nx] + a[x + (y + 1) * nx]);
}

// phase 2 is using shared memory with a halo
__global__ void stencil2d_sm(float *a, float *b, int nx, int ny) {
  // define some shared memory which will be num threads + 2, (1 for the halo
  // and 1 for the buffer for the halo)
  // in this case num threads = 16, consider movement to template to allow for
  // differences in thread num
  __shared__ float s[18][18];

  // current x and y will be the origin plus the current thread value
  int x = (blockDim.x) * blockIdx.x + threadIdx.x;
  int y = (blockDim.y) * blockIdx.y + threadIdx.y;

  // x and y in the shared buffer will just be the current thread value
  int xs = threadIdx.x;
  int ys = threadIdx.y;

  if (x >= 0 && y >= 0 && x <= nx - 1 && y <= ny - 1) {
    // copy global mem to shared mem
    s[ys + 1][xs + 1] = a[x + y * nx];
    // ys first because of row major, +1 for halo so that our data will sit from
    // 1-16 ie the inner 16 with the halo on the outside ie rows and cols 0 and
    // 17
  };
  __syncthreads(); // so all copies to shared memory are created before
                   // continuing

  // also load actual values of the halo region
  if (xs == 0 && x != 0) {
    s[ys + 1][xs] = a[x - 1 + y * nx];
  }
  if (ys == 0 && y != 0) {
    s[ys][xs + 1] = a[x + (y - 1) * nx];
  }

  if (ys + 1 == blockDim.y && y < ny - 1) {
    s[ys + 2][xs + 1] = a[x + (y + 1) * nx];
  }

  if (xs + 1 == blockDim.x && x < nx - 1) {
    s[ys + 1][xs + 2] = a[x + 1 + (y)*nx];
  }
  __syncthreads();

  // if thread in halo zone of the shared mem skip
  if (x < 1 || x >= nx - 1 || y < 1 || y >= ny - 1)
    return;

  // else perform the stencil calculation from shared mem and save to global mem
  // b[x + y*nx] will be at s[ys+1][xs+1]
  b[x + y * nx] = 0.25f * (s[ys + 1][xs + 2] + s[ys + 1][xs] +
                           s[ys + 2][xs + 1] + s[ys][xs + 1]);
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

int main(int argc, char *argv[]) {
  int nx = 1024;
  int ny = 1024;
  int iter = 10000;

  thrustHvec<float> a(nx * ny);
  thrustDvec<float> dev_a(nx * ny);
  thrustDvec<float> dev_b(nx * ny);
  thrustDvec<float> dev_a_sm(nx * ny);
  thrustDvec<float> dev_b_sm(nx * ny);
  thrustDvec<float> dev_a_rt(nx * ny);
  thrustDvec<float> dev_b_rt(nx * ny);

  // make top edge and bottom edge 1 and left and right 0 and corners 0.5

  for (int x = 0; x < nx; x++) {
    a[x] = a[x + (ny - 1) * nx] = 100.0f;
  }

  a[0] = a[nx - 1] = a[(ny - 1) * nx] = a[nx * ny - 1] = 50.0f;

  dev_a = a;
  dev_b = a;
  dim3 threads = {16, 16, 1};
  dim3 blocks = {(nx + threads.x - 1) / threads.x,
                 (ny + threads.y - 1) / threads.y, 1};
  printf("blocks %d %d\n", blocks.x, blocks.y);

  cxxtimer::Timer tim;
  tim.start();
  for (int k = 0; k < iter / 2; k++) {
    stencil2d<<<blocks, threads>>>(dev_a.data().get(), dev_b.data().get(), nx,
                                   ny);
    stencil2d<<<blocks, threads>>>(dev_b.data().get(), dev_a.data().get(), nx,
                                   ny);
  }
  cudaDeviceSynchronize();
  tim.stop();
  a = dev_b;
  printf("device middle value %.8f\n", a[(ny / 2) * nx + nx / 2]);
  double t = tim.count<cxxtimer::ms>();
  double gflops = (double)(iter * 4) * (double)nx * ny / (t * 1e6);
  printf("Time: %f ms\n", t);
  printf("GFlops: %f\n\n\n", gflops);

  tim.reset();
  tim.start();
  for (int k = 0; k < iter / 2; k++) {
    stencil2d_sm<<<blocks, threads>>>(dev_a_sm.data().get(),
                                      dev_b_sm.data().get(), nx, ny);
    stencil2d_sm<<<blocks, threads>>>(dev_b_sm.data().get(),
                                      dev_a_sm.data().get(), nx, ny);
  }
  cudaDeviceSynchronize();
  tim.stop();
  a = dev_b_sm;
  printf("device middle value %.8f\n", a[(ny / 2) * nx + nx / 2]);
  double t2 = tim.count<cxxtimer::ms>();
  double gflops2 = (double)(iter * 4) * (double)nx * ny / (t2 * 1e6);
  printf("Time: %f ms\n", t2);
  printf("GFlops: %f\n", gflops2);

  tim.reset();
  dim3 threads_rt = {16, 4, 1};
  dim3 blocks_rt = {(nx + threads_rt.x - 1) / threads_rt.x, (ny + 16 - 1) / 16,
                    1};

  tim.start();
  for (int k = 0; k < iter / 2; k++) {
    stencil2d_rt<<<blocks_rt, threads_rt>>>(dev_a_rt.data().get(),
                                            dev_b_rt.data().get(), nx, ny);
    stencil2d_rt<<<blocks_rt, threads_rt>>>(dev_b_rt.data().get(),
                                            dev_a_rt.data().get(), nx, ny);
  }
  cudaDeviceSynchronize();
  tim.stop();
  a = dev_b_rt;
  printf("device middle value %.8f\n", a[(ny / 2) * nx + nx / 2]);
  double t3 = tim.count<cxxtimer::ms>();
  double gflops3 = (double)(iter * 4) * (double)nx * ny / (t3 * 1e6);
  printf("Time: %f ms\n", t3);
  printf("GFlops: %f\n", gflops3);

  return 0;
}
