#include "cx.h"
#include "cxtimers.h"
#include <random>

__global__ void reduce0(double *x, int m) {
  int tid =
      threadIdx.x + blockIdx.x * blockDim.x; // global thread index for 1d grid
  x[tid] += x[tid + m];                      //
}

__global__ void reduce1(double *x,
                        int m) { // NB only works when threads * blocks < m
  int tid =
      threadIdx.x + blockIdx.x * blockDim.x; // global thread index for 1d grid
  double tsum = 0.0;
  for (int k = tid; k < m;
       k += blockDim.x *
            gridDim.x) { // stride of total threads per block for efficiency
    tsum += x[k];
  }
  x[tid] = tsum;
}

__global__ void reduce2(double *y, double *x,
                        int m) { // why as output array and x as input array
  // using shared memory
  extern __shared__ float tsum[]; // array of floats in shared memory

  int id = threadIdx.x;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  int stride = gridDim.x * blockDim.x;

  tsum[id] = 0.0f;

  for (int k = tid; k < m; k += stride)
    tsum[id] += x[k];
  __syncthreads();

  // pow 2 reduction loop
  for (int k = blockDim.x / 2; k > 0; k /= 2) {
    if (id < k) {
      tsum[id] += tsum[id + k];
      __syncthreads();
    }
  }
  if (id == 0) {
    y[blockIdx.x] = tsum[0];
  }
}

__global__ void reduce3(double *y, double *x, int m) {
  // using shared memory
  extern __shared__ float tsum[]; // array of floats in shared memory

  int id = threadIdx.x;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  int stride = gridDim.x * blockDim.x;

  tsum[id] = 0.0f;

  for (int k = tid; k < m; k += stride)
    tsum[id] += x[k];
  __syncthreads();

  int blocks2 = 288; // cx::pow2ceil( blockDim.x );

  for (int k = blocks2 / 2; k > 0; k /= 2) {
    if (id < k && k + id < blockDim.x)
      tsum[id] += tsum[id + k];
    __syncthreads();
  }

  if (id == 0)
    y[blockIdx.x] = tsum[0];
}

__global__ void reduce4(double *y, double *x, int m) {
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
  // only warp 0 elements needede
  if (id < 16)
    tsum[id] += tsum[id + 16];
  __syncwarp();
  if (id < 8)
    tsum[id] += tsum[id + 8];
  __syncwarp();
  if (id < 4)
    tsum[id] += tsum[id + 4];
  __syncwarp();
  if (id < 2)
    tsum[id] += tsum[id + 2];
  __syncwarp();
  if (id < 1)
    tsum[id] += tsum[id + 1];
  __syncwarp();

  if (id == 0)
    y[blockIdx.x] = tsum[0] + tsum[1];
}

int main(int argc, char **argv) {
  int N =
      (argc > 1) ? atoi(argv[1]) : 1 << 24; // shifts bit 24 to the left ie 2^24
  N = 1 << 24;
  int threads = (argc > 2) ? atoi(argv[2]) : 256;
  int blocks = (N + threads - 1) / threads < N / threads
                   ? (N + threads - 1) / threads
                   : 256; // ceiling division

  thrust::host_vector<double> h_x(N);
  thrust::device_vector<double> d_x(N);
  thrust::device_vector<double> d_y(N);

  thrust::device_vector<double> d_r2(N);
  thrust::device_vector<double> d_r2y(blocks);
  thrust::device_vector<double> xr3(N);
  thrust::device_vector<double> yr3(blocks);
  thrust::device_vector<double> xr4(N);
  thrust::device_vector<double> yr4(blocks);

  std::default_random_engine gen(
      1); // random number generator with fixed seed for reproducibility
  std::uniform_real_distribution<double> fran(0.0, 1.0);

  for (int k = 0; k < N; k++) {
    h_x[k] = fran(gen);
  }
  d_x = h_x;  // copy data to device
  d_y = h_x;  // copy data to device
  d_r2 = h_x; // copy data to device
  xr3 = h_x;
  xr4 = h_x;
  // d_r2y = h_x;

  auto start = std::chrono::high_resolution_clock::now();

  double host_sum = 0.0;
  for (int k = 0; k < N; k++) {
    host_sum += h_x[k];
  } // reduce on host

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;

  // printf("sum of %d random numbers: host %.1f %.3f ms\n", N, host_sum,
  // elapsed.count() * 1000);

  auto start2 = std::chrono::high_resolution_clock::now();

  for (int m = N / 2; m >= 1; m /= 2) {
    int threads = std::min(256, m);
    int nblocks = std::max(m / 256, 1);
    reduce0<<<nblocks, threads>>>(d_x.data().get(), m);

  } // reduce on device
  cudaDeviceSynchronize();
  double gpu_sum = d_x[0]; // copy result back to host

  auto end2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed2 = end2 - start2;
  printf("REDUCE0\n");
  printf("sum of %d random numbers: host %.3f %.3f ms, GPU %.3f %.3f ms\n", N,
         host_sum, elapsed.count() * 1000, gpu_sum, elapsed2.count() * 1000);
  printf("Speed Up: %.2fx\n", elapsed.count() / elapsed2.count());

  // reduce1
  auto start3 = std::chrono::high_resolution_clock::now();
  reduce1<<<blocks, threads>>>(d_y.data().get(), N);
  reduce1<<<1, threads>>>(d_y.data().get(), blocks * threads);
  reduce1<<<1, 1>>>(d_y.data().get(), threads);
  cudaDeviceSynchronize();
  double gpu_sum1 = d_y[0];
  auto end3 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed3 = end3 - start3;

  printf("REDUCE1\n");
  printf(
      "sum of %d random numbers: REDUCE0 %.3f %.3f ms, REDUCE1 %.3f %.3f ms\n",
      N, gpu_sum, elapsed2.count() * 1000, gpu_sum1, elapsed3.count() * 1000);
  printf("Speed Up: %.2fx\n", elapsed2.count() / elapsed3.count());

  // reduce2
  auto start4 = std::chrono::high_resolution_clock::now();
  reduce2<<<blocks, threads, threads * sizeof(float)>>>(d_r2y.data().get(),
                                                        d_r2.data().get(), N);
  reduce2<<<1, blocks, blocks * sizeof(float)>>>(d_r2.data().get(),
                                                 d_r2y.data().get(), blocks);

  cudaDeviceSynchronize();
  double gpu_sum2 = d_r2[0];
  auto end4 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed4 = end4 - start4;

  printf("REDUCE2\n");
  printf(
      "sum of %d random numbers: REDUCE1 %.3f %.3f ms, REDUCE2 %.3f %.3f ms\n",
      N, gpu_sum1, elapsed3.count() * 1000, gpu_sum2, elapsed4.count() * 1000);
  printf("Speed Up: %.2fx\n", elapsed3.count() / elapsed4.count());

  // reduc3
  auto start5 = std::chrono::high_resolution_clock::now();
  reduce3<<<blocks, threads, threads * sizeof(float)>>>(yr3.data().get(),
                                                        xr3.data().get(), N);
  reduce3<<<1, blocks, blocks * sizeof(float)>>>(xr3.data().get(),
                                                 yr3.data().get(), blocks);

  cudaDeviceSynchronize();
  double gpu_sum3 = xr3[0];
  auto end5 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed5 = end5 - start5;

  printf("REDUCE3\n");
  printf(
      "sum of %d random numbers: REDUCE2 %.3f %.3f ms, REDUCE3 %.3f %.3f ms\n",
      N, gpu_sum2, elapsed4.count() * 1000, gpu_sum3, elapsed5.count() * 1000);
  printf("Speed Up: %.2fx\n", elapsed4.count() / elapsed5.count());

  // reduce4
  auto start6 = std::chrono::high_resolution_clock::now();
  reduce4<<<blocks, threads, threads * sizeof(float)>>>(yr4.data().get(),
                                                        xr4.data().get(), N);
  reduce4<<<1, blocks, blocks * sizeof(float)>>>(xr4.data().get(),
                                                 yr4.data().get(), blocks);

  cudaDeviceSynchronize();
  double gpu_sum4 = xr4[0];
  auto end6 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed6 = end6 - start6;

  printf("REDUCE4\n");
  printf(
      "sum of %d random numbers: REDUCE3 %.3f %.3f ms, REDUCE4 %.3f %.3f ms\n",
      N, gpu_sum3, elapsed5.count() * 1000, gpu_sum4, elapsed6.count() * 1000);
  printf("Speed Up: %.2fx\n", elapsed5.count() / elapsed6.count());

  return 0;
}
