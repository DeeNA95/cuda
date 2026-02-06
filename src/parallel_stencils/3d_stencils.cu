#include "../cx.h"
#include "../cxtimers.h"
#include <cuda_runtime.h>

// converting a kernel from 2d to 3d there are 2 options:
// add a for loop over the z dim in the existing kernel so that the
// one thread processes all points with a fixed x and y stepping though the full
// range of z
//
// add more threads to the kernel launch so that each point in the 3d grid is
// processed by a different thread

// v1
template <typename T>
__global__ void stencil3d_1(cr_Ptr<T> a, r_Ptr<T> b, int nx, int ny, int nz) {
  // nb the values incapsulated by the [] in a lambda are the capture clauses,
  // they denote what external variables the lambda can use within its scope,
  // when paired with a & its a capture by reference so no copy overhead
  auto idx = [&nx, &ny](int z, int y, int x) {
    return (z * ny + y) * nx + x;
  }; // the global id in the 3d grid
     //
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  // int z = blockDim.z * blockIdx.z + threadIdx.z;

  if (x < 1 || y < 1 || x >= nx - 1 || y > ny - 1)
    return; // leave out edges because of stencil algo boundary

  for (int z = 1; z < nz - 1; z++) {
    b[idx(z, y, x)] =
        (T)(1.0 / 6.0) *
        (a[idx(z, y, x + 1)] + a[idx(z, y, x - 1)] + a[idx(z, y + 1, x)] +
         a[idx(z, y - 1, x)] + a[idx(z + 1, y, x)] + a[idx(z - 1, y, x)]);
  };
}

// v2
template <typename T>
__global__ void stencil3d_2(cr_Ptr<T> a, r_Ptr<T> b, int nx, int ny, int nz) {
  auto idx = [&nx, &ny](int z, int y, int x) { return (z * ny + y) * nx + x; };

  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  int z = blockDim.z * blockIdx.z + threadIdx.z;

  if (x < 1 || y < 1 || x >= nx - 1 || y >= ny - 1 || z < 1 || z >= nz - 1)
    return;

  b[idx(z, y, x)] =
      (T)(1.0 / 6.0) *
      (a[idx(z, y, x + 1)] + a[idx(z, y, x - 1)] + a[idx(z, y + 1, x)] +
       a[idx(z, y - 1, x)] + a[idx(z + 1, y, x)] + a[idx(z - 1, y, x)]);
}

int main(int argc, char *argv[]) {
  // 1. Parse command line (e.g., ./stencil 256 256 256 100)
  int nx = (argc > 1) ? atoi(argv[1]) : 128;
  int ny = (argc > 2) ? atoi(argv[2]) : 128;
  int nz = (argc > 3) ? atoi(argv[3]) : 128;
  int iterations = (argc > 4) ? atoi(argv[4]) : 10000;

  // 2. Setup buffers (using your thrustDvec or similar)
  size_t size = (size_t)nx * ny * nz;
  thrustDvec<float> d_in(size), d_out(size);

  // 3. Define execution configuration
  dim3 threads(16, 16, 1); // For v1 (2D grid)
  dim3 blocks((nx + 15) / 16, (ny + 15) / 16, 1);

  dim3 threads_v2(8, 8, 8); // For v2 (3D grid)
  dim3 blocks_v2((nx + 7) / 8, (ny + 7) / 8, (nz + 7) / 8);

  // --- Test v1 ---
  // Start Timer
  cxxtimer::Timer tim;
  tim.start();
  for (int i = 0; i < iterations; ++i) {
    stencil3d_1<float><<<blocks, threads>>>(d_in.data().get(),
                                            d_out.data().get(), nx, ny, nz);
  }
  cudaDeviceSynchronize();
  tim.stop();
  double t1 = tim.count<cxxtimer::ms>();
  printf("V1 time %.3fms\n", t1);
  // Stop Timer & Print Result

  // --- Test v2 ---
  // Reset buffers, Start Timer, Loop v2, Synchronize, Stop Timer
  tim.start();
  for (int i = 0; i < iterations; ++i) {
    stencil3d_2<float><<<blocks_v2, threads_v2>>>(
        d_in.data().get(), d_out.data().get(), nx, ny, nz);
  }
  cudaDeviceSynchronize();
  tim.stop();

  double t2 = tim.count<cxxtimer::ms>();
  printf("V2 time %.3fms\n", t2);

  return 0;
}
