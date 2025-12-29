#include "../cx.h"
#include "../cxtimers.h"
#include <cuda_runtime.h>

__global__ void stencil2d(cr_Ptr<float> a, r_Ptr<float> b, int nx, int ny) {
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

int main(int argc, char *argv[]) {
  int nx = (argc > 1) ? atoi(argv[1]) : 1024;
  int ny = (argc > 2) ? atoi(argv[2]) : 1024;
  int iter_host = (argc > 3) ? atoi(argv[3]) : 1000;
  int iter_gpu = (argc > 4) ? atoi(argv[4]) : 10000;

  int size = nx * ny;

  thrustHvec<float> a(size);
  thrustHvec<float> b(size);
  thrustDvec<float> dev_a(size);
  thrustDvec<float> dev_b(size);
  thrustDvec<float> c(9);

  auto idx = [&nx](int y, int x) { return y * nx + x; };
  for (int y = 0; y < ny; y++)
    a[idx(y, 0)] = a[idx(y, nx - 1)] = 1.0f;

  a[idx(0, 0)] = a[idx(0, nx - 1)] = a[idx(ny - 1, 0)] =
      a[idx(ny - 1, nx - 1)] = 0.5f;
  dev_a = a;
  dev_b = a;

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
  tim.start();
  for (int k = 0; k < iter_gpu / 2; k++) {
    stencil2d<<<blocks, threads>>>(dev_a.data().get(), dev_b.data().get(), nx,
                                   ny);
    stencil2d<<<blocks, threads>>>(dev_b.data().get(), dev_a.data().get(), nx,
                                   ny);
  }
  cudaDeviceSynchronize();
  tim.stop();
  a = dev_b;
  printf("device middle value %.3f\n", a[(ny / 2) * nx + nx / 2]);
  double t2 = tim.count<cxxtimer::ms>();

  double gflops_gpu = (double)(iter_gpu * 4) * (double)size / (t2 * 1e6);
  double speedup = gflops_gpu / gflops_host;

  tim.reset();
  tim.start();
  for (int k = 0; k < iter_gpu / 2; k++) {
    stencil2d_sm<18, 18>
        <<<blocks, threads>>>(dev_a.data().get(), dev_b.data().get(), nx, ny);
    stencil2d_sm<18, 18>
        <<<blocks, threads>>>(dev_b.data().get(), dev_a.data().get(), nx, ny);
  }
  cudaDeviceSynchronize();
  tim.stop();
  double t3 = tim.count<cxxtimer::ms>();
  double gflops_gpu2 = (double)(iter_gpu * 4) * (double)size / (t3 * 1e6);
  double speedup2 = gflops_gpu2 / gflops_host;

  tim.reset();
  tim.start();
  for (int k = 0; k < iter_gpu / 2; k++) {
    stencil2d9pt<<<blocks, threads>>>(dev_a.data().get(), dev_b.data().get(),
                                      nx, ny, c.data().get());
    stencil2d9pt<<<blocks, threads>>>(dev_b.data().get(), dev_a.data().get(),
                                      nx, ny, c.data().get());
  }
  cudaDeviceSynchronize();
  tim.stop();
  double t4 = tim.count<cxxtimer::ms>();
  double gflops_gpu3 = (double)(iter_gpu * 4) * (double)size / (t4 * 1e6);
  double speedup3 = gflops_gpu3 / gflops_host;

  printf("host iter %8d time %9.3fms GFlops %8.3f\n", iter_host, t1,
         gflops_host);
  printf("gpu iter %8d time %9.3fms GFlops %8.3f\n", iter_gpu, t2, gflops_gpu);
  printf("gpu shared iter %8d time %9.3fms GFlops %8.3f\n", iter_gpu, t3,
         gflops_gpu2);
  printf("gpu 9pt iter %8d time %9.3fms GFlops %8.3f\n", iter_gpu, t4,
         gflops_gpu3);
  printf("Speedup: %.3f\n", speedup);
  printf("Speedup shared: %.3f\n", speedup2);
  printf("Speedup 9pt: %.3f\n", speedup3);
  return 0;
}
