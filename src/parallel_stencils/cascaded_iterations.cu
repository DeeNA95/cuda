#include "../cx.h"
#include "../cxtimers.h"
#include "stencils.h"
#include <cooperative_groups.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

template <typename T>
__global__ void Zoomfrom(r_Ptr<T> a, r_Ptr<T> b, cr_Ptr<T> aold, int nx,
                         int ny) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  if (x >= nx || y >= ny)
    return;

  int mx = nx / 2;
  auto idx = [&nx](int y, int x) { return y * nx + x; };
  auto mdx = [&mx](int y, int x) { return y * mx + x; };

  if (x > 0 && x < nx - 1 && y < ny - 1)
    a[idx(y, x)] = aold[mdx(y / 2, x / 2)]; // interior
  // top
  else if (y == 0 && x > 0 && x < nx - 1)
    a[idx(y, x)] = T(0); // cast 0 to type T
  else if (y == ny - 1 && x > 0 && x < nx - 1)
    a[idx(y, x)] = T(0); // bottom

  // sides
  else if (x == 0 && y > 0 && y < ny - 1)
    a[idx(y, x)] = T(1);
  else if (x == nx - 1 && y > 0 && y < ny - 1)
    a[idx(y, x)] = T(1);

  // corners
  else if (x == 0 && y == 0)
    a[idx(y, x)] = T(0.5);
  else if (x == 0 && y == ny - 1)
    a[idx(y, x)] = T(0.5);
  else if (x == nx - 1 && y == 0)
    a[idx(y, x)] = T(0.5);
  else if (x == nx - 1 && y == ny - 1)
    a[idx(y, x)] = T(0.5);

  b[idx(y, x)] = a[idx(y, x)];
}

template <typename T> int cascade(int nx, int ny, int iter) {
  int nx_start = std::min(nx, 32);
  int size = nx_start * nx_start;

  // init buffers
  thrustDvec<T> dev_a(size);
  thrustDvec<T> dev_b(size);
  thrustDvec<T> dev_aold(size);

  cxxtimer::Timer tim;
  tim.start();

  for (int mx = nx_start; mx < nx; mx *= 2) {
    int my = mx;
    dim3 threads(16, 16, 1);
    dim3 blocks((mx + 15) / 16, (my + 15) / 16, 1);
    int size = mx * my;

    if (mx > nx_start) {
      dev_a.resize(size);
      dev_b.resize(size);
    }

    Zoomfrom<T><<<blocks, threads>>>(dev_a.data().get(), dev_b.data().get(),
                                     dev_aold.data().get(), mx, my);
    int check = (mx == nx) ? 5000 : 2500;
    double diff_cut = (mx == nx) ? 1.0e-14 : 1.0e-09;

    for (int k = 0; k < iter / 2; k++) {
      stencil2d<T>
          <<<blocks, threads>>>(dev_a.data().get(), dev_b.data().get(), mx, my

          );
      stencil2d<T>
          <<<blocks, threads>>>(dev_b.data().get(), dev_a.data().get(), mx, my);

      if (k > 0 && k % check == 0) {
        cudaDeviceSynchronize();
        double diff =
            array_diff_max<T>(dev_a.data().get(), dev_b.data().get(), mx, my);
        if (diff < diff_cut)
          break;
      };
    }
    cudaDeviceSynchronize();

    if (mx > nx_start)
      dev_aold.resize(size);
    if (mx < nx)
      dev_aold = dev_a;
  }

  tim.stop();
  double t1 = tim.count<cxxtimer::ms>();

  auto a = dev_a;

  char name[256];
  sprintf(name, "cascade%d_%d.raw", nx, (int)sizeof(T));

  // cx::write_raw(name, a.data(), nx*nx);
  printf("cascade time %.3f ms\n", t1);
  return 0;
}

int main(int argc, char *argv[]) {

  int type = (argc > 1) ? atoi(argv[1]) : 0;
  int nx = (argc > 1) ? atoi(argv[2]) : 1024;
  int ny = (argc > 2) ? atoi(argv[3]) : 1024;
  int iter_gpu = (argc > 3) ? atoi(argv[3]) : 10000;

  if (type == 1)
    cascade<double>(ny, nx, iter_gpu);
  else
    cascade<float>(ny, nx, iter_gpu);

  std::atexit([] { cudaDeviceReset(); });
  return 0;
}
