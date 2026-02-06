// image filters using stencils
//
#include "../../cx.h"
#include "cuda_runtime.h"
#define STB_IMAGE_IMPLEMENTATION
#include "../../../vendor/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../../vendor/stb_image_write.h"
#include <iostream>
#include <string.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

// simplw inefficient
__global__ void filter9PT(cr_Ptr<uchar> a, r_Ptr<uchar> b, int nx, int ny,
                          cr_Ptr<float> c) {
  /*
   * a is the image greyscale
   * b is the buffer for the filter calcs
   * nx ny are image dims
   * c is the 9 point filter coeffs
   * */
  // uchar is an unsigned int from 0-255, char is from -127 -128
  // becase c++ stores chars as their ascii values
  auto idx = [&nx](int y, int x) { return y * nx + x; };

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < 0 || y < 0 || x >= nx || y >= ny)
    return;

  int xl = max(0, x - 1);
  int yl = max(0, y - 1);
  int xh = min(nx - 1, x + 1);
  int yh = min(ny - 1, y + 1);

  float v = c[0] * a[idx(yl, xl)] + c[1] * a[idx(yl, x)] +
            c[2] * a[idx(yl, xh)] + c[3] * a[idx(y, xl)] + c[4] * a[idx(y, x)] +
            c[5] * a[idx(y, xh)] + c[6] * a[idx(yh, xl)] +
            c[7] * a[idx(yh, x)] + c[8] * a[idx(yh, xh)];

  uint f = (uint)(v + 0.5f);
  b[idx(y, x)] = (uchar)min(255, max(0, (int)f));
}

// use constant memory

__constant__ float fc[9];
__global__ void filter9PT_fc(cr_Ptr<uchar> a, r_Ptr<uchar> b, int nx, int ny) {
  /*
   * a is the image greyscale
   * b is the buffer for the filter calcs
   * nx ny are image dims
   * c is the 9 point filter coeffs
   * */
  // uchar is an unsigned int from 0-255, char is from -127 -128
  // becase c++ stores chars as their ascii values
  // with this however we could have to copy the initiated c data to fc
  // cudaMemcpyToSymbol(fc, c.data(), 9*sizeof(float)) in the main fucntion
  // before call to this kernel
  auto idx = [&nx](int y, int x) { return y * nx + x; };

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < 0 || y < 0 || x >= nx || y >= ny)
    return;

  int xl = max(0, x - 1);
  int yl = max(0, y - 1);
  int xh = min(nx - 1, x + 1);
  int yh = min(ny - 1, y + 1);

  float v =
      fc[0] * a[idx(yl, xl)] + fc[1] * a[idx(yl, x)] + fc[2] * a[idx(yl, xh)] +
      fc[3] * a[idx(y, xl)] + fc[4] * a[idx(y, x)] + fc[5] * a[idx(y, xh)] +
      fc[6] * a[idx(yh, xl)] + fc[7] * a[idx(yh, x)] + fc[8] * a[idx(yh, xh)];

  uint f = (uint)(v + 0.5f);
  b[idx(y, x)] = (uchar)min(255, max(0, (int)f));
}

// still inefficient each thread reads from global memroy and each pixel is read
// by 9 threads saved by using 32byte uchar4 then plcaing in shared mem
// each thread handles 4 elemes of the image stored in conssecutive bytes of one
// 32 but word designed specifically for 2d blocks of dim 16x16 with shared mem
// at 66x18 to hold 64x16 bytes with a halo 1 byte deep
__global__ void filter9PT_3(cr_Ptr<uchar> a, r_Ptr<uchar> b, int nx, int ny) {

  __shared__ uchar as[18][66];
  auto idx = [&nx](int y, int x) { return y * nx + x; };

  // origins within shared memory will bw 1 byte from the edges ie 1,1 to 65,17
  int x0 = blockIdx.x * 64;
  int y0 = blockIdx.y * 16;
  int xa = x0 + threadIdx.x * 4;
  int ya = y0 + threadIdx.y;

  // x,y in shared mem
  int x = threadIdx.x * 4 + 1;
  int y = threadIdx.y + 1;
  const uchar4 a4 = reinterpret_cast<const uchar4 *>(a)[idx(ya, xa) / 4];
  as[y][x] = a4.x;
  as[y][x + 1] = a4.y;
  as[y][x + 2] = a4.z;
  as[y][x + 3] = a4.w;

  // warp 0 threads 0-15: copy top (y0-1) row to halo
  if (y == 1) {
    int ytop = max(0, y0 - 1);
    as[0][x] = a[idx(ytop, xa)];
    as[0][x + 1] = a[idx(ytop, xa + 1)];
    as[0][x + 2] = a[idx(ytop, xa + 2)];
    as[0][x + 3] = a[idx(ytop, xa + 3)];

    if (threadIdx.x == 0) {
      // top corners
      int xleft = max(0, x0 - 1);
      as[0][0] = a[idx(ytop, xleft)];
      int xright = min(nx - 1, x0 + 64);
      as[0][65] = a[idx(ytop, xright)];
    };
    int xlft = max(0, x0 - 1);

    // halo left edges
    as[threadIdx.x + 1][0] = a[idx(y0 + threadIdx.x, xlft)];
  };

  // awrp 1 thread 0-15: copy bottom row (y0+16) to halo
  if (y == 3) {
    int ybot = min(ny - 1, y0 + 16);
    as[17][x] = a[idx(ybot, xa)];
    as[17][x + 1] = a[idx(ybot, xa + 1)];
    as[17][x + 2] = a[idx(ybot, xa + 2)];
    as[17][x + 3] = a[idx(ybot, xa + 3)];

    if (threadIdx.x == 0) {
      // vottom corners
      int xleft = max(0, x0 - 1);
      as[17][0] = a[idx(ybot, xleft)];
      int xright = min(nx - 1, x0 + 64);
      as[17][65] = a[idx(ybot, xright)];
    }
    int xrgt = min(nx - 1, x0 + 64);

    // right halo edges
    as[threadIdx.x + 1][65] = a[idx(y0 + threadIdx.x, xrgt)];
  }
  __syncthreads();
  uchar bout[4];

  for (int k = 0; k < 4; k++) {
    float v = fc[0] * as[y - 1][x - 1] + fc[1] * as[y - 1][x] +
              fc[2] * as[y - 1][x + 1] + fc[3] * as[y][x - 1] +
              fc[4] * as[y][x] + fc[5] * as[y][x + 1] +
              fc[6] * as[y + 1][x - 1] + fc[7] * as[y + 1][x] +
              fc[8] * as[y + 1][x + 1];

    uint kf = (uint)(v + 0.5f);
    bout[k] = (uchar)min(255, max(0, kf));
    x++;
  }
  reinterpret_cast<uchar4 *>(b)[idx(ya, xa) / 4] =
      reinterpret_cast<uchar4 *>(bout)[0];
}

__global__ void filter9PT_RGB(cr_Ptr<uchar> a, r_Ptr<uchar> b, int nx_rgb,
                              int ny, cr_Ptr<float> c) {
  auto idx = [&nx_rgb](int y, int x) { return y * nx_rgb + x; };

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  // nx_rgb is now image_width * 3
  if (x < 0 || y < 0 || x >= nx_rgb || y >= ny)
    return;

  // We step by 3 to stay in the same color channel (R->R, G->G, B->B)
  int xl = max(x % 3, x - 3);
  int yl = max(0, y - 1);
  int xh = min(nx_rgb - (3 - (x % 3)), x + 3);
  int yh = min(ny - 1, y + 1);

  // parallel stencil
  float v = c[0] * a[idx(yl, xl)] + c[1] * a[idx(yl, x)] +
            c[2] * a[idx(yl, xh)] + c[3] * a[idx(y, xl)] + c[4] * a[idx(y, x)] +
            c[5] * a[idx(y, xh)] + c[6] * a[idx(yh, xl)] +
            c[7] * a[idx(yh, x)] + c[8] * a[idx(yh, xh)];

  uint f = (uint)(v + 0.5f);
  b[idx(y, x)] = (uchar)min(255, max(0, (int)f));
}

// int main(int argc, char* argv[]) {
//     if (argc < 2) {
//         printf("Usage: %s <input_image>\n", argv[0]);
//         return 1;
//     }

//     const char* output_path = (argc < 3) ? argv[3] : "filtered.png";

//     // 1. Load Image as Greyscale
//     int nx, ny, channels;
//     unsigned char* h_input = stbi_load(argv[1], &nx, &ny, &channels, 1);
//     if (!h_input) return 1;

//     size_t num_pixels = nx * ny;
//     thrust::device_vector<unsigned char> d_input(h_input, h_input +
//     num_pixels); thrust::device_vector<unsigned char> d_output(num_pixels);

//     // 2. Define a simple 3x3 Sharpen filter
//     float h_coeffs[9] = { 0, -1,  0,
//                          -1,  5, -1,
//                           0, -1,  0 };
//     thrust::device_vector<float> d_coeffs(h_coeffs, h_coeffs + 9);

//     // 3. Launch Kernel
//     dim3 threads(16, 16);
//     dim3 blocks((nx + 15) / 16, (ny + 15) / 16);

//     filter9PT<<<blocks, threads>>>(
//         d_input.data().get(),
//         d_output.data().get(),
//         nx, ny,
//         d_coeffs.data().get()
//     );

//     // 4. Copy back and Save
//     thrust::host_vector<unsigned char> h_output = d_output;

//     stbi_write_png(output_path, nx, ny, 1, h_output.data(), nx);

//     stbi_image_free(h_input);
//     printf("Filter applied. Output saved to filtered_output.png\n");
//     return 0;
// }

int main(int argc, char *argv[]) {
  if (argc < 2) {
    printf("Usage: %s <input_image> [output_path]\n", argv[0]);
    return 1;
  }

  const char *output_path = (argc >= 3) ? argv[2] : "filtered_rgb.png";

  // 1. Load Image as RGB (force 3 channels)
  int nx, ny, channels;
  unsigned char *h_input = stbi_load(argv[1], &nx, &ny, &channels, 3);
  if (!h_input) {
    printf("Error loading image\n");
    return 1;
  }

  // num_pixels remains nx * ny, but total data is nx * ny * 3
  size_t total_elements = nx * ny * 3;
  thrust::device_vector<unsigned char> d_input(h_input,
                                               h_input + total_elements);
  thrust::device_vector<unsigned char> d_output(total_elements);

  // 2. Define filter coefficients (e.g., Sharpen)
  float h_coeffs[9] = {0, -1, 0, -1, 5, -1, 0, -1, 0};
  thrust::device_vector<float> d_coeffs(h_coeffs, h_coeffs + 9);

  // 3. Launch Kernel
  // We treat the image as having width (nx * 3) so the stencil
  // operations stay within the same color channel (R-to-R, G-to-G, B-to-B)
  int nx_rgb = nx * 3;
  dim3 threads(16, 16);
  dim3 blocks((nx_rgb + 15) / 16, (ny + 15) / 16);

  filter9PT_RGB<<<blocks, threads>>>(
      (cr_Ptr<uchar>)d_input.data().get(), (r_Ptr<uchar>)d_output.data().get(),
      nx_rgb, ny, (cr_Ptr<float>)d_coeffs.data().get());

  cudaDeviceSynchronize();

  // 4. Copy back and Save as RGB (3 channels)
  thrust::host_vector<unsigned char> h_output = d_output;
  stbi_write_png(output_path, nx, ny, 3, h_output.data(), nx * 3);

  stbi_image_free(h_input);
  printf("RGB Filter applied. Output saved to %s\n", output_path);
  return 0;
}
