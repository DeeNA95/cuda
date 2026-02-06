// image filters using stencils
//
#include "cuda_runtime.h"
#include "../../cx.h"
#define STB_IMAGE_IMPLEMENTATION
#include "../../../vendor/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../../vendor/stb_image_write.h"
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <string.h>

__global__ void filter9PT(cr_Ptr<uchar> a, r_Ptr<uchar> b, int nx, int ny, cr_Ptr<float> c)
{
   /*
    * a is the image greyscale
    * b is the buffer for the filter calcs
    * nx ny are image dims
    * c is the 9 point filter coeffs
    * */
   // uchar is an unsigned int from 0-255, char is from -127 -128
   // becase c++ stores chars as their ascii values
    auto idx = [&nx](int y, int x){ return y * nx + x; };

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < 0 || y < 0 || x >= nx || y >= ny) return;

    int xl = max(0, x - 1);
    int yl = max(0, y - 1);
    int xh = min(nx - 1, x + 1);
    int yh = min(ny - 1, y + 1);

    float v = c[0] * a[idx(yl, xl)] + c[1] * a[idx(yl, x)] +
              c[2] * a[idx(yl, xh)] + c[3] * a[idx(y, xl)] +
              c[4] * a[idx(y, x)]  + c[5] * a[idx(y, xh)] +
              c[6] * a[idx(yh, xl)] + c[7] * a[idx(yh, x)] +
              c[8] * a[idx(yh, xh)];

    uint f = (uint)(v + 0.5f);
    b[idx(y, x)] = (uchar)min(255, max(0, (int)f));
}

__global__ void filter9PT_RGB(cr_Ptr<uchar> a, r_Ptr<uchar> b, int nx_rgb, int ny, cr_Ptr<float> c)
{
    auto idx = [&nx_rgb](int y, int x){ return y * nx_rgb + x; };

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // nx_rgb is now image_width * 3
    if (x < 0 || y < 0 || x >= nx_rgb || y >= ny) return;

    // We step by 3 to stay in the same color channel (R->R, G->G, B->B)
    int xl = max(x % 3, x - 3);
    int yl = max(0,     y - 1);
    int xh = min(nx_rgb - (3 - (x % 3)), x + 3);
    int yh = min(ny - 1, y + 1);

    //parallel stencil
    float v = c[0] * a[idx(yl, xl)] + c[1] * a[idx(yl, x)] +
              c[2] * a[idx(yl, xh)] + c[3] * a[idx(y, xl)] +
              c[4] * a[idx(y, x)]  + c[5] * a[idx(y, xh)] +
              c[6] * a[idx(yh, xl)] + c[7] * a[idx(yh, x)] +
              c[8] * a[idx(yh, xh)];

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
//     thrust::device_vector<unsigned char> d_input(h_input, h_input + num_pixels);
//     thrust::device_vector<unsigned char> d_output(num_pixels);

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

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s <input_image> [output_path]\n", argv[0]);
        return 1;
    }

    const char* output_path = (argc >= 3) ? argv[2] : "filtered_rgb.png";

    // 1. Load Image as RGB (force 3 channels)
    int nx, ny, channels;
    unsigned char* h_input = stbi_load(argv[1], &nx, &ny, &channels, 3);
    if (!h_input) {
        printf("Error loading image\n");
        return 1;
    }

    // num_pixels remains nx * ny, but total data is nx * ny * 3
    size_t total_elements = nx * ny * 3;
    thrust::device_vector<unsigned char> d_input(h_input, h_input + total_elements);
    thrust::device_vector<unsigned char> d_output(total_elements);

    // 2. Define filter coefficients (e.g., Sharpen)
    float h_coeffs[9] = { 0, -1,  0,
                         -1,  5, -1,
                          0, -1,  0 };
    thrust::device_vector<float> d_coeffs(h_coeffs, h_coeffs + 9);

    // 3. Launch Kernel
    // We treat the image as having width (nx * 3) so the stencil
    // operations stay within the same color channel (R-to-R, G-to-G, B-to-B)
    int nx_rgb = nx * 3;
    dim3 threads(16, 16);
    dim3 blocks((nx_rgb + 15) / 16, (ny + 15) / 16);

    filter9PT_RGB<<<blocks, threads>>>(
        (cr_Ptr<uchar>)d_input.data().get(),
        (r_Ptr<uchar>)d_output.data().get(),
        nx_rgb, ny,
        (cr_Ptr<float>)d_coeffs.data().get()
    );

    cudaDeviceSynchronize();

    // 4. Copy back and Save as RGB (3 channels)
    thrust::host_vector<unsigned char> h_output = d_output;
    stbi_write_png(output_path, nx, ny, 3, h_output.data(), nx * 3);

    stbi_image_free(h_input);
    printf("RGB Filter applied. Output saved to %s\n", output_path);
    return 0;
}
