#include "cx.h"
#include "random"
#include <thrust/host_vector.h>

int hostmatmul0(r_Ptr<float> C, cr_Ptr<float> B, cr_Ptr<float> A, int Ay,
                int Bx, int Ax) {
  // calc C[i * Bx + j] = A[i * Ax + k] * B[k * Bx + j]
  for (int i = 0; i < Ay; i++) {
    for (int j = 0; j < Bx; j++) {
      C[i * Bx + j] = 0.0;
      for (int k = 0; k < Ax; k++)
        C[i * Bx + j] += A[i * Ax + k] * B[k * Bx + j];
    }
  }
  return 0;
}

__global__ void devmatmul0(r_Ptr<float> C, cr_Ptr<float> B, cr_Ptr<float> A,
                           int Ay, int Bx, int Ax) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < Ay && j < Bx) {
    float sum = 0.0f;
    #pragma unroll 32
    for (int k = 0; k < Ax; k++) {
      sum += A[i * Ax + k] * B[k * Bx + j];
    }
    C[i * Bx + j] = sum;
  }
}

template <int TS>
__global__ void gputiled(float *__restrict C, float *__restrict A,
                         float *__restrict B, int Ay, int Ax, int Bx) {
  __shared__ float Atile[TS][TS];
  __shared__ float Btile[TS][TS];

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  float csum = 0.0f;
//   #pragma unroll TS

  // Iterate over tiles
  for (int t = 0; t < (Ax + TS - 1) / TS; ++t) {
    // Calculate global memory indices for this tile
    int a_col = t * TS + threadIdx.x;
    int b_row = t * TS + threadIdx.y;

    // Load one element of A and one element of B into shared memory
    // Add boundary checks
    if (row < Ay && a_col < Ax) {
      Atile[threadIdx.y][threadIdx.x] = A[row * Ax + a_col];
    } else {
      Atile[threadIdx.y][threadIdx.x] = 0.0f;
    }

    if (b_row < Ax && col < Bx) {
      Btile[threadIdx.y][threadIdx.x] = B[b_row * Bx + col];
    } else {
      Btile[threadIdx.y][threadIdx.x] = 0.0f;
    }

    __syncthreads();

    // Compute dot product for the current tile
    #pragma unroll TS
    for (int k = 0; k < TS; ++k) {
      csum += Atile[threadIdx.y][k] * Btile[k][threadIdx.x];
    }

    __syncthreads();
  }

  // Write the result to C with boundary check
  if (row < Ay && col < Bx) {
    C[row * Bx + col] = csum;
  }
}

int main(int argc, char **argv) {
  int Arow = (argc > 1) ? atoi(argv[1]) : 1024;
  int Acol = (argc > 2) ? atoi(argv[2]) : Arow;
  int Brow = Acol;
  int Bcol = (argc > 3) ? atoi(argv[3]) : Brow;
  int Crow = Arow, Ccol = Bcol;
  uint tilex = (argc > 4) ? atoi(argv[4]) : 32;
  uint tiley = (argc > 5) ? atoi(argv[5]) : tilex;

  thrust::host_vector<float> A(Arow * Acol);
  thrust::host_vector<float> B(Brow * Bcol);
  thrust::host_vector<float> C(Crow * Ccol);
  thrust::device_vector<float> devA(Arow * Acol);
  thrust::device_vector<float> devB(Brow * Bcol);
  thrust::device_vector<float> devC(Crow * Ccol);
  thrust::device_vector<float> A_tiled(Arow * Acol);
  thrust::device_vector<float> B_tiled(Brow * Bcol);
  thrust::device_vector<float> C_tiled(Crow * Ccol);

  std::default_random_engine gen(12345678);
  std::uniform_real_distribution<float> fran(0.0, 1.0);

  #pragma unroll 32
  for (int k = 0; k < Arow * Acol; k++)
    A[k] = fran(gen);
  #pragma unroll 32
  for (int k = 0; k < Brow * Bcol; k++)
    B[k] = fran(gen);

  devA = A_tiled = A;
  devB = B_tiled = B;

  // flops calc
  float flops = 2 * (float)Arow * (float)Acol * (float)Bcol;

  dim3 threads = {tilex, tiley, 1};
  dim3 blocks = {(Bcol + threads.x - 1) / threads.x,
                 (Arow + threads.y - 1) / threads.y, 1};

  auto start3 = std::chrono::high_resolution_clock::now();

  gputiled<32><<<blocks, threads>>>(C_tiled.data().get(), A_tiled.data().get(),
                                    B_tiled.data().get(), Arow, Bcol, Acol);

  cudaDeviceSynchronize();
  auto end3 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float> elapsed3 = end3 - start3;

  float gflops3 = flops / (elapsed3.count() * 1000000.0);
  float gbytes3 = gflops3 * 6.0; // 12 bytes per term
  printf("A %d x %d B %d x %d device time %.3f ms Gflops/sec %.3f\n", Arow,
         Acol, Brow, Bcol, elapsed3.count() * 1000.0, gflops3);
  std::cout << "GFLOPS: " << gflops3 << std::endl;
  std::cout << "GBytes/sec: " << gbytes3 << std::endl;
  std::cout << "Time taken: " << elapsed3.count() << " seconds" << std::endl;

  auto start2 = std::chrono::high_resolution_clock::now();
  devmatmul0<<<blocks, threads>>>(devC.data().get(), devA.data().get(),
                                  devB.data().get(), Arow, Bcol, Acol);
  cudaDeviceSynchronize();
  auto end2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float> elapsed2 = end2 - start2;

  float gflops2 = flops / (elapsed2.count() * 1000000.0);
  float gbytes2 = gflops2 * 6.0; // 12 bytes per term
  printf("A %d x %d B %d x %d device time %.3f ms Gflops/sec %.3f\n", Arow,
         Acol, Brow, Bcol, elapsed2.count() * 1000.0, gflops2);
  std::cout << "GFLOPS: " << gflops2 << std::endl;
  std::cout << "GBytes/sec: " << gbytes2 << std::endl;
  std::cout << "Time taken: " << elapsed2.count() << " seconds" << std::endl;

  // auto start = std::chrono::high_resolution_clock::now();
  // hostmatmul0(C.data(),A.data(),B.data(),Arow,Acol,Bcol);
  // auto end = std::chrono::high_resolution_clock::now();
  // std::chrono::duration<float> elapsed = end - start;

  // // printf("FLOPS = %f",flops );
  // float gflops= flops/(elapsed.count()*1000000.0);
  // float gbytes = gflops*6.0; // 12 bytes per term
  // printf("A %d x %d B %d x %d host time %.3f ms Gflops/sec
  // %.3f\n",Arow,Acol,Brow,Bcol,elapsed.count()*1000.0,gflops); std::cout <<
  // "GFLOPS: " << gflops << std::endl; std::cout << "GBytes/sec: " << gbytes <<
  // std::endl; std::cout << "Time taken: " << elapsed.count() << " seconds" <<
  // std::endl;

  // Copy results back to host to verify
  thrust::host_vector<float> C_host_from_devC = devC;
  thrust::host_vector<float> C_host_from_C_tiled = C_tiled;

  // Verify the results
  bool correct = true;
  for (int i = 0; i < Crow * Ccol; i++) {
    if (fabs(C_host_from_devC[i] - C_host_from_C_tiled[i]) > 1e-5) {
      correct = false;
      break;
    }
  }

  if (correct) {
    std::cout
        << "Verification successful: devmatmul0 and gputiled results match."
        << std::endl;
  } else {
    std::cout
        << "Verification failed: devmatmul0 and gputiled results do not match."
        << std::endl;
  }

  int twothree = 2 * Bcol + 3;

  // The original C vector is uninitialized because hostmatmul0 is commented
  // out. We will print the results from the GPU computations instead.
  printf("C[2,3] from devmatmul0: %f \n", C_host_from_devC[twothree]);
  printf("C[2,3] from gputiled: %f \n", C_host_from_C_tiled[twothree]);

  return 0;
}
