#include    "cuda_runtime.h"
#include "cublas_v2.h"
#include "cx.h"
#include "random"

int main(int argc, char **argv) {
  int Arow = (argc > 1) ? atoi(argv[1]) : 1024;
  int Acol = (argc > 2) ? atoi(argv[2]) : Arow;
  int Brow = Acol;
  int Bcol = (argc > 3) ? atoi(argv[3]) : Brow;
  int Crow = Arow, Ccol = Bcol;
  uint tilex = (argc > 4) ? atoi(argv[4]) : 32;
  uint tiley = (argc > 5) ? atoi(argv[5]) : tilex;

  thrust::host_vector<float> A(Arow*Acol);
  thrust::host_vector<float> B(Brow*Bcol);
  thrust::host_vector<float> C(Crow*Ccol);
  thrust::device_vector<float> devA(Arow*Acol);
  thrust::device_vector<float> devB(Brow*Bcol);
  thrust::device_vector<float> devC(Crow*Ccol);
  thrust::device_vector<float> devD(Crow*Ccol);

  //init a and b with random values
  std::default_random_engine gen(12345678);
  std::uniform_real_distribution<float> fran(0.0, 1.0);

  float flops = 2 * (float)Arow * (float)Acol * (float)Bcol;


  for (int i = 0; i < Arow*Acol; i++) A[i] = fran(gen);
  for (int i = 0; i < Brow*Bcol; i++) B[i] = fran(gen);

  devA = A; //copy to device
  devB = B;

  devC = C; //empty

  float alpha = 1.0; //for cublas def
  float beta = 1.0;

  cublasHandle_t handle;
  cublasCreate(&handle);

  //enable tensor cores
  cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

  auto start = std::chrono::high_resolution_clock::now();

  // perform matrix multiplication
  cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, Crow, Ccol, Arow, &alpha,
              devA.data().get(), Acol, devB.data().get(), Bcol, &beta,
              devC.data().get(), Crow);

  beta = 0.0f;
  // D = transpose(C)
  cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, Crow, Ccol, &alpha, devC.data().get(),
              Crow, &beta, devC.data().get(), Crow, devD.data().get(),
              Ccol);

  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float> elapsed = end - start;

  C = devD;


  float gflops = flops / (elapsed.count() * 1000000.0);
  float gbytes = gflops * 6.0; // 12 bytes per term
  printf("A %d x %d B %d x %d device time %.3f ms Gflops/sec %.3f\n", Arow,
         Acol, Brow, Bcol, elapsed.count() * 1000.0, gflops);
  std::cout << "GFLOPS: " << gflops << std::endl;
  std::cout << "GBytes/sec: " << gbytes << std::endl;
  std::cout << "Time taken: " << elapsed.count() << " seconds" << std::endl;


    return 0;
}

