// threadsidx is a 3 dim variable which allows us to identify the thread within
// a block each block can have up to 3 dimensions (x,y,z) and each dimension can
// have multiple threads this means when dealing with a vector we can use only x
// dimension when dealing with a matrix we can use x and y dimensions when
// dealing with a 3D volume we can use x,y and z dimensions (ie tensors)

#include <stdio.h>
#define N 4
#define N_BLOCKS 2

__global__ void MatAdd(float A[N][N], float B[N][N], float C[N][N]) {
  int row = threadIdx.y + blockIdx.y * blockDim.y;
  int col = threadIdx.x + blockIdx.x * blockDim.x;
  if (row < N && col < N)
    C[row][col] = A[row][col] + B[row][col];
}

int main() {
  float A[N][N], B[N][N], C[N][N];

  // fill the arrays A and B on the host a with sum(i,j) and B with mul(i,j)
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A[i][j] = static_cast<float>(i + j);
      B[i][j] = static_cast<float>(i * j);
    }
  }

  // runs the MatAdd kernel, with N x N threads in N_BLOCKS blocks
  dim3 threadsPerBlock(N, N); // threads per block is an inbuilt cuda function
                              // which takes in 2 or 3 dimensions
  // its of type dim3 which is a built in cuda type for 3 dimensions (threadidx)
  MatAdd<<<N_BLOCKS, threadsPerBlock>>>(A, B, C);
  cudaDeviceSynchronize();

  // Print results
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      printf("C[%d][%d] = %f + %f = %f\n", i, j, A[i][j], B[i][j], C[i][j]);
    }
  }

  return 0;
}
