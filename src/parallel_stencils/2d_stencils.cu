#include <cuda_runtime.h>
#include "../cx.h"
#include "../cxtimers.h"

__global__ void stencil2d(cr_Ptr<float> a, r_Ptr<float> b, int nx, int ny){
	auto idx = [&nx](int y, int x){return y*nx+x;};
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if(x<1 || y < 1 || x >= nx-1 || y >= ny-1) return ;

	b[idx(y,x)] = 0.25f*(a[idx(y,x+1)] + a[idx(y,x-1)]+ a[idx(y+1,x)] + a[idx(y-1,x)]);
}

int stencil2d_host (cr_Ptr<float> a, r_Ptr<float> b, int nx, int ny){
	auto idx = [&nx](int y, int x) {return y*nx+x;};
	for(int y=1; y<ny-1;y++){
		for(int x=1; x<nx-1; x++){
			b[idx(y,x)] = 0.25f*( a[idx(y,x+1)] + a[idx(y,x-1)] + a[idx(y-1,x)] + a[idx(y+1,x)]);

		}
	}
	return 0;
;}


int main( int argc, char *argv[]){
	int nx = (argc>1) ? atoi(argv[1]) : 1024;
	int ny = (argc>2) ? atoi(argv[2]) : 1024;
	int iter_host = (argc>3) ? atoi(argv[3]): 1000;
	int iter_gpu = (argc>4) ? atoi(argv[4]) : 10000;

	int size = nx*ny;

	thrustHvec<float> a(size);
	thrustHvec<float> b(size);
	thrustDvec<float> dev_a(size);
	thrustDvec<float> dev_b(size);

	auto idx = [&nx](int y, int x) {return y*nx+x;};
	for (int y=0; y<ny;y++) a[idx(y,0)] = a[idx(y,nx-1)] = 1.0f;

	a[idx(0,0)] = a[idx(0,nx-1)] = a[idx(ny-1,0)] = a[idx(ny-1,nx-1)] =0.5f;
	dev_a =a;
	dev_b =a;

	cxxtimer::Timer tim;
	tim.start();
	
	for (int k=0; k < iter_host/2; k++){
		stencil2d_host(a.data(), b.data(), nx, ny);
		stencil2d_host(b.data(),a.data(), nx, ny);
	}
	tim.stop();
	double t1 = tim.count<cxxtimer::ms>();
	double gflops_host = (double)(iter_host*4) * (double)size/(t1*1e6);
	
	dim3 threads = {16,16,1};
	dim3 blocks = { (nx+threads.z-1)/threads.x, (ny+threads.y-1)/threads.y,1};

	tim.reset();
	tim.start();
	for(int k=0; k < iter_gpu/2; k++){
		stencil2d<<<blocks,threads>>>(dev_a.data().get(),dev_b.data().get(), nx, ny);
		stencil2d<<<blocks,threads>>>(dev_b.data().get(), dev_a.data().get(), nx, ny);
	}
	cudaDeviceSynchronize();
	tim.stop();
	double t2 = tim.count<cxxtimer::ms>();

	double gflops_gpu = (double)(iter_gpu*4) * (double)size/(t2*1e6);
	double speedup = gflops_gpu/gflops_host;

	printf("host iter %8d time %9.3fms GFlops %8.3f\n", iter_host, t1, gflops_host);
	printf("gpu iter %8d time %9.3fms GFlops %8.3f\n", iter_gpu, t2, gflops_gpu);
	printf("Speedup: %.3f\n",speedup);

	return 0;



}
