#include <iostream>
#include <math.h>
using namespace std;
#define TILE_WIDTH 16
template<typename T>
__global__ void matrixMulKernelTiled(T const* M, T const* N, T* P, int h1, int w1, int h2, int w2)
{
	__shared__ T M_d_shared[TILE_WIDTH][TILE_WIDTH];
	__shared__ T N_d_shared[TILE_WIDTH][TILE_WIDTH];

	int row = blockIdx.y*TILE_WIDTH + threadIdx.y;
	int col = blockIdx.x*TILE_WIDTH + threadIdx.x;

	// assume h1 == w2
	T value = 0;
    int numTiles = (w1 + TILE_WIDTH - 1)/TILE_WIDTH;
	for(size_t ph = 0; ph < numTiles; ++ph)
	{
		// load the M and N tiles into shared memory
		if ((row < h1) and ((ph*TILE_WIDTH + threadIdx.x) < w1))
			M_d_shared[threadIdx.y][threadIdx.x] = M[row*w1 + ph*TILE_WIDTH + threadIdx.x];
		else
			M_d_shared[threadIdx.y][threadIdx.x] = 0.0f;
			
		if (((ph*TILE_WIDTH + threadIdx.y) < h2) and (col < w2))
			N_d_shared[threadIdx.y][threadIdx.x] = N[(ph*TILE_WIDTH + threadIdx.y)*w2 + col];
		else
			N_d_shared[threadIdx.y][threadIdx.x] = 0.0f;
			
		__syncthreads();
		for(size_t k = 0; k < TILE_WIDTH; k++)
		{
			value += (M_d_shared[threadIdx.y][k]*N_d_shared[k][threadIdx.x]);
		}
		__syncthreads();
	}
	if((row < h1) and (col < w2))
		P[row*w2 + col] = value;
}

template<typename T>
__host__
void launch_GMEM_kernel(int kernel_type, T const* M, T const* N, T* P, size_t h1, size_t w1, size_t h2, size_t w2)
{
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((w2 + TILE_WIDTH - 1) / TILE_WIDTH, (h1 + TILE_WIDTH - 1) / TILE_WIDTH) ;
    switch (kernel_type)
    {
    case 1:
        matrixMulKernelTiled<T><<<gridDim, blockDim>>>(M, N, P, h1, w1, h2, w2);
        break;
    
    default:
        cout<<"unidentified kernel launch type. No execution.\n";
        break;
    }
}

__host__
void MatrixMul(float* M, float* N, float* P, int h1, int w1, int h2, int w2)
{
    // Assume w1 == h2
    for (int row = 0; row < h1; ++row) {
        for (int col = 0; col < w2; ++col) {
            float value = 0.0f;
            for (int k = 0; k < w1; ++k) {
                value += M[row * w1 + k] * N[k * w2 + col];
            }
            P[row * w2 + col] = value;
        }
    }
}
int main() {
    // Matrix dimensions
    const int h1 = 16;
    const int w1 = 37;
    const int h2 = 37;
    const int w2 = 16;

    if (w1 != h2) {
        std::cerr << "Matrix dimensions invalid for multiplication!" << std::endl;
        return 1;
    }

    float* M_h = new float[h1 * w1];
    float* N_h = new float[h2 * w2];
    float* P_h = new float[h1 * w2];
    float* P_test_h = new float[h1 * w2];

    for (int i = 1; i <= h1 * w1; ++i) {
        M_h[i-1] = 1.0f*i;
    }
    for (int i = 1; i <= h2 * w2; ++i) {
        N_h[i-1] = 1.0f*i;
    }

    // Allocate device memory
    float *M_d, *N_d, *P_d;
    cudaMalloc(&M_d, h1 * w1 * sizeof(float));
    cudaMalloc(&N_d, h2 * w2 * sizeof(float));
    cudaMalloc(&P_d, h1 * w2 * sizeof(float));

    // Copy data to device
    cudaMemcpy(M_d, M_h, h1 * w1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(N_d, N_h, h2 * w2 * sizeof(float), cudaMemcpyHostToDevice);

    launch_GMEM_kernel<float>(1, M_d, N_d, P_d, h1, w1, h2, w2);
    cudaDeviceSynchronize();

    cudaMemcpy(P_h, P_d, h1 * w2 * sizeof(float), cudaMemcpyDeviceToHost);

    // checking if the output of the GPU kernel matches the actual expected output
    // getting accurate results from the naive, single thread matrix multiplication program
    MatrixMul(M_h, N_h, P_test_h, h1, w1, h2, w2);
    // Compare outputs only on the host (driver) thread
    bool correct = true;
    for(size_t i = 0; i < h1; i++)
    {
        for(size_t j = 0; j < w2; j++)
        {
            if(fabs(P_h[i*w2 + j] - P_test_h[i*w2 + j]) > 1e-4)
            {
                cout << "Mismatch at (" << i << "," << j << "): "
                     << "GPU=" << P_h[i*w2 + j] << ", CPU=" << P_test_h[i*w2 + j] << endl;
                correct = false;
            }
        }
    }
    if (correct)
        cout << "GPU Kernel output matches CPU output." << endl;
    else
        cout << "GPU Kernel provides wrong output!!!" << endl;
    // Cleanup
    delete[] M_h;
    delete[] N_h;
    delete[] P_h;
    cudaFree(M_d);
    cudaFree(N_d);
    cudaFree(P_d);

    return 0;
}

// nvprof ./matrix_mul_tiled_shared
/*
==124858== NVPROF is profiling process 124858, command: ./matrix_mul_tiled_shared
GPU Kernel output matches CPU output.
==124858== Profiling application: ./matrix_mul_tiled_shared
==124858== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   52.13%  3.1360us         1  3.1360us  3.1360us  3.1360us  void matrixMulKernelTiled<float>(float const *, float const *, float*, int, int, int, int)
                   27.13%  1.6320us         1  1.6320us  1.6320us  1.6320us  [CUDA memcpy DtoH]
                   20.74%  1.2480us         2     624ns     480ns     768ns  [CUDA memcpy HtoD]
      API calls:   98.99%  82.418ms         3  27.473ms  2.3650us  82.411ms  cudaMalloc
                    0.46%  380.55us         3  126.85us  19.817us  340.60us  cudaMemcpy
                    0.21%  173.51us       114  1.5220us     120ns  69.801us  cuDeviceGetAttribute
                    0.19%  160.10us         1  160.10us  160.10us  160.10us  cudaLaunchKernel
                    0.10%  87.163us         3  29.054us  2.6950us  79.329us  cudaFree
                    0.03%  21.490us         1  21.490us  21.490us  21.490us  cuDeviceGetName
                    0.01%  9.8790us         1  9.8790us  9.8790us  9.8790us  cudaDeviceSynchronize
                    0.01%  6.6130us         1  6.6130us  6.6130us  6.6130us  cuDeviceGetPCIBusId
                    0.00%  1.9320us         3     644ns     170ns  1.4520us  cuDeviceGetCount
                    0.00%     411ns         2     205ns     120ns     291ns  cuDeviceGet
                    0.00%     271ns         1     271ns     271ns     271ns  cuDeviceTotalMem
                    0.00%     220ns         1     220ns     220ns     220ns  cuModuleGetLoadingMode
                    0.00%     171ns         1     171ns     171ns     171ns  cuDeviceGetUuid
*/ 