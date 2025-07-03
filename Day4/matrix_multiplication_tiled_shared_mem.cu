#include <iostream>
#include <math.h>
using namespace std;
#define TILE_WIDTH 16
template<typename T>
__global__ void matrixMulKernelTiled(T const* M, T const* N, T* P, int h1, int w1, int h2, int w2)
{
	__shared__ T M_d_shared[TILE_WIDTH][TILE_WIDTH];
	__shared__ T N_d_shared[TILE_WIDTH][TILE_WIDTH];

	int row = blockIdx.x*TILE_WIDTH + threadIdx.y;
	int col = blockIdx.y*TILE_WIDTH + threadIdx.x;

	// assume h1 == w2
	T value = 0;
    int numTiles = (w1 + TILE_WIDTH - 1)/TILE_WIDTH;
	for(size_t ph = 0; ph < numTiles; ++ph)
	{
		// load the M and N tiles into shared memory
		M_d_shared[threadIdx.y][threadIdx.x] = M[row*w1 + ph*TILE_WIDTH + threadIdx.x];
		N_d_shared[threadIdx.y][threadIdx.x] = N[(ph*TILE_WIDTH + threadIdx.y)*h2 + col];
		__syncthreads();
		for(size_t k = 0; k < TILE_WIDTH; k++)
		{
			value += (M_d_shared[threadIdx.y][k]*N_d_shared[k][threadIdx.x]);
		}
		__syncthreads();
	}
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
    const int w1 = 16;
    const int h2 = 16;
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

    // Print a small part of the result
    cout << "Kernel function output (first 5x5 block):" << endl;
    for (int i = 0; i < 5 && i < h1; ++i) {
        for (int j = 0; j < 5 && j < w2; ++j) {
            cout << P_h[i * w2 + j] << " ";
        }
        cout << endl;
    }
    cout<< "------------------------" << endl;
    MatrixMul(M_h, N_h, P_test_h, h1, w1, h2, w2); // CPU version for testing
    cout << "Ideal output (first 5x5 block):" << endl;
    for (int i = 0; i < 5 && i < h1; ++i) {
        for (int j = 0; j < 5 && j < w2; ++j) {
            cout << P_test_h[i * w2 + j] << " ";
        }
        cout << endl;
    }
    cout<< "------------------------" << endl;
    // Cleanup
    delete[] M_h;
    delete[] N_h;
    delete[] P_h;
    cudaFree(M_d);
    cudaFree(N_d);
    cudaFree(P_d);

    return 0;
}