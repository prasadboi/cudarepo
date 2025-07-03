#include <iostream>
#include <math.h>
using namespace std;
#define BLOCKSIZE 16
// FIXING THE ISSUE OF NON COALESCED MEMORY ACCESS with the naive implementation
// also added good coding practices by referencing Lei Mao's Blog
template<typename T>
__global__
void matrixMulGlobalMemCoalesce(T const* M, T const* N, T* P, size_t h1, size_t w1, size_t h2, size_t w2){
	// row and column of C that current thread is working on
    size_t const row_idx = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
    size_t const col_idx = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

    if((row_idx < h1) and (col_idx < w2)){
        T val = 0;
        for(size_t k = 0; k < w1; ++k)
        {
            val += (M[row_idx*w1 + k]*N[k*w2 + col_idx]);
            
        }
        P[row_idx*w1 + col_idx] = val;
    }
}

template<typename T>
__host__
void launch_GMEM_kernel(int kernel_type, T const* M, T const* N, T* P, size_t h1, size_t w1, size_t h2, size_t w2)
{
    dim3 blockDim(BLOCKSIZE, BLOCKSIZE);
    dim3 gridDim(ceil(((float)h1)/((float)blockDim.x)), ceil(((float)w2)/((float)blockDim.y)));
    switch (kernel_type)
    {
    case 1:
        matrixMulGlobalMemCoalesce<T><<<gridDim, blockDim>>>(M, N, P, h1, w1, h2, w2);
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