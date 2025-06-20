#include <iostream>
using namespace std;

// P[row, col] = sum(M[row,k] * N[k, col]) for k = 0,1,2...,width-1
__global__
void MatrixMulKernel(float* M, float* N, float* P, int h1, int w1, int h2, int w2){
	// note that w1 must be equal to w2
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	if((row < h1) and (col < w2)){
		float value = 0.0;
		for(int k = 0; k < w1; k++){
			value += (M[row*w1 + k] * N[k*w2 + col]); // note the way the row major format is being applied to fetch the matrix values
		}
		P[row*w2 + col] = value;
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
    const int h1 = 128;
    const int w1 = 256;
    const int h2 = 256;
    const int w2 = 64;

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

    // Kernel launch parameters
    dim3 blockDim(25, 25);
    // note how the grid dims are being calculated!!!
    dim3 gridDim((h1 + blockDim.x - 1) / blockDim.x, (w2 + blockDim.y - 1) / blockDim.y);

    // Launch kernel
    MatrixMulKernel<<<gridDim, blockDim>>>(M_d, N_d, P_d, h1, w1, h2, w2);
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