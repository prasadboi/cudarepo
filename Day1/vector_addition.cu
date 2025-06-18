#include <iostream>
using namespace std;
#define ll long long int

__global__ // Marks a function as a kernel function that runs on the GPU. Callable from the Host (CPU) code. Runs on the Device (GPU).
// __device__ // Marks a function as a device function that runs on the GPU. Callable from other device or global functions. Runs on the Device (GPU).
void vecAddKernel(float* A, float* B, float* C, ll n){
	ll i = threadIdx.x + blockDim.x*blockIdx.x;
	if(i < n)
		C[i] = B[i] + A[i];
}
void vecAdd(float* A, float* B, float* C, ll n){
	float *A_d, *B_d, *C_d;
	ll size = n * sizeof(float);
	cudaMalloc((void **) &A_d, size);
	cudaMalloc((void **) &B_d, size);
	cudaMalloc((void **) &C_d, size);
	cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);
	vecAddKernel<<<ceil(n/256.0), 256>>>(A_d, B_d, C_d, n);
	cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);
	cudaFree(A_d);
	cudaFree(B_d);
	cudaFree(C_d);
}

void linearVecAdd(float* A, float* B, float* C, ll n){
    for(ll i = 0; i < n; i++)
        C[i] = A[i] + B[i];
}
int main()
{
    ll n = 1e4;
    float* A = new float[n];
    float* B = new float[n];
    float* C = new float[n];
    for(ll i = 0; i < n; i++) {
        A[i] = static_cast<float>(i);
        B[i] = static_cast<float>(i * 2);
    }
    // compare the time between vecAdd and linearVecAdd
    float timeVecAdd, timeLinearVecAdd;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    vecAdd(A, B, C, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timeVecAdd, start, stop);
    cout << "Time taken by vecAdd: " << timeVecAdd << " ms" << endl;

    cudaEventRecord(start);
    linearVecAdd(A, B, C, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timeLinearVecAdd, start, stop);
    cout << "Time taken by linearVecAdd: " << timeLinearVecAdd << " ms" << endl;
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}

// Execution:
// nvcc vector_addition.cu -o vector_addition
// nvprof ./vector_addition