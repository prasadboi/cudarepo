# Memory Architecture and data locality

- GOAL: How can one effectively organise and position data such that there is effecient access by a large number of threads?

## Importance of memory access efficiency

- [[Math Bound Matrix Multiplication Kernel]]
- COMPUTE_TO_GLOBAL_MEMORY_ACCESS_RATIO OR ARITHMETIC_INTENSITY OR COMPUTATIONAL_INTENSITY
  - Defined as the number of FLOPs performed for each byte access from teh global memory with a region of a program.
- Memory Bound programs (Low Computational Intensity)
  - Programs that are limited by the rate at which the data can be delivered from memory to the GPU cores.
- Compute Bound programs (Higher Computational Intensity)
  - Programs that are limited by the number of computations that can be performed by the GPU
    - not limited by the memory bandwidth.

## CUDA Memory Types

- Categories of GPU memory
  - Global Memory
    - R and W by the host
    - R and W by the device
    - Off the processor chip, implemented with DRAM tech.
      - long access latencies
      - low access BW
  - Constant Memory
    - R and W by the host
    - Short Latency, High BW, Read only access by the device
    - Stored in global memory but cached for future efficient access.
  - Local Memory
    - placed in global memory, and has similar latency but is not shared across threads.
    - Data that is private to the thread but cannot be stored in registers is stored here.
  - Registers
    - On-chip memory
    - Variables that reside in these types of memory can be accessed at very high speed in a highly parallel manner.
    - Allocated to individual threads.
    - Direct use of values possible by referencing the registers where the values are stored. Registers are limited in number, therefore oversubscribing should be avoided.
  - Shared Memory
    - On-chip memory
    - Variables that reside in these types of memory can be accessed at very high speed in a highly parallel manner.
    - Allocated to threads blocks.
    - Designed as part of the memory space that resides on the processor chip.
      - Requires a load and store operation, but these operations have low latency as the memory is on chip.
- NOTE: Register accesses involve fewer instructions than an access to the global memory
- NOTE: On chip memory - v. short access latencies and relatively low access BW
- ![[image-4.png]]
- NOTE:
  - If a var.s scope is single thread, a private version of the variale will be created for every thread - with each thread only allowed to access its private version of the variable.
  - If a variable’s lifetime is within a grid’s execution, it must be declared within the kernel function body and will be available for use only by the kernel’s code. If the kernel is invoked several times, the value of the variable is not maintained across these invocations. Each invocation must initialize the variable in order to use it. On the other hand, if a variable’s lifetime is throughout the entire application, it must be declared outside of any function body. The contents of these variables are maintained throughout the execution of the application and available to all kernels.

## Tiling for reduced memory traffic

- Strategy
  - Partition the data into subsets (tiles) so that each tile fits into the shared memory.
  - Criterion: Kernel computation on each of these tiles can be done indepently of each other.

## A tiled matrix multiplication kernel

```CPP
#define BLOCKSIZE 32
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
```
