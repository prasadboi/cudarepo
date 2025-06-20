Objectives

- colored -> grayscale
- blurring an image
- matrix multipcation

# Multi-dimensional grid organization

- grid consists of 1 or more blocks and each block consists of 1 or more threads
  - In general, a grid is a 3D array of blocks and a block is a 3D array of threads
    ```C
    	dim3 dimGrid(32,1,1)
    	dim3 dimBlock(128,1,1)
    	vecAddKernel<<<dimGrid, dimBlock>>>(...);
    ```
    This example generates a 1D grdi that consists fo 32 blocks, each of which consists of 128 threads. The total number of threads in the grid is 128\*32 = 4096
  - dimBlock and dimGrdi are host code var.s that are defined by the programmers. These variables can have any legal C variable name as long as they have the type dim3.
  - In case we want to dynamically assig the number of blocks based on the length of the
    ```C
      dim3 dimGrid(ceil(n/256.0), 1,1);
      dim3 dimBlock(256,1,1);
      vecAddKernal<<<dimGrid, dimBlock>>>(...);
    ```
  - Once the grid has been launched the grid and block dimensions will remain the same until the entire grid has finished exec.
  - gridDim and blockDim are built-int variables in a kernel and always reflect the dimensions of the grid and the blocks, respectively.
  - Range of the gridDim and blockDim values:
    - gridDim.x -> 1 to 2^31 - 1
    - gridDim.y -> 1 to 2^16 - 1
    - gridDim.z -> 1 to 2^16 - 1
    - blockDim.$\theta$ -> 0 to gridDim.$\theta$ -1
  - **The total size of a block in current CUDA systems is limited to 1024 threads. These threads can be distributed across the three dimensions in any way as long as the total number of threads does not exceed 1024.**
  - **The chocie of 1D, 2D, or 3D thread organizations i s usually based on the nature of the data.**
    - For e..g, pictures (2-D array of pixels), using a 2D grid that consists of 2D blocks is often convenient for processing the pixels in a picture.
      - Vertical (row) coordinate = blockIdx.Y \* blockDim.y + threadIdx.y
      - Horizontal (Column) coordinate = blockIdx.x \* blockDim.x + threadIdx.x

# Mapping threads to multidimensional data

- _NOTE:
  "We will refer to the dimensions of multidimensional data in descending order: the z dimension followed by the y dimension, and so on. For example, for a picture of n pixels in the vertical or y dimension and m pixels in the horizontal or x dimension, we will refer to it as a n 3 m picture. This follows the C multidimensional array indexing convention."_
- _NOTE:
  all multidimensional arrays in C are linearized. This is due to the use of a “flat” memory space in modern computers (see the “Memory Space” sidebar). In the case of statically allocated arrays, the compilers allow the programmers to use higher-dimensional indexing syntax, such as Pin_d[j][i], to access their elements. Under the hood, the compiler linearizes them into an equivalent 1D array and translates the multidimensional indexing syntax into a 1D offset. In the case of dynamically allocated arrays, the current CUDA C compiler leaves the work of such translation to the programmers, owing to lack of dimensional information at compile time._
  - Row Major Layout - used by C compiler
  - Column Major Layout - used by FORTRAN compiler
- Threads often perform complex operations on their data and need to cooperate with each other
  - Refer ImgBlur kernel

## colorToGrayscaleConversion

```CPP
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
```

## ImageBlur

- An image blurring function calculates the value of an output image pixel as a weighted sum of a partch of pixels encompassing the pixel in the input image.
  - Convolution pattern
- For this example we will not place any weight on the value of any pixel based on its distance from the target pixel (in practice, assigning such weights is quite common in convolution blurring approaches - such as Gaussian blur.)

```CPP
#define BLUR_SIZE 3
__global__
void blurKernel(unsigned char* Picture_in, unsigned char* Picture_out, int width, int height)
{
	int column = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if((column < width) and (row < height)){
		// assuming grayscale image as input
		int sum_pixel = 0;
		int count_pixel = 0;

		for(int i = -BLUR_SIZE; i <= BLUR_SIZE; i++){
			for(int j = -BLUR_SIZE; j <= BLUR_SIZE; j++)
				int nghbr_row = row + i;
				int nghbr_col = col + j;
				if((nghbr_col < width) and (nghbr_row < height)){
					sum_pixel += Picture_in[nghbr_row*width + nghbr_col];
					count_pixel++;
				}
			Picture_out[nghbr_row*width + nghbr_col] = (unsigned char)(sum_pixel/count_pixel);
		}
	}
}
```

## Matrix Multiplication

```CPP
// P[row, col] = sum(M[row,k] * N[k, col]) for k = 0,1,2...,width-1
__global__
void MatrixMulKernel(float* M, float* N, float* P, int h1, int w1, int h2, int w2){
	// note that w1 must be equal to w2
	int row = blockIdx.x * blockDim.x + rowIdx.x;
	int col = blockIdx.y * blockDim.y + rowIdx.y;
	if((row < h1) and (col < w2)){
		float value = 0.0;
		for(int k = 0; k < w1; k++){
			value += (M[row*w1 + k] * N[k*h2 + col]); // note the way the row major format is being applied to fetch the matrix values
		}
		M[row][col] = value;
	}
}
```

- “Since the size of a grid is limited by the maximum number of blocks per grid and threads per block, the size of the largest output matrix P that can be handled by matrixMulKernel will also be limited by these constraints. In the situation in which output matrices larger than this limit are to be computed, one can divide the output matrix into submatrices whose sizes can be covered by a grid and use the host code to launch a different grid for each submatrix. Alternatively, we can change the kernel code so that each thread calculates more P elements.” ([“Programming Massively Parallel Processors”, p. lxxxix](zotero://select/library/items/HILTB926)) ([pdf](zotero://open-pdf/library/items/575ZMX47?page=89&annotation=3U9XLS6Z))
- **Block size** is often chosen as a square (e.g., 16x16, 32x32, 25x25) for 2D problems, balancing occupancy and register/shared memory usage.
- **Grid size** is calculated to ensure every output element gets a thread, even if the matrix size isn’t a perfect multiple of the block size.
- The formula `(N + blockDim - 1) / blockDim` is a standard CUDA idiom for “ceiling division”.
