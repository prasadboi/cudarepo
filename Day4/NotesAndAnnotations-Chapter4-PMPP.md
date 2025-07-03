# Architecture of a Modern GPU

![[image 1.png]]

- Threads are assigned on a block by block basis
  - i.e. all threads in a block are simultaneously assigned to the same Streaming Multiprocessor (SM)
- Blocks need to reserve HW resources to execute, so only a limited number of blocks can be simultaneously assigned to a given SM. Therefore there is a limit on the tot. number of blocks that can be simultaneously executing in a CUDA device.
- While grids contain many more blocks than max blocks that can simultaneously exec on device, **runtime system** maintains a list of blocks that need to be executed - to assign new blocks to SMs, when prev. assigned blocks complete exec.
- “The assignment of threads to SMs on a block-by-block basis guarantees that threads in the same block are scheduled simultaneously on the same SM. This guarantee makes it possible for threads in the same block to interact with each other in ways that threads across different blocks cannot. This includes **barrier synchronization**,” (“Programming Massively Parallel Processors”, p. xciv) and acessing **low-latency shared memory** that resides on the SM.

# Synchronization and transparent scalability

![[image 2.png]]

- Threads of the same block can coordinate their activities using barrieer sync. function _\_\_syncthreads()_
  - _\_\_synchthreads()_ must be executed by all threads in a block.
  - i.e. if there exists a _\_\_synchthreads()_ statement in an if else statement,
    - If each path has a _\_\_synchthreads()_ statement, either all threads in a block exec. the then-path or all of them exec. the elseppath. The 2 _\_\_synchthreads()_ are diff. barrier sync. points.
  - Incorrect use of barrier sync. can lead to undefined exec. behavior/incorrect results/deadlock
  - To prevent excessive or indefinite waiting time during barrier sync. CUDA runtime sys. assigns exec. resources to all threads in a block as a unit SIMULTANEOUSLY.
- **by not allowing threads of diff. blocks to perform barrier sync. with each other, the CUDA runtime system can exec. blocks in any order relative to each other, since noe of them need to wait for each other. This flexibility enables scalable implementations.**
  ![[image.png]]
- The ability to execute the same application code on different hardware with different amounts of execution resources is referred to as **transparent scalability**, which reduces the burden on application developers and improves the usability of applications.

# Warps and SIMD Hardware

## Warps

- (For most implementations) Once a blck has been assigned to an SM, it is further divided into 32-thread units call **warps**.
  - Size of the warps is implementation specific and can vary in future generations of GPUs.
- A warp is the unit of thread scheduling in SMs.
- Blocks are paritioned into warps on the basis of thread indices.
  - If a block is org. into a 1-D array, the parition is straightforward.
  - For blocks that consist of multiple dimensions of threads, the dimensions will be projected into a linearised row-major layour before partitioning into warps.
    - The linear layout is determined by placing the rows with larger y and z coordinates after those with lower ones.
    - Example:
      if a block consists of two dimensions of threads, one will form the linear layout by placing all threads whose threadIdx.y is 1 after those whose threadIdx.y is 0. Threads whose threadIdx.y is 2 will be placed after those whose threadIdx.y is 1, and so on. Threads with the same threadIdx.y value are placed in consecutive positions in increasing threadIdx.x order.
      ![[image-2.png]]

## SIMD

- An SM is designed to execute all threads in a warp following the single-instruction, multiple-data (SIMD) model.
  - at any instant in time, one instruction is fetched and executed for all threads in the warp
- Because the SIMD hardware effectively restricts all threads in a warp to execute the same instruction at any point in time, the execution behavior of a warp is often referred to as single instruction, multiple-thread.
- The advantage of SIMD is that the cost of the control hardware, such as the instruction fetch/dispatch unit, is shared across many execution units. This design choice allows for a smaller percentage of the hardware to be dedicated to control and a larger percentage to be dedicated to increasing arithmetic throughput.

# Control Divergence

- SIMD exec works well when all threads within a warp follow the same exec. path
- When diff threads follow different exec paths, the SIMD hardware will take multiple passes through these paths, one pass for each path.
  - During each exec path, the other threads are not allowed to take effect.
- ![[image-3.png]]
- While this is executed sequentially in Pascal and prior architectures, scheduling is made feasible from the Volta Architecture onwards (2017).

### How to determine whether a particular condition statement results in thread divergence or not

- If the decision condition is based on threadIdx values, the control statement can potentially cause thread divergence.

- A prevalent reason for using a control construct with thread control divergence is handling boundary conditions when mapping threads to data.
- This is usually because the total number of threads needs to be a multiple of the thread block size, whereas the size of the data can be an arbitrary number.
- performance impact of control divergence decreases as vector size decreases
- **one cannot assume that all warps in the block will have the same execution timing with control divergence in play**
  - Therefore it is necessary to sync the warps at the end of their exec. as and when needed.
    - barrier sync. mech. like \_\_syncwarp()
  - (to ensure correctness)

# Warp Scheduling and latency tolerance

- When warps are assigned to a SM, there are generally more threads assigned to the SM than available in reality (i.e. the number of cores in the SM)
- Why have more warps assigned than the number of SMs in the first place?
  - To allow the GPU to tolerate long latency operations such as global mem. accesses.
- "GPU SMs achieves zero-overhead scheduling by holding all the execution states for the assigned warps in the hardware registers so there is no need to save and restore states when switching from one warp to another."

# Resource Partitioning and occupancy

- $occupancy = (number\_of\_warps\_assigned\_to\_SM) / (max\_number\_supported\_by\_the\_SM)$
- SMs support Dynamic Partitioning of Resources.
- Underutilization of resources during Dynamic partitioning
  - "In the example of the Ampere A100, we saw that the block size can be varied from 1024 to 64, resulting in 2 32 blocks per SM, respectively. In all these cases, the total number of threads assigned to the SM is 2048, which maximizes occupancy. Consider, however, the case when each block has 32 threads. In this case, the 2048 thread slots would need to be partitioned and assigned to 64 blocks. However, the Volta SM can support only 32 blocks slots at once. This means that only 1024 of the thread slots will be utilized, that is, 32 blocks with 32 threads each. The occupancy in this case is (1024 assigned threads)/(2048 maximum threads) = 50%."
  - "Another situation that could negatively affect occupancy occurs when the maximum number of threads per block is not divisible by the block size. In the example of the Ampere A100, we saw that up to 2048 threads per SM can be supported. However, if a block size of 768 is selected, the SM will be able to accommodate only 2 thread blocks (1536 threads), leaving 512 thread slots unutilized. In this case, neither the maximum threads per SM nor the maximum blocks per SM are reached. The occupancy in this case is (1536 assigned threads)/(2,048 maximum threads) = 75%."
- By dynamically partitioning registers in an SM across threads, the SM can accommodate many blocks if they require few registers per thread and fewer blocks if they require more registers per thread.
- "**be aware of potential impact of register resource limitations on occupancy**.
  - For example, the Ampere A100 GPU allows a maximum of 65,536 registers per SM. To run at full occupancy, each SM needs enough registers for 2048 threads, which means that each thread should not use more than (65,536 registers)/(2048 threads) = 32 registers per thread. For example, if a kernel uses 64 registers per thread, the maximum number of threads that can be supported with 65,536 registers is 1024 threads. In this case, the kernel cannot run with full occupancy regardless of what the block size is set to be. Instead, the occupancy will be at most 50%."
- “performance cliff,” - a slight increase in resource usage can result in significant reduction in parallelism and performance achieved. (e.g. using 33 registers when say each thread should ideally use 32 registers per thread (calc. using tot_reg / tot_threads))
- CUDA Occupancy Calculator: https://docs.nvidia.com/cuda/cuda-occupancy-calculator/index.html

# Querying Device Properties

- `cudaGetDeviceCount` returs the number of available CUDA devices in the system.
  - Usage:
    - `int deviceCount; cudaGetDeviceCount(&deviceCount);`
- ````
  cudaDeviceProp devProp;
  for (unsigned int i = 0; i < devCount; i++){
    cudaGetDeviceProperties(&devProp, i);
  	// decide if device has suggicient resources/capabilities.
  }```
  - all devices in the system are labelled from 0 to devCount-1 at runtime.
  - max threads allowed in a block in the queried device:
  	- `devProp.maxThreadsPerBlock`
  - number of SMs in the device
  	- `devProp.multiProcessorCount`
  - clock freq. of the device
  	- `devProp.clockRate`
  - max threads allowed along each dimension of a block
  	- `devProp.maxTreadsDim[0]` - x axis
  	- `devProp.maxTreadsDim[1]` - y axis
  	- `devProp.maxTreadsDim[2]` - z axis
  - Max number of blocks allowed along each dimension of a grid:
  	- `devProp.maxGridSize[0]` - x axis
  	- `devProp.maxGridSize[1]` - y axis
  	- `devProp.maxGridSize[2]` - z axis
  - number of registers tat are available in each SM
  	- `devProp.regsPerBlock`
  		- useful in determinig whether the kerel can achieve max. occupancy on a particular device or will be limited by its register usage.
  		- NOTE:
  			- "For most compute capability levels, the maximum number of registers that a block can use is indeed the same as the total number of registers that are available in the SM. However, for some compute capability levels, the maximum number of registers that a block can use is less than the total that are available on the SM."
  - size of the warps
  	- `devProp.warpSize`
  ````

etc.
