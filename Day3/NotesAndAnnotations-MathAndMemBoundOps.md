# Math and Memory Bound Operations - My notes

- **Math BW**
  - Rate at which math unit operation can be conducted by a processor - expressed in Ops/Second
  - If Floating Point Data - FLOPS
  - Math BW can be queries from HW specification
- **Mem BW**
  - Rate at which data can be read from or stored into semiconductor memory by a processsor, and is usually expressed in units of bytes/second (B/s).
  - Can be queries from HW specification / be computed theoretically
- **Data Reuse**

  - "If some data need to be repeatedly used for a certain operation, it is always a good idea to copy the data to cache to avoid the slow memory access repeatedly. This is often called data reuse."

  - For matrix multiplication - Math ops = 2\*N\*N\*N - Data Read (assuming the value in the matrices is of b bits) = 2\*b\*N\*N\*N

  - Idea with Data Reuse (Perfect Data reuse scenario) - Assume for AxB - All of A and 1 col of B fits in cache and is loaded - Now, $C[i][j] = \sum_{k=0}^{N-1} A[i][k] * B[k][j]$ - Now computing $C[i][j]$ (after A matrix and B column have been loaded into cache/fast mem) takes no reads. - Hence, total number of bits read is = - N\*N bits for reading all of A at the beginning = $b*N^2$ - N\*(N bits read for each column of B as required) = $N * (N*b)$ = $b*N^2$ - which is $2bN^2$. This way you can eliminate any unnecessary data reads. - Total best-case data movement = $2bN^2_{reads} + bN^2_{writes} = 3bN^2_{bits-transferred-in-total}$ - For comparison, the data movement without perfect data reuse comes out to be $2bN^3$

- **Math Bound vs Mem Bound operations**
  - $t_{math} = N_{op} / BW_{math}$
  - $t_{mem} = N_{byte}/BW_{mem}$
  - Math Bound operation - $(N_{op}/N_{byte}) > (BW_{math}/BW_{mem})$
  - Mem Bound Operation - $(N_{op}/N_{byte}) < (BW_{math}/BW_{mem})$
  - Arithmetic Intensity - $N_{op}/N_{byte}$

## Improving Mem-Bound Operations to Math-Bound Operations

- Looking at the earlier data reuse example:
  - Case1: No data reuse - $N_{op}/N_{byte} = (2N^3 / (2bN^3/8)) = 8/b : OP/byte$
  - Case 2: With ideal data reuse - $N_{op}/N_{byte} = (2N^3 / (3bN^2/8)) = 16N/3b : OP/byte$ - This means reusing data reduces memory access and improves the operation performance. A memory-bound operation could be improved to math-bound operation by optimizing the reuse of data in the implementation.
  - Sometimes, enlarging the size of the operands may also improve the memory-bound operation to math-bound operation. - Looking back at the same N×N matrix multiplication example, assuming the datatype is FP32 (b=32). - $N_{op}/N_{byte}=2N^3/(3bN^2/8)=16N/3b=N/6:OP/byte$
  - If the operation is performed on [NVIDIA A100 GPU](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet.pdf), the math bandwidth is 19.5 TFLOPS and the memory bandwidth for FP32 is 1.6 TB/sec. Therefore,
  - $({BW}_{math})/({BW}_{mem}) = 12.2:OP/byte$
  - When N>=74, the matrix multiplication operation is math-bound and when N<74, the N×N matrix multiplication is memory-bound.
- "Although for some operations, it is theoretically possible to turn the operation from memory-bound to math-bound by enlarging the size of the operands, because the reusable data size (cache size) on the hardware is limited, for some operations this strategy might not work in practice."

## References

- [Math-Bound VS Memory-Bound Operations](https://leimao.github.io/blog/Math-Bound-VS-Memory-Bound-Operations/)
