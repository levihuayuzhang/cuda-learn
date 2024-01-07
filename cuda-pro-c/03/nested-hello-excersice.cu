/**
 * professional cuda c
 * chapter 03 Figure 3-30 exersice
*/
#include <iostream>
#include <cuda_runtime.h>

/**
 * @brief 
 * 
 * @param iSize initial block size
 * @param iDepth the recursion depth
 * @return __global__ 
 */
__global__ void nestedHelloWorld(int const iSize, int iDepth) {
    int tid = threadIdx.x;
    printf("Recursion=%d: Hello World from thread %d block %d\n", 
        iDepth, tid, blockIdx.x);

    if (iSize == 1) return;
    
    int nthreads = iSize >> 1;
    
    if (tid == 0 && blockIdx.x == 0 && nthreads > 0) {
        nestedHelloWorld<<<2, nthreads>>>(nthreads, ++iDepth);
        printf("-------> nested execution depth: %d, %d nthreads per block\n", iDepth, nthreads);
    }
    
}

int main(int argc, char const *argv[])
{
    int size = 8; // total threads
    int blocksize = 8; // init block size
    int igrid = 1; // init grid size

    if (argc > 1) {
        igrid = atoi(argv[1]);
        size = igrid * blocksize;
    }

    dim3 block (blocksize, 1);
    dim3 grid ((size + block.x - 1) / block.x, 1);
    std::cout << argv[0] << " Excecution Confihuration: grid " <<
        grid.x << " block " << block.x << "\n";
    
    nestedHelloWorld<<<grid, block>>>(block.x, 0);
    
    cudaGetLastError();
    cudaDeviceReset();
    return 0;
}
