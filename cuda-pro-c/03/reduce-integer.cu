#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include "common.h"

int recursiveReduce(int *data, int const size)
{
    if (size == 1) return data[0];
    int const stride = size/2;
    
    for (int i = 0; i < stride; i++)
    {
        data[i] += data[i + stride];
    }

    return recursiveReduce(data, stride);
}



int main(int argc, char const *argv[])
{
    int dev = 0;
    cudaDeviceProp devicePro;
    cudaGetDeviceProperties(&devicePro, dev);
    std::cout << argv[0] << "starting reduction at " << std::endl;
    std::cout << "device " << dev << ": " << devicePro.name << std::endl;

    bool bRsult = false;
    int size  = 1 << 24;
    std::cout << "With arry size of " << size << std::endl;
    int blockSize = 512;
    
    if (argc > 1) blockSize = atoi(argv[1]);

    dim3 block (blockSize, 1);
	dim3 grid ((size + block.x - 1) / block.x, 1);
    std::cout << "Grid  " << grid.x << " Block " << block.x << std::endl;
	
    // host mem
	size_t bytes = size * sizeof(int);
    int *h_idata = (int *) malloc(bytes);
    int *h_odata = (int *) malloc(grid.x * sizeof(int));
    int *tmp = (int *) malloc(bytes);

    for (int i = 0; i < size; i++)
    {
        h_idata[i] = (int)(rand() *0xFF);
    }
    
    memcpy(tmp, h_idata, bytes);

    double iStart, iElaps;
    int gpu_sum = 0;

    int *d_idata = NULL;
    int *d_odata = NULL;
    cudaMalloc((void **) &d_idata, bytes);
    cudaMalloc((void **) &d_idata, grid.x * sizeof(int));

    iStart = seconds();
    int cpu_sum = recursiveReduce(tmp, size);
    iElaps = seconds() - iStart;
    std::cout << "CPU reduce recusice: " << iElaps << "sec cpu_sum: " 
                << cpu_sum << std::endl;

    

    std::cout << "end!!!" << std::endl;
    return 0;
}
