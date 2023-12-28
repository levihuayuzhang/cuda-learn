#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include "common.h"
#include <chrono>

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

/**
 * @brief Neighbored pair implementation with divergence
 * 
 * @param g_idata global input data address
 * @param d_odata global output data address
 * @param n data set size
 *
 */
__global__ void reduceNeighbored (int *g_idata, int *g_odata, unsigned int n)
{
    unsigned int tid = threadIdx.x;
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // data index for each bloack (local pointer) in global mem
	int *idata = g_idata + blockIdx.x * blockDim.x;

    // boundary check
	if (idx >= n) return;

	for (int stride = 1; stride < blockDim.x; stride *= 2)
	{
		if ((tid % (2 * stride)) == 0)
		{
			idata[tid] += idata[tid+stride];
		}
		__syncthreads(); // inner block sync (wait for each block to finish)
	}

    // store result from each block (1st elements) to out put array
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
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

    // initialize random data
    for (int i = 0; i < size; i++)
    {
        h_idata[i] = (int)(rand() *0xFF);
    }    
    memcpy(tmp, h_idata, bytes);

    // double iStart, iElaps;
    // device mem
    int *d_idata = NULL;
    int *d_odata = NULL;
    CHECK(cudaMalloc((void **) &d_idata, bytes));
    CHECK(cudaMalloc((void **) &d_odata, grid.x * sizeof(int)));

    // cpu reduction
    auto iStart = std::chrono::high_resolution_clock::now();
    int cpu_sum = recursiveReduce(tmp, size);
    auto endTime = std::chrono::high_resolution_clock::now();
    auto iElaps = std::chrono::duration_cast<std::chrono::microseconds>(endTime-iStart);
    double duration = iElaps.count();
    std::cout << "\nCPU reduce recusice: " << duration << " microsec cpu_sum: " 
                << cpu_sum << std::endl;

    int gpu_sum;
    // kernel 1: reduceNeighboared
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = std::chrono::high_resolution_clock::now();

    reduceNeighbored<<<grid, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();

    endTime = std::chrono::high_resolution_clock::now();
    iElaps = std::chrono::duration_cast<std::chrono::microseconds>(endTime-iStart);
    duration = iElaps.count();
    
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];

    std::cout << "GPU reduce neighbored: " << duration << " microsec gpu_sum: " 
                << gpu_sum << std::endl;
    if (gpu_sum != cpu_sum) std::cout << "Test result failed: sum result not match!" << std::endl;


    




    
    
    
    
    
    
    
    // clean resource
    free(h_idata); free(h_odata);
    cudaFree(d_idata); cudaFree(d_odata);
    cudaDeviceReset();
    

    std::cout << "end!!!" << std::endl;
    return 0;
}
