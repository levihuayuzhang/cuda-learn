// #include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define N (2048*2048)
#define THREADS_PER_BLOCK 512

__global__ void multi(int *a, int *b, int *c, int n){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n)
        c[index] = a[index] * b[index];
}

__global__ void helloass(void) {
    printf("hello ass from device..");
}

//populate vectors with random ints
void random_ints(int* a, int count) {
    for (int i=0; i < count; i++){
        a[i] = rand() % 1000;
    }
}

void DisplayHeader()
{
    const int kb = 1024;
    const int mb = kb * kb;

    std::wcout << "CUDA version:   v" << CUDART_VERSION << std::endl;    
    std::wcout << "Thrust version: v" << __CUDA_API_VER_MAJOR__ << "." << __CUDA_API_VER_MINOR__ << std::endl << std::endl; 

    int devCount;
    cudaGetDeviceCount(&devCount);
    std::wcout << "CUDA Devices: " << std::endl << std::endl;

    for(int i = 0; i < devCount; ++i)
    {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, i);
        std::wcout << i << ": " << props.name << ": " << props.major << "." << props.minor << std::endl;
        std::wcout << "  Global memory:   " << props.totalGlobalMem / mb << "mb" << std::endl;
        std::wcout << "  Shared memory:   " << props.sharedMemPerBlock / kb << "kb" << std::endl;
        std::wcout << "  Constant memory: " << props.totalConstMem / kb << "kb" << std::endl;
        std::wcout << "  Block registers: " << props.regsPerBlock << std::endl << std::endl;

        std::wcout << "  Warp size:         " << props.warpSize << std::endl;
        std::wcout << "  Threads per block: " << props.maxThreadsPerBlock << std::endl;
        std::wcout << "  Max block dimensions: [ " << props.maxThreadsDim[0] << ", " << props.maxThreadsDim[1]  << ", " << props.maxThreadsDim[2] << " ]" << std::endl;
        std::wcout << "  Max grid dimensions:  [ " << props.maxGridSize[0] << ", " << props.maxGridSize[1]  << ", " << props.maxGridSize[2] << " ]" << std::endl;
        std::wcout << std::endl;
    }
}

int main(void) {
    // DisplayHeader();
    helloass<<<1,1>>>();

    int *a, *b, *c; // host copies
    int *d_a, *d_b, *d_c; // device copies
    int size = N * sizeof(int); // array size

    // allocate memory for device copies
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);
    // memory for host copies
    a = (int*)malloc(size); random_ints(a, N);
    b = (int*)malloc(size); random_ints(b, N);
    c = (int*)malloc(size);

    printf("malloc end");

    // copy host intput value to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    printf("cuda meme copy");

    multi<<<(N-1+THREADS_PER_BLOCK)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(d_a, d_b, d_c, N);

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // for (int i = 0; i < N; i++) {
    //     printf("Result for c[%d] is %d.\n", i, c[i]);
    // }

    free(a);
    free(b);
    free(c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}