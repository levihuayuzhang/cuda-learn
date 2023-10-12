#include <stdio.h>
#include <iostream>
#include <cuda.h>
#define N 512

__global__ void multi(int *a, int *b, int *c){
    // c[blockIdx.x] = a[blockIdx.x] * b[blockIdx.x];
    c[threadIdx.x] = a[threadIdx.x] * b[threadIdx.x];
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

    // copy host intput value to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // multi<<<N,1>>>(d_a, d_b, d_c); // launch kernel on GPU with N blocks
    multi<<<1,N>>>(d_a, d_b, d_c); // launch kernel on GPU with N threads

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        printf("Result for c[%d] is %d.\n", i, c[i]);
    }

    free(a);
    free(b);
    free(c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}