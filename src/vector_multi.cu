/**
 * simple vector multi
 * 
 * Ref:
 * 1. https://zhuanlan.zhihu.com/p/34587739?utm_id=0
 * 2. https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
 * 
*/

#include <iostream>
#include <cuda_runtime.h>

#define N (1 << 20)

__global__ void multi(float *a, float *b, float *c, int n){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n; i+=stride)
        c[i] = a[i] * b[i];


    // if (index < n) {
    //     c[index] = a[index] * b[index];
    // }
}

int main(void) {
    // float *a, *b, *c; // host copies
    float *d_a, *d_b, *d_c; // device copies
    int size = N * sizeof(float); // array size

    // // allocate memory for device copies
    // cudaMalloc((void**)&d_a, size);
    // cudaMalloc((void**)&d_b, size);
    // cudaMalloc((void**)&d_c, size);
    // // memory for host copies
    // a = (float*)malloc(size); 
    // // random_ints(a, N);
    // b = (float*)malloc(size); 
    // // random_ints(b, N);
    // c = (float*)malloc(size);

    cudaMallocManaged(&d_a, size);
    cudaMallocManaged(&d_b, size);
    cudaMallocManaged(&d_c, size);



    for (int i = 0; i < N; ++i)
    {
        // a[i] = 10.0;
        // b[i] = 20.0;
        d_a[i] = 10.0;
        d_b[i] = 20.0;
    }

    // // copy host intput value to device
    // cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);

    multi<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);

    // cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    float maxError = 0.0;
    for (int i = 0; i < N; i++) {
        // maxError = fmax(maxError, fabs(c[i] - 200.0));
        maxError = fmax(maxError, fabs(d_c[i] - 200.0));
    }

    std::cout << "Max Error: " << maxError << std::endl;

    // free(a);
    // free(b);
    // free(c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}