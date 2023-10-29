#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// void random_floats(float* a, int count) {
//     for (int i=0; i < count; i++){
//         a[i] = rand() % 1000;
//     }
// }

__global__ void saxpy_cuda(int n, float a, float *x, float *y) {
    int t_id = threadIdx.x + blockIdx.x * blockDim.x; // thread index
    int stride = blockDim.x * gridDim.x;
    for (int i = t_id; i < n; i+= stride) {
        y[i] = x[i] + y[i];
    }
}

int main(void) {
    const int N = 1UL << 25;
    int size = N * sizeof(float);

    float *x, *y, alpha=2.0;

    // random_floats(x, N);
    // random_floats(y, N);
    cudaMalloc(&x, size);
    cudaMalloc(&y, size);

    saxpy_cuda<<<32, 1024>>>(N, alpha, x, y);
    cudaDeviceSynchronize();

    cudaFree(x);
    cudaFree(y);


}
