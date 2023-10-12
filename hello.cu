#include <stdio.h>
#include <iostream>
#include <cuda.h>
#define N 512

__global__ void multi(int *a, int *b, int *c){
    c[blockIdx.x] = a[blockIdx.x] * b[blockIdx.x];
}

//populate vectors with random ints
void random_ints(int* a, int count) {
    for (int i=0; i < count; i++){
        a[i] = rand() % 1000;
    }
}

int main(void) {
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

    multi<<<N,1>>>(d_a, d_b, d_c); // launch kernel on GPU with N blocks

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