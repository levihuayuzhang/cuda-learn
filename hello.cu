#include <stdio.h>
__global__ void multi(int *a, int *b, int *res){
    *res = (*a) * (*b);
}

int main(void) {
    int a, b, c; // host copies
    int *d_a, *d_b, *d_c; // device copies
    int size = sizeof(int);

    // allocate memory for device copies
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    a= 3;
    b= 4;

    // copy host intput value to device
    cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);

    multi<<<1,1>>>(d_a, d_b, d_c);

    cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    printf("Hello world! Result is %d", c);

    return 0;
}