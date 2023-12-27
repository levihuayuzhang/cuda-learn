/**
 * @file matrix_muti.cu
 * @author Huayu Zhang (zhanghuayu.dev@gmail.com)
 * 
 * @brief 
 * Matrix multi
 * 
 * Reference: https://zhuanlan.zhihu.com/p/34587739?utm_id=0
 * @version 0.1
 * @date 2023-10-29
 * 
 */
#include <iostream>
// #include <cuda.h>
#include <cuda_runtime.h>

struct Matrix
{
    int width;
    int height;
    float *elements;
};

__device__ float getElement(Matrix *A, int row, int col) {
    return A->elements[row * A->width + col];
}

__device__ void setElement(Matrix *A, int row, int col, float value) {
    A->elements[row * A->width + col] = value;
}

__global__ void matrixMulti(Matrix *A, Matrix *B, Matrix *C) {
    float CVal = 0.0; 

    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int column = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = 0; i < A->width; ++i) {
        CVal += getElement(A, row, i) * getElement(B, i, column);
    }

    setElement(C, row, column, CVal);
}


int main (void) {
    int width = 1<<10;
    int height = 1<<10;
    Matrix *A, *B, *C;

    cudaMallocManaged((void**)&A, sizeof(Matrix));
    cudaMallocManaged((void**)&B, sizeof(Matrix));
    cudaMallocManaged((void**)&C, sizeof(Matrix));

    int nBytes = width*height*sizeof(float);
    cudaMallocManaged((void**)&A->elements, nBytes);
    cudaMallocManaged((void**)&B->elements, nBytes);
    cudaMallocManaged((void**)&C->elements, nBytes);

    A->height = height;
    A->width = width;
    B->height = height;
    B->width = width;
    C->height = height;
    C->width = width;
    for (int i = 0; i < width * height; ++i)
    {
        A->elements[i] = 1.0;
        B->elements[i] = 2.0;
    }

    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
                    (height + blockSize.y - 1) / blockSize.y);

    matrixMulti<<<gridSize, blockSize>>>(A, B, C);

    cudaDeviceSynchronize();

    float maxError= 0.0;
    for (int i = 0; i < width*height; ++i) {
        maxError = fmax(maxError, fabs(C->elements[i] - 2*width));
    }
    std::cout<< "Max Error is: " << maxError << std::endl;

    return 0;
}