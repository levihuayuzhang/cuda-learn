/**
 * Single-Precision A·X Plus Y in C
 * 
 * 2012: https://developer.nvidia.com/blog/six-ways-saxpy/
 * 2021: https://developer.nvidia.com/blog/n-ways-to-saxpy-demonstrating-the-breadth-of-gpu-programming-options/
 * 
 * */

#include <stdlib.h>
#include <stdio.h>

float* saxpy_cpu(int n, float a, float *x, float *y)
{
	for (int i = 0; i < n; ++i)
		y[i] = a * x[i] + y[i];

    return y;
}

int main(void) {
    const int n = 1<<15;

    float x[n];
    float y[n];

    for (int i = 0; i < n; i++) {
        x[i] = (float)rand()/(float)(RAND_MAX/3);
        y[i] = (float)rand()/(float)(RAND_MAX/2);
    }

    float* res = saxpy_cpu(n, 2.0, x, y);

    for (int i = 0; i < n; i++) {
        if (i == 0)
            printf("%d: %f\n", i, res[i]);
    }

    return 0;
}
