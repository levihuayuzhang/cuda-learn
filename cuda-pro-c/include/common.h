// #include <sys/time.h>

#ifndef _COMMON_H
#define _COMMON_H

// check cuda error if exist when calling cuda runtime function
#define CHECK(call)                                                                         \
{                                                                                           \
    const cudaError_t error = call;                                                         \
    if (error != cudaSuccess)                                                               \
    {                                                                                       \
        fprintf(stderr, "Error at: %s:%d, ", __FILE__, __LINE__);                           \
        fprintf(stderr, "error code: %d, reason: %s\n", error, cudaGetErrorString(error));  \
        exit(1);                                                                            \
    }                                                                                       \
}                                                                                           \

// inline double seconds()
// {
//     struct timeval tp;
//     struct timezone tzp;
//     int i = gettimeofday(&tp, &tzp);
//     return (double)tp.tv_sec + (double)tp.tv_usec * 1.e-6;
// }

#endif