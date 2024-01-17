#include <cuda_runtime.h>
#include <iostream>

int cpuRecursiceReduce(int *data, int size) {
  if (size == 1) {
    return data[0]; // finished
  }

  int const stride = size / 2;
  for (int i = 0; i < stride; ++i) {
    data[i] = data[i] + data[i+stride];
  }

  return cpuRecursiceReduce(data, stride);
}


/**
 * @brief Dynamic Parallelism implementation
 *
 */

int main(int argc, char const *argv[]) {
  int dev = 0, gpu_sum;
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);
  std::cout << argv[0] << " is starting reduction at " << deviceProp.name
            << std::endl;
  cudaSetDevice(dev);

  bool bResult = false;

  int nblock = 2048;
  int nthread = 512; // init block size

  if (argc > 1) nblock = atoi(argv[1]);
  if (argc > 2) nthread = atoi(argv[2]);

  int size = nblock * nthread; // total number of elements
  dim3 block(nthread, 1);
  dim3 grid(size + block.x - 1, 1);
  printf("array %d, grid %d, block %d\n", size, grid.x, block.x);

  size_t bytes = size * sizeof(int);
  int *h_idata = (int *)malloc(bytes);
  int *h_odata = (int *)malloc(grid.x * sizeof(int));
  int *tmp = (int *)malloc(bytes); // for host computation

  for (int i = 0; i < size; ++i) {
    
  }





  return 0;
}
