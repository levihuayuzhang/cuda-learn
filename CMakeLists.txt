cmake_minimum_required(VERSION 3.22)
project(cuda-learn CXX C)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

enable_language(C CXX CUDA)
# https://cmake.org/cmake/help/latest/module/FindCUDAToolkit.html
find_package(CUDAToolkit REQUIRED)
set(CUDA_RUNTIME_LIBRARY Shared)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 89)

# enable debug device code
# https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")
endif()
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo") # use in optimize build, will hide will debugging

# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -m64")
set(CUDA_VERBOSE_BUILD ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -v")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")

# # define files
# file(GLOB cpu_source_files "${CMAKE_SOURCE_DIR}/src/*.cc")
# file(GLOB gpu_source_files "${CMAKE_SOURCE_DIR}/src/*.cu")
# add_executable(${PROJECT_NAME})
# target_sources(${PROJECT_NAME} PRIVATE ${cpu_source_files} ${gpu_source_files}) 

add_subdirectory(snipts)
add_subdirectory(saxpy)
add_subdirectory(cuda-pro-c)
