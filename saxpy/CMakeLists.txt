# cmake_minimum_required(VERSION 3.22)
# project(cuda-learn CXX C)
# enable_language(CUDA)

# # Find CUDA
# find_package(CUDA QUIET)
# if (CUDA_FOUND)
#     find_package(CUDA REQUIRED)
#     message(STATUS "CUDA Version: ${CUDA_VERSION_STRINGS}")
#     message(STATUS "CUDA Libararies: ${CUDA_LIBRARIES}")
# #     set(
# #         CUDA_NVCC_FLAGS
# #         ${CUDA_NVCC_FLAGS};
# #         -O3
# #         -gencode arch=compute_30,code=sm_30
# #         -gencode arch=compute_35,code=sm_35
# #         -gencode arch=compute_50,code=[sm_50,compute_50]
# #         -gencode arch=compute_52,code=[sm_52,compute_52]
# #         -gencode arch=compute_61,code=sm_61
# #         -gencode arch=compute_62,code=sm_62
# #   )
# #   add_definitions(-DGPU)
# else()
#   list(APPEND LIBRARIES "m")
# endif()

# find_package(CUDAToolkit REQUIRED)
# set(CUDA_RUNTIME_LIBRARY Shared)

# set(CMAKE_CUDA_STANDARD 17)
# set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# # CUDA Arch version
# set(CMAKE_CUDA_ARCHITECTURES 89)

# enable debug device code
# https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html
# if(CMAKE_BUILD_TYPE STREQUAL "Debug")
#     set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")
# endif()
# # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")

# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -m64")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -v")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")

# # define files
# file(GLOB cpu_source_files "${CMAKE_SOURCE_DIR}/src/*.cc")
# file(GLOB gpu_source_files "${CMAKE_SOURCE_DIR}/src/*.cu")
# add_executable(${PROJECT_NAME})
# target_sources(${PROJECT_NAME} PRIVATE ${cpu_source_files} ${gpu_source_files}) 

add_executable(saxpu-cpu saxpy.c)
add_executable(saxpu-cuda saxpy_cuda.cu)
