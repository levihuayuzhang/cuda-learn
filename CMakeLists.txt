cmake_minimum_required(VERSION 3.22)
project(cuda-learn CXX C)
enable_language(CUDA)

find_package(CUDAToolkit REQUIRED)
# set(CUDA_RUNTIME_LIBRARY Shared)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --std=c++11")

# CUDA Arch version
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89)
endif()
# set_property(TARGET hello PROPERTY CUDA_ARCHITECTURES 89)

# enable debug device code
# https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")
endif()
# target_compile_options(hello PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>) # for specific target
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")

# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -m64")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -v")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")

# # define files
# file(GLOB cpu_source_files "${CMAKE_SOURCE_DIR}/src/*.cc")
# file(GLOB gpu_source_files "${CMAKE_SOURCE_DIR}/src/*.cu")
# add_executable(${PROJECT_NAME})
# target_sources(${PROJECT_NAME} PRIVATE ${cpu_source_files} ${gpu_source_files}) 


add_executable(hello)
target_sources(hello PRIVATE ${CMAKE_SOURCE_DIR}/src/hello.cu) 
set_target_properties(hello PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(saxpu-cpu ${CMAKE_SOURCE_DIR}/src/saxpy/saxpu.c)
