cmake_minimum_required(VERSION 3.22)
# project(snipts CXX C)
# enable_language(CUDA)

# find_package(CUDAToolkit REQUIRED)
# set(CUDA_RUNTIME_LIBRARY Shared)

# set(CMAKE_CUDA_STANDARD 17)
# set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# # CUDA Arch version
# set(CMAKE_CUDA_ARCHITECTURES 89)

# if(CMAKE_BUILD_TYPE STREQUAL "Debug")
#     set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")
# endif()
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")

# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -m64")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -v")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")

# # define files
# file(GLOB cpu_source_files "${CMAKE_SOURCE_DIR}/src/*.cc")
# file(GLOB gpu_source_files "${CMAKE_SOURCE_DIR}/src/*.cu")
# add_executable(${PROJECT_NAME})
# target_sources(${PROJECT_NAME} PRIVATE ${cpu_source_files} ${gpu_source_files}) 

add_executable(vector_multi)
target_sources(vector_multi PRIVATE vector_multi.cu)
# set_target_properties(vector_multi PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# target_link_libraries(vector_multi ${CUDA_LIBRARIES})

add_executable(matrix_multi)
target_sources(matrix_multi PRIVATE vector_multi.cu) 
