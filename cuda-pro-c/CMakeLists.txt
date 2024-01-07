add_executable(reduce-integer)
target_sources(reduce-integer PRIVATE 03/reduce-integer.cu)
target_include_directories(reduce-integer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)


add_executable(nested-hello)
target_sources(nested-hello PRIVATE 03/nested-hello-excersice.cu)
# to support dynamic parallelism, set relocatable device code to true, and link the cuda device runtime library, enable the cuda separable compilation
# https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#nvcc-options-for-separate-compilation
target_compile_options(nested-hello PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    --device-c # this equal to -rdc=true --compile
    >)
set_target_properties(nested-hello PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# target_link_libraries(nested-hello cudadevrt) # enabled automatically in CUDA12.3