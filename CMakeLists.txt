cmake_minimum_required(VERSION 3.22)

project(cuda_op_lab LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86)

include(cmake/CompilerFlags.cmake)
find_package(CUDAToolkit REQUIRED)

enable_testing()

add_subdirectory(operators/gemm/sgemm)