add_library(sgemm INTERFACE)

target_include_directories(sgemm INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(sgemm INTERFACE
  CUDA::cudart
  CUDA::cublas
)

add_executable(bench_sgemm
  benchmark/bench_sgemm.cu
)

target_link_libraries(bench_sgemm PRIVATE 
  sgemm
  sgemm_v0
)

add_library(sgemm_v0 STATIC
  kernels/naive.cu
)

target_include_directories(sgemm_v0 PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(sgemm_v0 PUBLIC
  CUDA::cudart
)

set_target_properties(sgemm_v0 PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)