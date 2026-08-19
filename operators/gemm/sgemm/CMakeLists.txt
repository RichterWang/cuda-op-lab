add_library(sgemm INTERFACE)
# create to manage public header and dependencies

target_include_directories(sgemm INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}
) # find headers in this directory

target_link_libraries(sgemm INTERFACE
  CUDA::cudart
  CUDA::cublas
) # link this library

add_executable(bench_sgemm
  benchmark/bench_sgemm.cu
) # add executeable file as entry

# target link library, each time need modify here
target_link_libraries(bench_sgemm PRIVATE 
  sgemm
  sgemm_v0
  sgemm_v1
  sgemm_v2
  sgemm_v3
)

# sgemm naive add and link
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

#sgemm shared memory add and link
add_library(sgemm_v1 STATIC
  kernels/shared_mem.cu
)

target_include_directories(sgemm_v1 PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(sgemm_v1 PUBLIC
  CUDA::cudart
)

set_target_properties(sgemm_v1 PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

# sgemm tiled register add and link
add_library(sgemm_v2 STATIC
  kernels/register_tiled.cu
)

target_include_directories(sgemm_v2 PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(sgemm_v2 PUBLIC
  CUDA::cudart
)

set_target_properties(sgemm_v2 PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

# sgemm vector register add and link
add_library(sgemm_v3 STATIC
  kernels/vector_register.cu
)

target_include_directories(sgemm_v3 PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(sgemm_v3 PUBLIC
  CUDA::cudart
)

set_target_properties(sgemm_v3 PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)