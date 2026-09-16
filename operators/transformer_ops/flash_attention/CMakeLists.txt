add_library(flash_attention INTERFACE)

target_include_directories(flash_attention INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/common/include
)

target_link_libraries(flash_attention INTERFACE
  CUDA::cudart
)

function(add_flash_attention_kernel target_name source_file)
  add_library(${target_name} STATIC ${source_file})

  target_link_libraries(${target_name} PUBLIC
    flash_attention
  )

  set_target_properties(${target_name} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
  )
endfunction()

# v0: unfused, three kernels, score matrix materialized in a workspace. 
add_flash_attention_kernel(flash_attention_v0 kernels/v0_unfused.cu)

# v1: fused, one block per query row, online softmax
add_flash_attention_kernel(flash_attention_v1 kernels/v1_fused_row.cu)

add_executable(bench_flash_attention
  benchmark/bench_flash_attention.cu
)

target_link_libraries(bench_flash_attention PRIVATE
  flash_attention
  flash_attention_v0
  flash_attention_v1
)

add_test(NAME flash_attention_bench
  COMMAND bench_flash_attention
)
