# ---------------------------------------------------------------------------
# flash_attention
#
# Layout:
#   flash_attention.h                    public launcher declarations
#   kernels/*.cu                         one translation unit per version
#   benchmark/bench_flash_attention.cu   correctness + TFLOPS benchmark
#
# Same shape as rms_norm and stable_softmax. The three operators share no code:
# they differ in dtype, in what the bottleneck is, and in what their benchmarks
# have to measure, so independent target graphs mean none can break another.
#
# Global settings (CUDA arch, C++/CUDA 17, CUDAToolkit, enable_testing) come from
# the root CMakeLists.txt and are inherited here.
# ---------------------------------------------------------------------------

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

# v0: unfused, three kernels, score matrix materialized in a workspace. Also
# holds attention_flops and unfused_workspace_bytes, which every other target and
# the benchmark need, so it is a dependency of the benchmark regardless of whether
# the v0 kernels are being timed.
add_flash_attention_kernel(flash_attention_v0 kernels/v0_unfused.cu)

# v1: fused, one block per query row, online softmax. Never writes the score
# matrix.
add_flash_attention_kernel(flash_attention_v1 kernels/v1_fused_row.cu)

add_executable(bench_flash_attention
  benchmark/bench_flash_attention.cu
)

target_link_libraries(bench_flash_attention PRIVATE
  flash_attention
  flash_attention_v0
  flash_attention_v1
)

# The benchmark returns non-zero when any kernel drifts past the ULP tolerance
# against the bf16-rounded double reference, so it doubles as a regression test.
add_test(NAME flash_attention_bench
  COMMAND bench_flash_attention
)
