# ---------------------------------------------------------------------------
# stable_softmax
#
# Layout:
#   stable_softmax.h            public launcher declarations
#   kernels/*.cu                one translation unit per optimization version
#   benchmark/*.cu              correctness + bandwidth benchmark entry point
#
# Global settings (CUDA arch, C++/CUDA 17, CUDAToolkit, enable_testing) come
# from the root CMakeLists.txt and are inherited here, so they are not repeated.
# ---------------------------------------------------------------------------

# INTERFACE target: single place to manage public headers and common deps.
add_library(stable_softmax INTERFACE)

target_include_directories(stable_softmax INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/common/include
)

target_link_libraries(stable_softmax INTERFACE
  CUDA::cudart
)

# Helper to remove the per-kernel boilerplate. Each kernel library links the
# INTERFACE target PUBLIC-ly, so it inherits include dirs and cudart, and
# propagates them to whoever links the kernel library.
function(add_softmax_kernel target_name source_file)
  add_library(${target_name} STATIC ${source_file})

  target_link_libraries(${target_name} PUBLIC
    stable_softmax
  )

  set_target_properties(${target_name} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
  )
endfunction()

# v0: baseline, one block per row, shared-memory tree reduction, three passes.
add_softmax_kernel(stable_softmax_v0 kernels/v0_naive.cu)

# v1: same structure as v0, reductions moved to __shfl_xor_sync.
add_softmax_kernel(stable_softmax_v1 kernels/v1_warp_reduce.cu)

# v2: one row per lane group + float4, no barriers. Links v1 because its
# dispatcher forwards wide rows to launch_softmax_adaptive rather than
# duplicating a block-per-row path.
add_softmax_kernel(stable_softmax_v2 kernels/v2_subwarp.cu)
target_link_libraries(stable_softmax_v2 PUBLIC stable_softmax_v1)

# v3: online softmax, one fused (max, sum) pass instead of two separate ones.
add_softmax_kernel(stable_softmax_v3 kernels/v3_online.cu)

# Not a softmax: plain copy, measures achievable bandwidth as the real roofline.
add_softmax_kernel(stable_softmax_copy kernels/copy_baseline.cu)

# Not a softmax: 1-read vs 3-read probes, used to measure how much of the extra
# read traffic actually reaches DRAM.
add_softmax_kernel(stable_softmax_probe kernels/traffic_probe.cu)

# Benchmark executable. Declared after the kernel libraries so the dependency
# direction reads top-down.
add_executable(bench_stable_softmax
  benchmark/bench_stable_softmax.cu
)

target_link_libraries(bench_stable_softmax PRIVATE
  stable_softmax
  stable_softmax_v0
  stable_softmax_v1
  stable_softmax_v2
  stable_softmax_v3
  stable_softmax_copy
)

# The benchmark returns non-zero when any accuracy check fails, so it doubles
# as a regression test under ctest.
add_test(NAME stable_softmax_bench
  COMMAND bench_stable_softmax
)

# Diagnostic tool, not a test: verifies the DRAM-traffic and occupancy claims
# that the README derives from the bandwidth numbers. Deliberately kept out of
# ctest since it reports measurements rather than pass/fail.
add_executable(measure_bottleneck
  benchmark/measure_bottleneck.cu
)

target_link_libraries(measure_bottleneck PRIVATE
  stable_softmax
  stable_softmax_v1
  stable_softmax_v2
  stable_softmax_copy
  stable_softmax_probe
)
