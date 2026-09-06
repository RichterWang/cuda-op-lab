# ---------------------------------------------------------------------------
# rms_norm
#
# Layout:
#   rms_norm.h                  public launcher declarations
#   kernels/*.cu                one translation unit per optimization version
#   benchmark/*.cu              correctness + bandwidth benchmark entry point
#
# Same shape as stable_softmax but a separate set of targets: the two operators
# share no code, differ in storage dtype, and have separate benchmarks, so
# keeping the target graphs independent means neither can break the other.
#
# Global settings (CUDA arch, C++/CUDA 17, CUDAToolkit, enable_testing) come from
# the root CMakeLists.txt and are inherited here.
# ---------------------------------------------------------------------------

# INTERFACE target: single place to manage public headers and common deps.
add_library(rms_norm INTERFACE)

target_include_directories(rms_norm INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/common/include
)

target_link_libraries(rms_norm INTERFACE
  CUDA::cudart
)

# Helper mirroring add_softmax_kernel: each kernel library links the INTERFACE
# target PUBLIC-ly, inheriting include dirs and cudart and propagating them.
function(add_rms_norm_kernel target_name source_file)
  add_library(${target_name} STATIC ${source_file})

  target_link_libraries(${target_name} PUBLIC
    rms_norm
  )

  set_target_properties(${target_name} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
  )
endfunction()

# v0: baseline, one block per row, shared-memory tree reduction, two passes.
# Also holds launch_rmsnorm_3d, which forwards to v1, hence the link below.
add_rms_norm_kernel(rms_norm_v0 kernels/v0_naive.cu)

# v1: warp shuffle reduction plus 128-bit access, applied together.
add_rms_norm_kernel(rms_norm_v1 kernels/v1_warp_vec.cu)

# The 3d entry point lives in v0 but dispatches to v1, so v0 depends on v1.
target_link_libraries(rms_norm_v0 PUBLIC rms_norm_v1)

# v2: row held in registers across the two passes. Falls back to v1 for rows it
# cannot cache, hence the link.
add_rms_norm_kernel(rms_norm_v2 kernels/v2_register_resident.cu)
target_link_libraries(rms_norm_v2 PUBLIC rms_norm_v1)

# Not an RMSNorm: plain bf16 copy, measures the achievable bandwidth ceiling.
add_rms_norm_kernel(rms_norm_copy kernels/copy_baseline.cu)

# Diagnostic, not part of the operator: v1's kernel with block size and occupancy
# exposed as independent knobs, so the two can be varied one at a time.
add_rms_norm_kernel(rms_norm_probe kernels/occupancy_probe.cu)

# Benchmark executable. Declared after the kernel libraries so the dependency
# direction reads top-down.
add_executable(bench_rms_norm
  benchmark/bench_rms_norm.cu
)

target_link_libraries(bench_rms_norm PRIVATE
  rms_norm
  rms_norm_v0
  rms_norm_v1
  rms_norm_v2
  rms_norm_copy
)

# The benchmark returns non-zero when any kernel drifts past the ULP tolerance
# against the bf16-rounded reference, so it doubles as a regression test.
add_test(NAME rms_norm_bench
  COMMAND bench_rms_norm
)

# Occupancy experiment. Its own executable rather than a mode of bench_rms_norm,
# because it produces a one-off decision rather than a regression metric, and it
# is deliberately not registered with add_test for the same reason.
add_executable(measure_occupancy
  benchmark/measure_occupancy.cu
)

target_link_libraries(measure_occupancy PRIVATE
  rms_norm
  rms_norm_probe
  rms_norm_copy
)
