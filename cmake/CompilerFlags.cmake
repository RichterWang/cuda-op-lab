add_compile_options(
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)