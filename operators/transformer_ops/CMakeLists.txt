# transformer_ops aggregator: only forwards to concrete operator directories.
# Adding a new op (layernorm / rmsnorm / flash_attention ...) means adding one
# line here, the root CMakeLists.txt never needs to change.

add_subdirectory(stable_softmax)
add_subdirectory(rms_norm)
add_subdirectory(flash_attention)
