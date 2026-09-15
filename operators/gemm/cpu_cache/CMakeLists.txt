cmake_minimum_required(VERSION 3.10)

# add source file
set(CPU_CACHE_SOURCES
    src/naive.cpp
    src/blocked.cpp
    src/recursive.cpp
    src/utils.cpp
)

# create static library
add_library(cpu_cache STATIC ${CPU_CACHE_SOURCES})

# set include dictory
target_include_directories(cpu_cache PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# set compile setting
target_compile_options(cpu_cache PRIVATE
    -Wall
    -Wextra
    -O3
    -march=native
)

# benchmark part:
# add benchmark
add_executable(bench_cpu_matmul
    benchmark/bench_cpu_matmul.cpp
)

# link library
target_link_libraries(bench_cpu_matmul
    cpu_cache
)

# set benchmark compile option
target_compile_options(bench_cpu_matmul PRIVATE
    -Wall
    -Wextra
    -O3
    -march=native
)

# performance profiling executables
add_executable(perf_naive
    benchmark/perf_naive.cpp
)

add_executable(perf_blocked
    benchmark/perf_blocked.cpp
)

add_executable(perf_recursive
    benchmark/perf_recursive.cpp
)

# link library for perf executables
target_link_libraries(perf_naive cpu_cache)
target_link_libraries(perf_blocked cpu_cache)
target_link_libraries(perf_recursive cpu_cache)

# set compile options for perf executables
target_compile_options(perf_naive PRIVATE -Wall -Wextra -O3 -march=native)
target_compile_options(perf_blocked PRIVATE -Wall -Wextra -O3 -march=native)
target_compile_options(perf_recursive PRIVATE -Wall -Wextra -O3 -march=native)

# set C++ standard
set_target_properties(cpu_cache bench_cpu_matmul perf_naive perf_blocked perf_recursive PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)