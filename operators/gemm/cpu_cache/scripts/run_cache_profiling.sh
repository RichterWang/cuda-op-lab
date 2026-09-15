#!/bin/bash
# Cache profiling with hardware counters (optional extension)
# Fixed to single P-core, split measurements to avoid counter multiplex
# Usage: ./run_cache_profiling.sh [P_CORE_ID]

set -e

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BUILD_DIR="build/operators/gemm/cpu_cache"
RESULTS_DIR="operators/gemm/cpu_cache/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# CPU core to pin to (default: 0, should be a P-core)
P_CORE=${1:-0}

# Test parameters - only key configurations for cache analysis
REPEATS=3

# Key configurations to profile
declare -a CONFIGS=(
    "512:naive::"
    "512:blocked:64:"
    "512:recursive::32"
    "1024:naive::"
    "1024:blocked:128:"
    "1024:recursive::64"
)

echo -e "${GREEN}=== Cache Profiling with Hardware Counters ===${NC}"
echo "Timestamp: $TIMESTAMP"
echo "Pinned to CPU core: $P_CORE"
echo ""

# Check if perf is available
if ! command -v perf &> /dev/null; then
    echo -e "${RED}Error: perf command not found${NC}"
    echo "Please install linux-tools package"
    exit 1
fi

# Check if executables exist
if [ ! -f "$BUILD_DIR/perf_naive" ] || [ ! -f "$BUILD_DIR/perf_blocked" ] || [ ! -f "$BUILD_DIR/perf_recursive" ]; then
    echo -e "${RED}Error: Performance executables not found in $BUILD_DIR${NC}"
    echo "Please run 'cmake --build build' first"
    exit 1
fi

# Create results directory
mkdir -p "$RESULTS_DIR"
PROFILE_FILE="$RESULTS_DIR/cache_profile_${TIMESTAMP}.txt"

echo "Results will be saved to: $PROFILE_FILE"
echo ""

# Check perf permissions
echo "Checking perf permissions..."
PARANOID=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo "unknown")
if [ "$PARANOID" != "-1" ] && [ "$PARANOID" != "0" ] && [ "$PARANOID" != "1" ]; then
    echo -e "${YELLOW}Warning: perf_event_paranoid is set to $PARANOID${NC}"
    echo "You may need to run with sudo or adjust permissions:"
    echo "  sudo sysctl -w kernel.perf_event_paranoid=-1"
    echo ""
fi

# Redirect output to file
exec > >(tee -a "$PROFILE_FILE")
exec 2>&1

echo "=== Cache Profiling Configuration ==="
echo "Timestamp: $TIMESTAMP"
echo "CPU Core: $P_CORE"
echo "Repeats per config: $REPEATS"
echo "Configurations to test: ${#CONFIGS[@]}"
echo ""
echo "========================================"
echo ""

# Process each configuration
for config in "${CONFIGS[@]}"; do
    IFS=':' read -r size algo block_size threshold <<< "$config"
    
    # Determine executable and parameters
    if [ "$algo" == "naive" ]; then
        exe="$BUILD_DIR/perf_naive"
        params="$size"
        config_name="naive"
    elif [ "$algo" == "blocked" ]; then
        exe="$BUILD_DIR/perf_blocked"
        params="$size $block_size"
        config_name="blocked(bs=$block_size)"
    elif [ "$algo" == "recursive" ]; then
        exe="$BUILD_DIR/perf_recursive"
        params="$size $threshold"
        config_name="recursive(t=$threshold)"
    fi
    
    echo -e "${YELLOW}### ${size}x${size} - $config_name ###${NC}"
    echo ""
    
    # Round 1: Basic execution metrics
    echo "--- Round 1: cycles, instructions, L1-dcache-loads ---"
    taskset -c $P_CORE perf stat -e cycles,instructions,L1-dcache-loads \
        --repeat $REPEATS \
        $exe $params 2>&1
    echo ""
    
    # Round 2: Cache miss metrics
    echo "--- Round 2: L1-dcache-load-misses, LLC-load-misses ---"
    taskset -c $P_CORE perf stat -e L1-dcache-load-misses,LLC-load-misses,cache-references,cache-misses \
        --repeat $REPEATS \
        $exe $params 2>&1
    echo ""
    
    echo "========================================"
    echo ""
done

echo -e "${GREEN}=== Cache profiling completed ===${NC}"
echo ""
echo "Analysis summary:"
echo "  - Configurations tested: ${#CONFIGS[@]}"
echo "  - Measurements per config: 2 rounds (avoiding counter multiplex)"
echo "  - Repeats per measurement: $REPEATS"
echo "  - Total perf runs: $((${#CONFIGS[@]} * 2))"
echo ""
echo "Results saved to: ${BLUE}$PROFILE_FILE${NC}"
echo ""
echo "Next steps:"
echo "  1. Review cache miss rates"
echo "  2. Compare L1 vs LLC miss patterns"
echo "  3. Correlate with theoretical locality analysis"
