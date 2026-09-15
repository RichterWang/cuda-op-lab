#!/bin/bash
# Basic performance testing for CPU matrix multiplication
# Measures time, GFLOPS, and speedup - satisfies core course requirements
# Usage: ./run_basic_tests.sh [--quiet]

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

# Test parameters
MATRIX_SIZES=(128 256 512 1024 300 500 768)
BLOCK_SIZES=(16 32 64 128)
THRESHOLDS=(16 32 64 128)
REPEATS=5

# Quiet mode
QUIET=false
if [[ "$1" == "--quiet" ]]; then
    QUIET=true
fi

log() {
    if [[ "$QUIET" == false ]]; then
        echo -e "$@"
    fi
}

log "${GREEN}=== Basic Performance Testing ===${NC}"
log "Timestamp: $TIMESTAMP"
log ""

# Check if executables exist
if [ ! -f "$BUILD_DIR/perf_naive" ] || [ ! -f "$BUILD_DIR/perf_blocked" ] || [ ! -f "$BUILD_DIR/perf_recursive" ]; then
    echo -e "${RED}Error: Performance executables not found in $BUILD_DIR${NC}"
    echo "Please run 'cmake --build build' first"
    exit 1
fi

# Create results directory
mkdir -p "$RESULTS_DIR"
JSON_FILE="$RESULTS_DIR/basic_perf_${TIMESTAMP}.json"
CSV_FILE="$RESULTS_DIR/basic_perf_${TIMESTAMP}.csv"
LOG_FILE="$RESULTS_DIR/basic_perf_${TIMESTAMP}.log"

log "Results will be saved to:"
log "  JSON: $JSON_FILE"
log "  CSV:  $CSV_FILE"
log "  Log:  $LOG_FILE"
log ""

# Initialize JSON output
echo "{" > "$JSON_FILE"
echo "  \"timestamp\": \"$TIMESTAMP\"," >> "$JSON_FILE"
echo "  \"config\": {" >> "$JSON_FILE"
echo "    \"matrix_sizes\": [${MATRIX_SIZES[@]}]," >> "$JSON_FILE"
echo "    \"block_sizes\": [${BLOCK_SIZES[@]}]," >> "$JSON_FILE"
echo "    \"thresholds\": [${THRESHOLDS[@]}]," >> "$JSON_FILE"
echo "    \"repeats\": $REPEATS" >> "$JSON_FILE"
echo "  }," >> "$JSON_FILE"
echo "  \"results\": [" >> "$JSON_FILE"

# Initialize CSV output
echo "matrix_size,algorithm,parameter,time_ms,gflops,speedup" > "$CSV_FILE"

# Function to run test and extract time and GFLOPS
run_test() {
    local exe=$1
    local size=$2
    local param=$3
    local times=()
    local gflops_values=()
    
    for ((i=1; i<=$REPEATS; i++)); do
        if [ -n "$param" ]; then
            output=$($exe $size $param 2>&1)
        else
            output=$($exe $size 2>&1)
        fi
        
        # Extract time in milliseconds and GFLOPS from output
        time_ms=$(echo "$output" | grep -oP 'Average time: \K[0-9.]+' || echo "0")
        gflops=$(echo "$output" | grep -oP 'Performance:\s+\K[0-9.]+' || echo "0")
        
        times+=($time_ms)
        gflops_values+=($gflops)
    done
    
    # Calculate average time
    sum_time=0
    for t in "${times[@]}"; do
        sum_time=$(echo "$sum_time + $t" | bc -l)
    done
    avg_time=$(echo "scale=4; $sum_time / $REPEATS" | bc -l)
    
    # Calculate average GFLOPS
    sum_gflops=0
    for g in "${gflops_values[@]}"; do
        sum_gflops=$(echo "$sum_gflops + $g" | bc -l)
    done
    avg_gflops=$(echo "scale=4; $sum_gflops / $REPEATS" | bc -l)
    
    # Return both values separated by colon
    echo "$avg_time:$avg_gflops"
}

first_result=true

# Test each matrix size
for size in "${MATRIX_SIZES[@]}"; do
    log "${YELLOW}### Testing matrix size: ${size}x${size} ###${NC}"
    
    if [ "$first_result" = false ]; then
        echo "    }," >> "$JSON_FILE"
    fi
    first_result=false
    
    echo "    {" >> "$JSON_FILE"
    echo "      \"matrix_size\": $size," >> "$JSON_FILE"
    
    # Test naive algorithm
    log "  Testing naive algorithm..."
    result=$(run_test "$BUILD_DIR/perf_naive" "$size" "")
    naive_time=$(echo "$result" | cut -d: -f1)
    naive_gflops=$(echo "$result" | cut -d: -f2)
    
    log "    Time: ${naive_time} ms, GFLOPS: ${naive_gflops}"
    
    echo "      \"naive\": {" >> "$JSON_FILE"
    echo "        \"time_ms\": $naive_time," >> "$JSON_FILE"
    echo "        \"gflops\": $naive_gflops" >> "$JSON_FILE"
    echo "      }," >> "$JSON_FILE"
    
    echo "$size,naive,N/A,$naive_time,$naive_gflops,1.0" >> "$CSV_FILE"
    
    # Test blocked algorithm with different block sizes
    log "  Testing blocked algorithm..."
    echo "      \"blocked\": [" >> "$JSON_FILE"
    
    first_block=true
    for bs in "${BLOCK_SIZES[@]}"; do
        result=$(run_test "$BUILD_DIR/perf_blocked" "$size" "$bs")
        blocked_time=$(echo "$result" | cut -d: -f1)
        blocked_gflops=$(echo "$result" | cut -d: -f2)
        speedup=$(echo "scale=4; $naive_time / $blocked_time" | bc -l)
        
        log "    block_size=$bs: Time=${blocked_time} ms, GFLOPS=${blocked_gflops}, Speedup=${speedup}x"
        
        if [ "$first_block" = false ]; then
            echo "        }," >> "$JSON_FILE"
        fi
        first_block=false
        
        echo "        {" >> "$JSON_FILE"
        echo "          \"block_size\": $bs," >> "$JSON_FILE"
        echo "          \"time_ms\": $blocked_time," >> "$JSON_FILE"
        echo "          \"gflops\": $blocked_gflops," >> "$JSON_FILE"
        echo "          \"speedup\": $speedup" >> "$JSON_FILE"
        
        echo "$size,blocked,$bs,$blocked_time,$blocked_gflops,$speedup" >> "$CSV_FILE"
    done
    echo "        }" >> "$JSON_FILE"
    echo "      ]," >> "$JSON_FILE"
    
    # Test recursive algorithm with different thresholds
    log "  Testing recursive algorithm..."
    echo "      \"recursive\": [" >> "$JSON_FILE"
    
    first_recursive=true
    for t in "${THRESHOLDS[@]}"; do
        result=$(run_test "$BUILD_DIR/perf_recursive" "$size" "$t")
        recursive_time=$(echo "$result" | cut -d: -f1)
        recursive_gflops=$(echo "$result" | cut -d: -f2)
        speedup=$(echo "scale=4; $naive_time / $recursive_time" | bc -l)
        
        log "    threshold=$t: Time=${recursive_time} ms, GFLOPS=${recursive_gflops}, Speedup=${speedup}x"
        
        if [ "$first_recursive" = false ]; then
            echo "        }," >> "$JSON_FILE"
        fi
        first_recursive=false
        
        echo "        {" >> "$JSON_FILE"
        echo "          \"threshold\": $t," >> "$JSON_FILE"
        echo "          \"time_ms\": $recursive_time," >> "$JSON_FILE"
        echo "          \"gflops\": $recursive_gflops," >> "$JSON_FILE"
        echo "          \"speedup\": $speedup" >> "$JSON_FILE"
        
        echo "$size,recursive,$t,$recursive_time,$recursive_gflops,$speedup" >> "$CSV_FILE"
    done
    echo "        }" >> "$JSON_FILE"
    echo "      ]" >> "$JSON_FILE"
    
    log ""
done

# Close JSON
echo "    }" >> "$JSON_FILE"
echo "  ]" >> "$JSON_FILE"
echo "}" >> "$JSON_FILE"

log "${GREEN}=== All tests completed ===${NC}"
log ""
log "Summary:"
log "  - Tested ${#MATRIX_SIZES[@]} matrix sizes"
log "  - Tested ${#BLOCK_SIZES[@]} block sizes"
log "  - Tested ${#THRESHOLDS[@]} thresholds"
log "  - Each configuration repeated $REPEATS times"
log "  - Total tests: $((${#MATRIX_SIZES[@]} * (1 + ${#BLOCK_SIZES[@]} + ${#THRESHOLDS[@]})))"
log ""
log "Results saved to:"
log "  ${BLUE}$JSON_FILE${NC}"
log "  ${BLUE}$CSV_FILE${NC}"
