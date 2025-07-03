#!/bin/bash

# Script to run all benchmarks with different configurations
set -e

echo "=== GPU Benchmarks Runner ==="
echo "Starting benchmarks at $(date)"

# Create results directory if it doesn't exist
mkdir -p /app/results

# Function to run a single benchmark
run_benchmark() {
    local benchmark_name=$1
    local device=$2
    local sizes=${3:-"1000 10000 100000"}
    
    echo "Running benchmark: $benchmark_name on $device"
    echo "Sizes: $sizes"
    
    if [ -d "/app/benchmarks/$benchmark_name" ]; then
        # Run with different sizes
        for size in $sizes; do
            echo "  Running size $size..."
            python /app/run.py "/app/benchmarks/$benchmark_name" \
                --size $size \
                --device $device \
                --repetitions 50 \
                --burnin 3 \
                > "/app/results/${benchmark_name}_${device}_${size}.log" 2>&1 || {
                echo "  Error running $benchmark_name with size $size on $device"
                continue
            }
        done
        
        # Run with all backends
        echo "  Running with all backends..."
        python /app/run.py "/app/benchmarks/$benchmark_name" \
            --size 10000 \
            --device $device \
            --repetitions 100 \
            --burnin 5 \
            > "/app/results/${benchmark_name}_${device}_all_backends.log" 2>&1 || {
            echo "  Error running $benchmark_name with all backends on $device"
        }
    else
        echo "  Benchmark directory not found: /app/benchmarks/$benchmark_name"
    fi
}

# List of available benchmarks
BENCHMARKS=(
    "equation_of_state"
    "isoneutral_mixing"
    "pytorch_benchmark"
    "turbulent_kinetic_energy"
)

# Check if GPU is available
if nvidia-smi &> /dev/null; then
    echo "GPU detected, running GPU benchmarks..."
    DEVICE="gpu"
else
    echo "No GPU detected, running CPU benchmarks..."
    DEVICE="cpu"
fi

# Run each benchmark
for benchmark in "${BENCHMARKS[@]}"; do
    echo ""
    echo "=== Running $benchmark ==="
    run_benchmark "$benchmark" "$DEVICE"
done

# Run PyTorch specific benchmark with different sizes
echo ""
echo "=== Running PyTorch benchmark with various sizes ==="
run_benchmark "pytorch_benchmark" "$DEVICE" "500 1000 5000 10000"

echo ""
echo "=== All benchmarks completed at $(date) ==="
echo "Results saved in /app/results/"

# Generate summary
echo ""
echo "=== Benchmark Summary ==="
echo "Results files created:"
ls -la /app/results/ || echo "No results directory found"

# Show disk usage
echo ""
echo "Disk usage:"
du -sh /app/results/ 2>/dev/null || echo "No results to measure"

echo ""
echo "=== Benchmarks finished successfully ===" 