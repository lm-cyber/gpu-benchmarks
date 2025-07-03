#!/usr/bin/env python3
"""
Advanced PyTorch Benchmarking Example

This script demonstrates PyTorch's built-in benchmark utilities including:
- torch.utils.benchmark.Timer
- Blocked autorange for statistical analysis
- Comparison tables
- Fuzzing for automated input generation
- Callgrind instruction counting

Usage:
    python benchmarks/pytorch_benchmark/advanced_benchmark_example.py
"""

import torch
import torch.utils.benchmark as benchmark
from itertools import product


def batched_dot_mul_sum(a, b):
    """Computes batched dot by multiplying and summing"""
    return a.mul(b).sum(-1)


def batched_dot_bmm(a, b):
    """Computes batched dot by reducing to bmm"""
    a = a.reshape(-1, 1, a.shape[-1])
    b = b.reshape(-1, b.shape[-1], 1)
    return torch.bmm(a, b).flatten(-3)


def main():
    print("=== PyTorch Advanced Benchmarking Example ===\n")
    
    # Test data
    x = torch.randn(10000, 64)
    
    # Verify correctness
    assert batched_dot_mul_sum(x, x).allclose(batched_dot_bmm(x, x))
    print("✓ Both implementations produce identical results\n")
    
    # 1. Basic timing comparison
    print("1. Basic Timing Comparison")
    print("-" * 40)
    
    t0 = benchmark.Timer(
        stmt='batched_dot_mul_sum(x, x)',
        setup='from __main__ import batched_dot_mul_sum',
        globals={'x': x})
    
    t1 = benchmark.Timer(
        stmt='batched_dot_bmm(x, x)',
        setup='from __main__ import batched_dot_bmm',
        globals={'x': x})
    
    print(t0.timeit(100))
    print(t1.timeit(100))
    print()
    
    # 2. Blocked autorange for better statistics
    print("2. Blocked Autorange (Better Statistics)")
    print("-" * 45)
    
    m0 = t0.blocked_autorange(min_run_time=1)
    m1 = t1.blocked_autorange(min_run_time=1)
    
    print(m0)
    print(m1)
    print(f"mul/sum - Mean: {m0.mean * 1e6:.2f} us, Median: {m0.median * 1e6:.2f} us")
    print(f"bmm     - Mean: {m1.mean * 1e6:.2f} us, Median: {m1.median * 1e6:.2f} us")
    print()
    
    # 3. Multi-threaded comparison
    print("3. Multi-threaded Performance Comparison")
    print("-" * 45)
    
    results = []
    sizes = [64, 1024, 10000]
    threads = [1, 4, 8]
    
    for size, num_threads in product(sizes, threads):
        x_test = torch.randn(size, size)
        
        for func_name, stmt in [('mul/sum', 'batched_dot_mul_sum(x, x)'), 
                               ('bmm', 'batched_dot_bmm(x, x)')]:
            results.append(benchmark.Timer(
                stmt=stmt,
                setup=f'from __main__ import batched_dot_mul_sum, batched_dot_bmm',
                globals={'x': x_test},
                num_threads=num_threads,
                label='Batched dot',
                sub_label=f'[{size}x{size}]',
                description=func_name,
            ).blocked_autorange(min_run_time=0.5))
    
    compare = benchmark.Compare(results)
    compare.trim_significant_figures()
    compare.print()
    print()
    
    # 4. GPU benchmarking (if available)
    if torch.cuda.is_available():
        print("4. GPU Benchmarking")
        print("-" * 25)
        
        x_gpu = torch.randn(10000, 1024, device='cuda')
        
        t0_gpu = benchmark.Timer(
            stmt='batched_dot_mul_sum(x, x)',
            setup='from __main__ import batched_dot_mul_sum',
            globals={'x': x_gpu})
        
        t1_gpu = benchmark.Timer(
            stmt='batched_dot_bmm(x, x)',
            setup='from __main__ import batched_dot_bmm',
            globals={'x': x_gpu})
        
        print("GPU Results:")
        print(t0_gpu.blocked_autorange(min_run_time=1))
        print(t1_gpu.blocked_autorange(min_run_time=1))
        print()
    
    # 5. Fuzzed input generation
    print("5. Fuzzed Input Generation")
    print("-" * 30)
    
    from torch.utils.benchmark import Fuzzer, FuzzedParameter, FuzzedTensor
    
    fuzzer = Fuzzer(
        parameters=[
            FuzzedParameter('batch_size', minval=1, maxval=1000, distribution='loguniform'),
            FuzzedParameter('feature_size', minval=64, maxval=2048, distribution='loguniform'),
        ],
        tensors=[
            FuzzedTensor('x', size=('batch_size', 'feature_size'), 
                        min_elements=64, max_elements=100000, probability_contiguous=0.8)
        ],
        seed=42,
    )
    
    fuzz_results = []
    for tensors, tensor_params, params in fuzzer.take(5):
        sub_label = f"{params['batch_size']:<4} x {params['feature_size']:<4}"
        if not tensor_params['x']['is_contiguous']:
            sub_label += " (discontiguous)"
            
        for func_name, stmt in [('mul/sum', 'batched_dot_mul_sum(x, x)'), 
                               ('bmm', 'batched_dot_bmm(x, x)')]:
            fuzz_results.append(benchmark.Timer(
                stmt=stmt,
                setup='from __main__ import batched_dot_mul_sum, batched_dot_bmm',
                globals=tensors,
                label='Fuzzed batched dot',
                sub_label=sub_label,
                description=func_name,
            ).blocked_autorange(min_run_time=0.3))
    
    fuzz_compare = benchmark.Compare(fuzz_results)
    fuzz_compare.trim_significant_figures()
    fuzz_compare.print()
    print()
    
    # 6. Demonstrate measurement serialization
    print("6. Measurement Serialization")
    print("-" * 30)
    
    import pickle
    
    # Serialize a measurement
    measurement = t0.blocked_autorange(min_run_time=0.5)
    serialized = pickle.dumps(measurement)
    deserialized = pickle.loads(serialized)
    
    print(f"Original:     {measurement}")
    print(f"Deserialized: {deserialized}")
    print(f"Results match: {str(measurement) == str(deserialized)}")
    print()
    
    print("=== Benchmarking Complete ===")
    print("\nKey Takeaways:")
    print("- Use blocked_autorange() for more reliable measurements")
    print("- Thread count significantly affects performance")
    print("- GPU acceleration depends on problem size")
    print("- Fuzzed inputs help identify edge cases")
    print("- PyTorch benchmarks are serializable for A/B testing")


if __name__ == "__main__":
    main() 