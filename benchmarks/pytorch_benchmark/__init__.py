import math
import importlib
import functools


def generate_inputs(size):
    import numpy as np

    np.random.seed(17)

    # Generate different tensor sizes for benchmarking
    batch_size = max(1, int(size ** 0.3))
    feature_size = max(64, int(size ** 0.35))
    
    # Create test tensors for various operations
    inputs = {
        'batched_dot_a': np.random.randn(batch_size, feature_size).astype(np.float32),
        'batched_dot_b': np.random.randn(batch_size, feature_size).astype(np.float32),
        'matrix_a': np.random.randn(batch_size, feature_size, feature_size).astype(np.float32),
        'matrix_b': np.random.randn(batch_size, feature_size, feature_size).astype(np.float32),
        'conv_input': np.random.randn(batch_size, 64, 32, 32).astype(np.float32),
        'linear_input': np.random.randn(batch_size, feature_size).astype(np.float32),
    }
    
    return inputs


def try_import(backend):
    try:
        return importlib.import_module(f".pytorch_benchmark_{backend}", __name__)
    except ImportError:
        return None


def get_callable(backend, size, device="cpu"):
    backend_module = try_import(backend)
    inputs = generate_inputs(size)
    if hasattr(backend_module, "prepare_inputs"):
        inputs = backend_module.prepare_inputs(inputs, device=device)
    return functools.partial(backend_module.run, inputs, device=device)


__implementations__ = (
    "pytorch",
) 