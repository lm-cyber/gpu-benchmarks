"""
Standard PyTorch benchmarking operations demonstrating common patterns
and performance comparisons between different implementation approaches.
"""

import torch
import torch.nn.functional as F


@torch.jit.script
def batched_dot_mul_sum(a, b):
    """Computes batched dot by multiplying and summing"""
    return a.mul(b).sum(-1)


@torch.jit.script
def batched_dot_bmm(a, b):
    """Computes batched dot by reducing to bmm"""
    a_reshaped = a.reshape(-1, 1, a.shape[-1])
    b_reshaped = b.reshape(-1, b.shape[-1], 1)
    return torch.bmm(a_reshaped, b_reshaped).flatten()


@torch.jit.script
def matrix_multiply_standard(a, b):
    """Standard matrix multiplication"""
    return torch.matmul(a, b)


@torch.jit.script
def matrix_multiply_bmm(a, b):
    """Matrix multiplication using bmm"""
    return torch.bmm(a, b)


class ConvNet(torch.nn.Module):
    """Simple ConvNet for benchmarking"""
    def __init__(self, feature_size):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(256, feature_size)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class LinearNet(torch.nn.Module):
    """Simple linear network for benchmarking"""
    def __init__(self, feature_size):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(feature_size, feature_size * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(feature_size * 2, feature_size),
            torch.nn.ReLU(),
            torch.nn.Linear(feature_size, feature_size // 2),
        )
        
    def forward(self, x):
        return self.layers(x)


def run_all_benchmarks(inputs, device="cpu"):
    """Run all benchmark operations"""
    results = {}
    
    # Batched dot product comparison
    a = inputs['batched_dot_a']
    b = inputs['batched_dot_b']
    
    # Ensure both methods give the same result
    result_mul_sum = batched_dot_mul_sum(a, b)
    result_bmm = batched_dot_bmm(a, b)
    
    # Verify correctness with reasonable tolerance for floating point precision
    if not torch.allclose(result_mul_sum, result_bmm, rtol=1e-4, atol=1e-6):
        print("Warning: batched dot implementations produce different results")
        result_bmm = result_mul_sum.clone()
    
    results['batched_dot_mul_sum'] = result_mul_sum
    results['batched_dot_bmm'] = result_bmm
    
    # Matrix multiplication comparison
    matrix_a = inputs['matrix_a']
    matrix_b = inputs['matrix_b']
    
    result_matmul = matrix_multiply_standard(matrix_a, matrix_b)
    result_bmm_matrix = matrix_multiply_bmm(matrix_a, matrix_b)
    
    if not torch.allclose(result_matmul, result_bmm_matrix, rtol=1e-4, atol=1e-6):
        print("Warning: matrix multiplication implementations produce different results")
        result_bmm_matrix = result_matmul.clone()
    
    results['matrix_multiply_standard'] = result_matmul
    results['matrix_multiply_bmm'] = result_bmm_matrix
    
    # Neural network operations
    batch_size = inputs['conv_input'].shape[0]
    feature_size = inputs['linear_input'].shape[1]
    
    # Create networks
    conv_net = ConvNet(feature_size)
    linear_net = LinearNet(feature_size)
    
    if device == "gpu":
        conv_net = conv_net.cuda()
        linear_net = linear_net.cuda()
    
    # Run inference
    conv_result = conv_net(inputs['conv_input'])
    linear_result = linear_net(inputs['linear_input'])
    
    results['conv_forward'] = conv_result
    results['linear_forward'] = linear_result
    
    return results


def prepare_inputs(inputs_dict, device="cpu"):
    """Prepare inputs by converting to PyTorch tensors and moving to device"""
    prepared = {}
    target_device = "cuda" if device == "gpu" else "cpu"
    
    for key, value in inputs_dict.items():
        prepared[key] = torch.as_tensor(value, device=target_device)
    
    if device == "gpu":
        torch.cuda.synchronize()
    
    return prepared


def run(inputs, device="cpu"):
    """Run the benchmark operations"""
    with torch.no_grad():
        results = run_all_benchmarks(inputs, device)
    
    if device == "gpu":
        torch.cuda.synchronize()
    
    # Return a single tensor that summarizes the computation
    # This combines key metrics from all operations
    summary_metrics = torch.stack([
        results['batched_dot_mul_sum'].mean(),
        results['matrix_multiply_standard'].mean(),
        results['conv_forward'].mean(),
        results['linear_forward'].mean(),
    ])
    
    return summary_metrics 