import torch
from functools import lru_cache
from torch.distributions import Normal

@lru_cache(maxsize=128)
def matrix_power(s0, s1, f2, t):
    if not isinstance(s0, torch.Tensor):
        s0 = torch.tensor([s0], dtype=torch.float64)
    if not isinstance(s1, torch.Tensor):
        s1 = torch.tensor([s1], dtype=torch.float64)
    if not isinstance(f2, torch.Tensor):
        f2 = torch.tensor([f2], dtype=torch.float64)

    batch_size = s0.shape[0]
    a = torch.zeros((batch_size, 3, 3), dtype=torch.float64)

    # Add small epsilon to diagonal for numerical stability
    epsilon = 1e-10
    a[:, 0, 2] = f2
    a[:, 1, 0] = s0
    a[:, 2, 1] = s1
    a += torch.eye(3, dtype=torch.float64) * epsilon

    result = torch.matrix_power(a, t)

    return result

def p0(s0, s1, f2, h, l, p, t):
    if not isinstance(h, torch.Tensor):
        h = torch.tensor([h], dtype=torch.float64)
    if not isinstance(l, torch.Tensor):
        l = torch.tensor([l], dtype=torch.float64)
    if not isinstance(p, torch.Tensor):
        p = torch.tensor([p], dtype=torch.float64)

    # Compute matrix power
    a_n = matrix_power(s0, s1, f2, t)
    last_column = a_n[:, :, 2]  # Select the last column

    # Get the largest value and the row index for each matrix in the batch
    max_values, row_indices = torch.max(last_column, dim=1)

    # Initialize the results tensor
    results = torch.zeros_like(max_values, dtype=torch.float64)

    # Divide max_values by the corresponding h, l, or p
    epsilon = torch.tensor(1e-10, dtype=torch.float64)
    results[row_indices == 0] = max_values[row_indices == 0] / torch.maximum(h[row_indices == 0], epsilon)
    results[row_indices == 1] = max_values[row_indices == 1] / torch.maximum(l[row_indices == 1], epsilon)
    results[row_indices == 2] = max_values[row_indices == 2] / torch.maximum(p[row_indices == 2], epsilon)

    return results

def det(s0, s1, f2, t):
    a_n = matrix_power(s0, s1, f2, t)
    last_column = a_n[:, :, 2]  # Select the last column

    max_values, row_indices = torch.max(last_column, dim=1)
    epsilon = torch.tensor(1e-10, dtype=torch.float64)
    log_max_values = torch.log(torch.maximum(max_values, epsilon))

    # Assign the negative log for inverse effect
    log_result = -log_max_values
    return torch.exp(torch.clip(log_result, -30, 30))

def distribution(variable, mu, sigma):

    if not isinstance(variable, torch.Tensor):
        variable = torch.tensor([variable], dtype=torch.float64)

    dist = torch.distributions.normal.Normal(mu, sigma)
    result = torch.exp(dist.log_prob(variable))
    return result