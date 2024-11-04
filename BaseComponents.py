import torch
from functools import lru_cache

@lru_cache(maxsize=128)
def matrix_power(s0, s1, f2, t):
    batch_size = s0.shape[0]
    a = torch.zeros((batch_size, 3, 3), dtype=torch.float64)

    # Add small epsilon to diagonal for numerical stability
    epsilon = 1e-10
    a[:, 0, 2] = f2
    a[:, 1, 0] = s0
    a[:, 2, 1] = s1
    a += torch.eye(3, dtype=torch.float64) * epsilon

    # Use logarithm of matrix for better numerical stability
    try:
        result = torch.matrix_exp(t * torch.matrix_log(a))
    except:
        result = torch.matrix_power(a, t)

    return result

def p0(s0, s1, f2, h, l, p, t):
    a_n = matrix_power(s0, s1, f2, t)

    h = h.squeeze()
    l = l.squeeze()
    p = p.squeeze()

    # Use softmax for smoother transitions between masks
    epsilon = 1e-10
    weights = torch.softmax(torch.stack([
        torch.abs(a_n[:, 0, 2]),
        torch.abs(a_n[:, 1, 2]),
        torch.abs(a_n[:, 2, 2])
    ]), dim=0)

    result = (weights[0] * a_n[:, 0, 2] / (h + epsilon) +
              weights[1] * a_n[:, 1, 2] / (l + epsilon) +
              weights[2] * a_n[:, 2, 2] / (p + epsilon))

    return torch.clip(result, -1e3, 1e3)

def det(s0, s1, f2, t):
    a_n = matrix_power(s0, s1, f2, t)

    # Use softmax for smoother transitions
    epsilon = 1e-10
    weights = torch.softmax(torch.stack([
        torch.abs(a_n[:, 0, 2]),
        torch.abs(a_n[:, 1, 2]),
        torch.abs(a_n[:, 2, 2])
    ]), dim=0)

    result = (weights[0] / (a_n[:, 0, 2] + epsilon) +
              weights[1] / (a_n[:, 1, 2] + epsilon) +
              weights[2] / (a_n[:, 2, 2] + epsilon))

    return torch.clip(result, -1e3, 1e3)
