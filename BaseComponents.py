import torch
from functools import lru_cache

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
    a_n = matrix_power(s0, s1, f2, t)

    h = h.squeeze()
    l = l.squeeze()
    p = p.squeeze()

    last_column = a_n[:, :, 2]  # Select the last column

    # Get the largest value and the row index for each matrix in the batch
    max_values, row_indices = torch.max(last_column, dim=1)

    for i in range(len(max_values)):
        if row_indices[i] == 0:
            result = max_values[i] / h
        elif row_indices[i] == 1:
            result = max_values[i] / l
        elif row_indices[i] == 2:
            result = max_values[i] / p
        # Use modified_value as needed
    return result


def det(s0, s1, f2, t):
    a_n = matrix_power(s0, s1, f2, t)

    last_column = a_n[:, :, 2]  # Select the last column

    # Get the largest value and the row index for each matrix in the batch
    max_values, row_indices = torch.max(last_column, dim=1)

    for i in range(len(max_values)):
        if row_indices[i] == 0:
            result = 1/ max_values[i]
        elif row_indices[i] == 1:
            result = 1/ max_values[i]
        elif row_indices[i] == 2:
            result = 1/ max_values[i]
        # Use modified_value as needed
    return result

