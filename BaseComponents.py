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

    epsilon = 1e-100
    a[:, 0, 1] = 1/s0
    a[:, 1, 2] = 1/s1
    a[:, 2, 0] = 1/f2
    a += torch.eye(3, dtype=torch.float64) * epsilon

    result = torch.matrix_power(a, t)

    return result

def matrix_product(s0, s1, f2, h, l, p, t, case):
    def ensure_tensor(x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor([x], dtype=torch.float64)
        return x.squeeze()

    s0 = ensure_tensor(s0)
    s1 = ensure_tensor(s1)
    f2 = ensure_tensor(f2)
    h = ensure_tensor(h)
    l = ensure_tensor(l)
    p = ensure_tensor(p)

    # Create a batch tensor if inputs have different batch sizes
    batch_size = max(s0.numel(), s1.numel(), f2.numel(), h.numel(), l.numel(), p.numel())

    # Broadcast tensors to the same batch size
    s0 = s0.repeat(batch_size) if s0.numel() == 1 else s0
    s1 = s1.repeat(batch_size) if s1.numel() == 1 else s1
    f2 = f2.repeat(batch_size) if f2.numel() == 1 else f2
    h = h.repeat(batch_size) if h.numel() == 1 else h
    l = l.repeat(batch_size) if l.numel() == 1 else l
    p = p.repeat(batch_size) if p.numel() == 1 else p

    # Create vector b with broadcasting
    b = torch.zeros((batch_size, 3, 1), dtype=torch.float64)
    b[:, 0, 0] = h
    b[:, 1, 0] = l
    b[:, 2, 0] = p

    # Compute matrix power
    a = matrix_power(s0, s1, f2, t)

    # Perform matrix multiplication for each batch
    product = torch.bmm(a, b)

    return product[:, case, 0]

def det(s0, s1, f2, t):
    det = (1 / s0 * s1 * f2) ** -t
    return abs(det)

def distribution(variable, mu, sigma):

    if not isinstance(variable, torch.Tensor):
        variable = torch.tensor([variable], dtype=torch.float64)

    dist = torch.distributions.normal.Normal(mu, sigma)
    result = torch.exp(dist.log_prob(variable))
    return result

