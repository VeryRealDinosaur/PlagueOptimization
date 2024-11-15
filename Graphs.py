from torch.distributions import Normal
from torchquad import set_up_backend
from torchquad.integration.monte_carlo import MonteCarlo
import numpy as np
import matplotlib.pyplot as plt

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


set_up_backend("torch", data_type="float64")
vegas = MonteCarlo()

def fun1_h(h, t):

    mu_0 = 0.5743
    sigma_0 = 2.3477
    mu_1 = 22.7972
    sigma_1 = 48.8559
    mu_2 = 1.0132
    sigma_2 = 4.7313
    mu_3 = 0.0413
    sigma_3 = 2.0098

    def integrand(x):
        s0, s1, f2, l, p = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4]

        # Use log-space calculations for better numerical stability
        log_result = torch.zeros_like(s0)

        # Calculate angular terms
        log_result -= torch.log1p(l ** 2)
        log_result -= torch.log1p(p ** 2)

        # Add distribution terms
        log_result += Normal(mu_0, sigma_0).log_prob(s0)
        log_result += Normal(mu_1, sigma_1).log_prob(s1)
        log_result += Normal(mu_2, sigma_2).log_prob(f2)

        # Calculate p0 and its distribution
        p0_val = p0(s0, s1, f2, h, torch.atan(l), torch.atan(p), t)
        log_result += Normal(mu_3, sigma_3).log_prob(p0_val)

        # Add determinant
        det_val = det(s0, s1, f2, t)
        log_result += torch.log(torch.abs(det_val) + 1e-10)

        # Convert back from log space with careful handling
        result = torch.exp(torch.clip(log_result, -30, 30))
        return result

    # Use importance sampling by focusing on regions where the integrand is likely to be large
    s0_range = [mu_0 - 6 * sigma_0, mu_0 + 6 * sigma_0]
    s1_range = [mu_1 - 6 * sigma_1, mu_1 + 6 * sigma_1]
    f2_range = [mu_2 - 6 * sigma_2, mu_2 + 6 * sigma_2]

    domain = [
        s0_range,
        s1_range,
        f2_range,
        [-np.pi / 2, np.pi / 2],  # Reduced range for angular variables
        [-np.pi / 2, np.pi / 2],
    ]

    # Use stratified sampling
    N = 300000  # Increased number of points
    result = vegas.integrate(integrand, dim=5, N=N, integration_domain=domain)
    return result.item()

def average_fun1_h(h, t, n_samples=100):
    samples = []

    for i in range(n_samples):
        result = fun1_h(h, t)
        samples.append(result)

        # Print progress
        if (i + 1) % 10 == 0:
            print(f"Completed {i + 1}/{n_samples} samples")

    # Use median and MAD for more robust statistics
    median = np.median(samples)
    mad = np.median(np.abs(samples - median))

    # Calculate trimmed mean (excluding top and bottom 10%)
    trimmed_samples = np.sort(samples)[n_samples // 10:-n_samples // 10]
    trimmed_mean = np.mean(trimmed_samples)
    trimmed_std = np.std(trimmed_samples)

    print("H expectation")
    print("\nRobust Statistics:")
    print(f"Median:     {median:.6e}")
    print(f"MAD:        {mad:.6e}")
    print("\nTrimmed Statistics (middle 80%):")
    print(f"Mean:       {trimmed_mean:.6e}")
    print(f"Std Dev:    {trimmed_std:.6e}")

    return trimmed_mean

def main():
    x = np.linspace(-1, 5, 10)
    y = np.linspace(-1, 5, 10)

    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            temph = X[i, j]
            h = temph.astype(int)
            print(h)
            tempt = Y[i, j]
            t = tempt.astype(int)
            print(t)
            Z[i, j] = average_fun1_h(h, t)

    # Plotting the result
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_wireframe(X, Y, Z, color='green')
    ax.set_title('plot')
    plt.show()

if __name__ == '__main__':
    main()