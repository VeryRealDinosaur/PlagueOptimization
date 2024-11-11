import torch
from functools import lru_cache
from torch.distributions import Normal
from torchquad import set_up_backend
from torchquad.integration.monte_carlo import MonteCarlo
import numpy as np
from scipy import optimize

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

h=[9,17,18,26,23,20,19,12,6,10,1,1,1]
l=[3,14,50,74,56,40,126,40,37,36,48,76,91]
p=[1,1,2,1,1,1,3,4,2,1,5,2,9]
t=[7,10,14,17,21,25,29,32,36,40,45,52,57]

def fun1_h(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, t):
    def integrand(x):
        s0, s1, f2, h, l, p = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5]

        # Use log-space calculations for better numerical stability
        log_result = torch.zeros_like(s0)

        # Calculate angular terms
        log_result += torch.log(torch.abs(torch.atan(h))) - torch.log1p(h ** 2)
        log_result -= torch.log1p(l ** 2)
        log_result -= torch.log1p(p ** 2)

        # Add distribution terms
        log_result += Normal(mu_0, sigma_0).log_prob(s0)
        log_result += Normal(mu_1, sigma_1).log_prob(s1)
        log_result += Normal(mu_2, sigma_2).log_prob(f2)

        # Calculate p0 and its distribution
        p0_val = p0(s0, s1, f2, torch.atan(h), torch.atan(l), torch.atan(p), t)
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
        [-np.pi / 2, np.pi / 2]
    ]

    # Use stratified sampling
    N = 300000  # Increased number of points
    result = vegas.integrate(integrand, dim=6, N=N, integration_domain=domain)
    return result.item()
def average_expectation_h(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, t, n_samples=100):
    samples = []

    for i in range(n_samples):
        result = fun1_h(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, t)
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



def fun1_l(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, t):
    def integrand(x):
        s0, s1, f2, h, l, p = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5]

        # Use log-space calculations for better numerical stability
        log_result = torch.zeros_like(s0)

        # Calculate angular terms
        log_result += torch.log(torch.abs(torch.atan(l))) - torch.log1p(l ** 2)
        log_result -= torch.log1p(h ** 2)
        log_result -= torch.log1p(p ** 2)

        # Add distribution terms
        log_result += Normal(mu_0, sigma_0).log_prob(s0)
        log_result += Normal(mu_1, sigma_1).log_prob(s1)
        log_result += Normal(mu_2, sigma_2).log_prob(f2)

        # Calculate p0 and its distribution
        p0_val = p0(s0, s1, f2, torch.atan(h), torch.atan(l), torch.atan(p), t)
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
    f2_range = [mu_2 - 3 * sigma_2, mu_2 + 6 * sigma_2]

    domain = [
        s0_range,
        s1_range,
        f2_range,
        [-np.pi / 2, np.pi / 2],
        [-np.pi / 2, np.pi / 2],
        [-np.pi / 2, np.pi / 2]
    ]

    # Use stratified sampling
    N = 300000  # Increased number of points
    result = vegas.integrate(integrand, dim=6, N=N, integration_domain=domain)
    return result.item()
def average_expectation_l(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, t, n_samples=100):
    samples = []

    for i in range(n_samples):
        result = fun1_l(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, t)
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



def fun1_p(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, t):
    def integrand(x):
        s0, s1, f2, h, l, p = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5]

        # Use log-space calculations for better numerical stability
        log_result = torch.zeros_like(s0)

        # Calculate angular terms
        log_result += torch.log(torch.abs(torch.atan(p))) - torch.log1p(p ** 2)
        log_result -= torch.log1p(h ** 2)
        log_result -= torch.log1p(l ** 2)

        # Add distribution terms
        log_result += Normal(mu_0, sigma_0).log_prob(s0)
        log_result += Normal(mu_1, sigma_1).log_prob(s1)
        log_result += Normal(mu_2, sigma_2).log_prob(f2)

        # Calculate p0 and its distribution
        p0_val = p0(s0, s1, f2, torch.atan(h), torch.atan(l), torch.atan(p), t)
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
        [-np.pi / 2, np.pi / 2],
        [-np.pi / 2, np.pi / 2],
        [-np.pi / 2, np.pi / 2]
    ]

    # Use stratified sampling
    N = 300000  # Increased number of points
    result = vegas.integrate(integrand, dim=6, N=N, integration_domain=domain)
    return result.item()
def average_expectation_p(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, t, n_samples=100):
    samples = []

    for i in range(n_samples):
        result = fun1_p(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, t)
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

def addition(params, h, l, p, t):
    mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3 = params
    suma = torch.tensor(0.0, requires_grad=True)

    file_path = "/Users/jovany/PycharmProjects/PlagueOptimization/OptimumValues.txt"
    with open(file_path, "a") as file:

        for i in range(len(h)):
            term_h = (h[i] - average_expectation_h(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, t[i])) ** 2
            term_l = (l[i] - average_expectation_l(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, t[i])) ** 2
            term_p = (p[i] - average_expectation_p(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, t[i])) ** 2
            suma = suma + term_h + term_l + term_p

        print(f"Function value: {suma}")
        print("Parameters:")
        print(f"  H (μ₀: {mu_0:8.4f}, σ₀: {sigma_0:8.4f})")
        print(f"  L (μ₁: {mu_1:8.4f}, σ₁: {sigma_1:8.4f})")
        print(f"  P (μ₂: {mu_2:8.4f}, σ₂: {sigma_2:8.4f})")
        print(f"  P0 (μ₃: {mu_3:8.4f}, σ₃: {sigma_3:8.4f})")
        print("-" * 60)

        file.write(f"Function value: {suma}\n")
        file.write("Parameters:\n")
        file.write(f"  H (μ₀: {mu_0:8.4f}, σ₀: {sigma_0:8.4f})\n")
        file.write(f"  L (μ₁: {mu_1:8.4f}, σ₁: {sigma_1:8.4f})\n")
        file.write(f"  P (μ₂: {mu_2:8.4f}, σ₂: {sigma_2:8.4f})\n")
        file.write(f"  P0 (μ₃: {mu_3:8.4f}, σ₃: {sigma_3:8.4f})\n")
        file.write("-" * 60 + "\n")

    return suma.detach().numpy()


def starting_points(h,l,p,t):
    h_np=np.array(h)
    l_np=np.array(l)
    p_np=np.array(p)
    t_np=np.array(t)

    delta_t_np=np.diff(t_np)

    mu_0 = np.average(h_np[1:], weights=delta_t_np)
    mu_1 = np.average(l_np[1:], weights=delta_t_np)
    mu_2 = np.average(p_np[1:], weights=delta_t_np)

    variance_0 = np.average((h_np[1:] - mu_0)**2, weights=delta_t_np)
    variance_1 = np.average((l_np[1:] - mu_1)**2, weights=delta_t_np)
    variance_2 = np.average((p_np[1:] - mu_2)**2, weights=delta_t_np)

    sigma_0 = np.sqrt(variance_0)
    sigma_1 = np.sqrt(variance_1)
    sigma_2 = np.sqrt(variance_2)

    print(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, 1, 1)

    return np.array([mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, 1, 1])

def main():
    initial = np.array([0.5743, 2.3477, 22.7972, 48.8559, 1.0132, 4.7313, 0.0413, 2.0098])

    result = optimize.minimize(
        fun=addition,
        x0=initial,
        args=(h, l, p, t),
        method='nelder-mead',
    )
    print("\nOptimización Completada:")
    print(result)


if __name__ == '__main__':
    main()