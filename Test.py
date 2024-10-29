import torch
from functools import lru_cache
from torch.distributions import Normal
from torchquad import set_up_backend
from torchquad.integration.monte_carlo import MonteCarlo
import numpy as np
from scipy import optimize


h=[9,17,18] #,26,23,20,19,12,6,10,1,1,1]
l=[3,14,50] #,74,56,40,126,40,37,36,48,76,91]
p=[1,1,2] #,1,1,1,3,4,2,1,5,2,9]
t=[7,10,14] #,17,21,25,29,32,36,40,45,52,57]

set_up_backend("torch", data_type="float64")

vegas = MonteCarlo()

@lru_cache(maxsize=128)
def matrix_power(s0, s1, f2, t):

    batch_size = s0.shape[0]
    a = torch.zeros((batch_size, 3, 3), dtype=torch.float64)

    a[:, 0, 2] = f2
    a[:, 1, 0] = s0
    a[:, 2, 1] = s1

    result = torch.matrix_power(a, t)
    return result

def p0(s0, s1, f2, h, l, p, t):
    a_n = matrix_power(s0, s1, f2, t)

    h = h.squeeze()
    l = l.squeeze()
    p = p.squeeze()

    mask_0 = torch.abs(a_n[:, 0, 2]) > 1e-10
    mask_1 = torch.abs(a_n[:, 1, 2]) > 1e-10
    mask_2 = torch.abs(a_n[:, 2, 2]) > 1e-10

    result = torch.zeros_like(s0)
    result[mask_0] = a_n[mask_0, 0, 2] / h[mask_0]
    result[~mask_0 & mask_1] = a_n[~mask_0 & mask_1, 1, 2] / l[~mask_0 & mask_1]
    result[~mask_0 & ~mask_1 & mask_2] = a_n[~mask_0 & ~mask_1 & mask_2, 2, 2] / p[~mask_0 & ~mask_1 & mask_2]

    return result

def det(s0, s1, f2, t):
    a_n = matrix_power(s0, s1, f2, t)

    mask_0 = torch.abs(a_n[:, 0, 2]) > 1e-10
    mask_1 = torch.abs(a_n[:, 1, 2]) > 1e-10
    mask_2 = torch.abs(a_n[:, 2, 2]) > 1e-10

    result = torch.zeros_like(s0)
    result[mask_0] = 1 / a_n[mask_0, 0, 2]
    result[~mask_0 & mask_1] = 1 / a_n[~mask_0 & mask_1, 1, 2]
    result[~mask_0 & ~mask_1 & mask_2] = 1 / a_n[~mask_0 & ~mask_1 & mask_2, 2, 2]

    return result

def distribution(variable, mu, sigma):
    dist = Normal(mu, sigma)
    return torch.exp(dist.log_prob(variable))



def fun1_h(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, h, t):

    def integrand(x):

        s0, s1, f2, l, p = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4]

        return torch.atan((distribution(s0, mu_0, sigma_0) *
             distribution(s1, mu_1, sigma_1) *
             distribution(f2, mu_2, sigma_2) *
             distribution(p0(s0, s1, f2, h, l, p, t), mu_3, sigma_3) *
             det(s0, s1, f2, t))) / (1 + (distribution(s0, mu_0, sigma_0) *
                                         distribution(s1, mu_1, sigma_1) *
                                         distribution(f2, mu_2, sigma_2) *
                                         distribution(p0(s0, s1, f2, h, l, p, t), mu_3, sigma_3) *
                                         det(s0, s1, f2, t)) ** 2)


    domain = [
        [mu_0 - 6 * sigma_0, mu_0 + 6 * sigma_0],
        [mu_1 - 6 * sigma_1, mu_1 + 6 * sigma_1],
        [mu_2 - 6 * sigma_0, mu_2 + 6 * sigma_2],
        [-np.pi / 2, np.pi / 2],
        [-np.pi / 2, np.pi / 2],
    ]
    result = vegas.integrate(integrand, dim=5, N=100000, integration_domain=domain)
    return result.item()

def expectation_h(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, t):
    def integrand(h):

        return torch.atan(h*fun1_h(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, h, t))/(1+(h*fun1_h(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, h, t))**2)

    domain = [
        [-np.pi / 2, np.pi / 2]
    ]
    result = vegas.integrate(integrand, dim=1, N=100000, integration_domain=domain)
    return result.item()

def average_expectation_h(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, t):
    samples = []
    for i in range(100):
        result = expectation_h(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, t)
        samples.append(result)

    mean = sum(samples) / len(samples)
    variance = sum((x - mean) ** 2 for x in samples) / (len(samples) - 1)
    std_dev = variance ** 0.5

    print("Statistics:")
    print("Mean:       {:.6e}".format(mean))
    print("Variance:   {:.6e}".format(variance))
    print("Std Dev:    {:.6e}".format(std_dev))

    return mean



def fun1_l(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, l, t):

    def integrand(x):

        s0, s1, f2, h, p = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4]

        return torch.atan((distribution(s0, mu_0, sigma_0) *
             distribution(s1, mu_1, sigma_1) *
             distribution(f2, mu_2, sigma_2) *
             distribution(p0(s0, s1, f2, h, l, p, t), mu_3, sigma_3) *
             det(s0, s1, f2, t))) / (1 + (distribution(s0, mu_0, sigma_0) *
                                         distribution(s1, mu_1, sigma_1) *
                                         distribution(f2, mu_2, sigma_2) *
                                         distribution(p0(s0, s1, f2, h, l, p, t), mu_3, sigma_3) *
                                         det(s0, s1, f2, t)) ** 2)

    domain = [
        [mu_0 - 6 * sigma_0, mu_0 + 6 * sigma_0],
        [mu_1 - 6 * sigma_1, mu_1 + 6 * sigma_1],
        [mu_2 - 6 * sigma_0, mu_2 + 6 * sigma_2],
        [-np.pi / 2, np.pi / 2],
        [-np.pi / 2, np.pi / 2]
    ]
    result = vegas.integrate(integrand, dim=5, N=100000, integration_domain=domain)
    return result.item()

def expectation_l(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, t):
    def integrand(l):

        return torch.atan(l*fun1_l(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, l, t))/(1+(l*fun1_l(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, l, t))**2)

    domain = [
        [-np.pi / 2, np.pi / 2]
    ]
    result = vegas.integrate(integrand, dim=1, N=100000, integration_domain=domain)
    return result.item()

def average_expectation_l(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, t):
    samples = []
    for i in range(100):
        result = expectation_l(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, t)
        samples.append(result)

    mean = sum(samples) / len(samples)
    variance = sum((x - mean) ** 2 for x in samples) / (len(samples) - 1)
    std_dev = variance ** 0.5

    print("Statistics:")
    print("Mean:       {:.6e}".format(mean))
    print("Variance:   {:.6e}".format(variance))
    print("Std Dev:    {:.6e}".format(std_dev))

    return mean



def fun1_p(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, p, t):

    def integrand(x):

        s0, s1, f2, h, l = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4]

        return torch.atan((distribution(s0, mu_0, sigma_0) *
                           distribution(s1, mu_1, sigma_1) *
                           distribution(f2, mu_2, sigma_2) *
                           distribution(p0(s0, s1, f2, h, l, p, t), mu_3, sigma_3) *
                           det(s0, s1, f2, t))) / (1 + (distribution(s0, mu_0, sigma_0) *
                                                        distribution(s1, mu_1, sigma_1) *
                                                        distribution(f2, mu_2, sigma_2) *
                                                        distribution(p0(s0, s1, f2, h, l, p, t), mu_3, sigma_3) *
                                                        det(s0, s1, f2, t)) ** 2)

    domain = [
        [mu_0 - 6 * sigma_0, mu_0 + 6 * sigma_0],
        [mu_1 - 6 * sigma_1, mu_1 + 6 * sigma_1],
        [mu_2 - 6 * sigma_0, mu_2 + 6 * sigma_2],
        [-np.pi / 2, np.pi / 2],
        [-np.pi / 2, np.pi / 2]
    ]
    result = vegas.integrate(integrand, dim=5, N=100000, integration_domain=domain)
    return result.item()

def expectation_p(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, t):
    def integrand(p):

        return torch.atan(p*fun1_p(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, p, t))/(1+(p*fun1_p(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, p, t))**2)

    domain = [
        [-np.pi / 2, np.pi / 2]
    ]
    result = vegas.integrate(integrand, dim=1, N=100000, integration_domain=domain)
    return result.item()

def average_expectation_p(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, t):
    samples = []
    for i in range(100):
        result = expectation_p(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, t)
        samples.append(result)

    mean = sum(samples) / len(samples)
    variance = sum((x - mean) ** 2 for x in samples) / (len(samples) - 1)
    std_dev = variance ** 0.5

    print("Statistics:")
    print("Mean:       {:.6e}".format(mean))
    print("Variance:   {:.6e}".format(variance))
    print("Std Dev:    {:.6e}".format(std_dev))

    return mean



def expectation_p0(mu_3, sigma_3):
    def integrand(p_0):
        return p_0*distribution(p_0, mu_3, sigma_3)

    domain = [
        [mu_3 - 6 * sigma_3, mu_3 + 6 * sigma_3],
    ]
    result = vegas.integrate(integrand, dim=1, N=100000, integration_domain=domain)
    return result.item()

def average_expectation_p0(mu_3, sigma_3):
    samples = []
    for i in range(100):
        result = expectation_p0(mu_3, sigma_3)
        samples.append(result)

    mean = sum(samples) / len(samples)
    variance = sum((x - mean) ** 2 for x in samples) / (len(samples) - 1)
    std_dev = variance ** 0.5

    print("Statistics:")
    print("Mean:       {:.6e}".format(mean))
    print("Variance:   {:.6e}".format(variance))
    print("Std Dev:    {:.6e}".format(std_dev))

    return mean



def addition(params, h, l, p, p0i, t):
    mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3 = params
    suma = torch.tensor(0.0, requires_grad=True)
    for i in range(len(h)):
        term_h = (h[i] - average_expectation_h(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, t[i])) ** 2
        term_l = (l[i] - average_expectation_l(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, t[i])) ** 2
        term_p = (p[i] - average_expectation_p(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, t[i])) ** 2
        term_p0 = (p0i - average_expectation_p0(mu_3, sigma_3)) ** 2
        suma = suma + term_h + term_l + term_p + term_p0

    print(f"Function value: {suma}")
    print("Parameters:")
    print(f"  H (μ₀: {mu_0:8.4f}, σ₀: {sigma_0:8.4f})")
    print(f"  L (μ₁: {mu_1:8.4f}, σ₁: {sigma_1:8.4f})")
    print(f"  P (μ₂: {mu_2:8.4f}, σ₂: {sigma_2:8.4f})")
    print(f"  P0 (μ₃: {mu_3:8.4f}, σ₃: {sigma_3:8.4f})")
    print("-" * 60)

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

    return np.array([mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, 10, 1])

def main():
    initial = starting_points(h,l,p,t)

    result = optimize.minimize(
        fun=addition,
        x0=initial,
        args=(h, l, p, 10, t),
        method='nelder-mead',
    )
    print("\nOptimización Completada:")
    print(result)


if __name__ == '__main__':
    main()
