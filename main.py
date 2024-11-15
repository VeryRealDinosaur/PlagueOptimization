import torch
from torch.distributions import Normal
from torchquad import set_up_backend
import numpy as np
from scipy import optimize
from torchquad.integration.monte_carlo import MonteCarlo
from torchquad.integration.vegas import VEGAS
from scipy.stats import norm
from torchquad.integration.gaussian import GaussLegendre

from BaseComponents import *

set_up_backend("torch", data_type="float64")
Gauss = GaussLegendre()

h=[9,17,18,26,23,20,19,12,6,10,1,1,1]
l=[3,14,50,74,56,40,126,40,37,36,48,76,91]
p=[1,1,2,1,1,1,3,4,2,1,5,2,9]
t=[7,10,14,17,21,25,29,32,36,40,45,52,57]

def fun1_h(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, t):
    def integrand(x):
        s0, s1, f2, h, l, p = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5]

        result = ((torch.atan(h)*distribution(s0,mu_0,sigma_0)
                *distribution(s1,mu_1,sigma_1)*distribution(f2,mu_2,sigma_2)
                *distribution(p0(s0, s1, f2, torch.atan(h), torch.atan(l), torch.atan(p), t),mu_3,sigma_3)
                *det(s0,s1,f2,t)))/((1 + h ** 2) * (1 + l ** 2) * (1 + p ** 2))

        return result

    # Use importance sampling by focusing on regions where the integrand is likely to be large

    domain = [
        [mu_0 - 6 * sigma_0, mu_0 + 6 * sigma_0],
        [mu_1 - 6 * sigma_1, mu_1 + 6 * sigma_1],
        [mu_2 - 6 * sigma_2, mu_2 + 6 * sigma_2],
        [-np.pi / 2, np.pi / 2],
        [-np.pi / 2, np.pi / 2],
        [-np.pi / 2, np.pi / 2]
    ]
    # Use stratified sampling
    N = 300000  # Increased number of points
    result = Gauss.integrate(integrand, dim=6, N=N, integration_domain=domain)
    return result.item()

def fun1_l(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, t):
    def integrand(x):
        s0, s1, f2, h, l, p = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5]

        result = ((torch.atan(l)*distribution(s0,mu_0,sigma_0)
                *distribution(s1,mu_1,sigma_1)*distribution(f2,mu_2,sigma_2)
                *distribution(p0(s0, s1, f2, torch.atan(h), torch.atan(l), torch.atan(p), t),mu_3,sigma_3)
                *det(s0,s1,f2,t)))/((1 + h ** 2) * (1 + l ** 2) * (1 + p ** 2))

        return result

    # Use importance sampling by focusing on regions where the integrand is likely to be large

    domain = [
        [mu_0 - 6 * sigma_0, mu_0 + 6 * sigma_0],
        [mu_1 - 6 * sigma_1, mu_1 + 6 * sigma_1],
        [mu_2 - 6 * sigma_2, mu_2 + 6 * sigma_2],
        [-np.pi / 2, np.pi / 2],
        [-np.pi / 2, np.pi / 2],
        [-np.pi / 2, np.pi / 2]
    ]
    # Use stratified sampling
    N = 300000  # Increased number of points
    result = Gauss.integrate(integrand, dim=6, N=N, integration_domain=domain)
    return result.item()

def fun1_p(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, t):
    def integrand(x):
        s0, s1, f2, h, l, p = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5]

        result = ((torch.atan(p)*distribution(s0,mu_0,sigma_0)
                *distribution(s1,mu_1,sigma_1)*distribution(f2,mu_2,sigma_2)
                *distribution(p0(s0, s1, f2, torch.atan(h), torch.atan(l), torch.atan(p), t),mu_3,sigma_3)
                *det(s0,s1,f2,t)))/((1 + h ** 2) * (1 + l ** 2) * (1 + p ** 2))

        return result

    # Use importance sampling by focusing on regions where the integrand is likely to be large

    domain = [
        [mu_0 - 6 * sigma_0, mu_0 + 6 * sigma_0],
        [mu_1 - 6 * sigma_1, mu_1 + 6 * sigma_1],
        [mu_2 - 6 * sigma_2, mu_2 + 6 * sigma_2],
        [-np.pi / 2, np.pi / 2],
        [-np.pi / 2, np.pi / 2],
        [-np.pi / 2, np.pi / 2]
    ]
    # Use stratified sampling
    N = 300000  # Increased number of points
    result = Gauss.integrate(integrand, dim=6, N=N, integration_domain=domain)
    return result.item()


def addition(params, h, l, p, t):
    mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3 = params
    suma = torch.tensor(0.0, requires_grad=True)

    file_path = "/Users/jovany/PycharmProjects/PlagueOptimization/OptimumValues.txt"
    with open(file_path, "a") as file:

        for i in range(len(h)):
            term_h = (h[i] - fun1_h(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, t[i])) ** 2
            term_l = (l[i] - fun1_l(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, t[i])) ** 2
            term_p = (p[i] - fun1_p(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, t[i])) ** 2
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



def main():
    initial = np.array([0.5743, 2.3477, 22.7972, 48.8559, 1.0132, 4.7313, 0.0413, 2.0098])

    print(matrix_power(5, 4, 7, 7))

    print(det(-5, 4, 7, 7,))

    val = p0(-20,10,12,-1,0.3244,0.23,7)
    print(val)

    print("dists")

    d=norm.pdf(-1, loc=-1, scale=1)
    print(d)

    print(distribution(-1,-1,1))


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