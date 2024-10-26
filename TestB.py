import cProfile
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import pstats
import io
import torch
from torchquad import MonteCarlo, set_up_backend

#DEBUGING TOOLS
"""def create_callback(t, h, l, p, p0i):
    def callback(xk):
        callback.count += 1

        fval = addition(xk, t, h, l, p, p0i)

        print(f"\nIteration {callback.count}:")
        print(f"  Parameters:")
        print(f"    μ₀: {xk[0]:.4f}, σ₀: {xk[1]:.4f}")
        print(f"    μ₁: {xk[2]:.4f}, σ₁: {xk[3]:.4f}")
        print(f"    μ₂: {xk[4]:.4f}, σ₂: {xk[5]:.4f}")
        print(f"    μ₃: {xk[6]:.4f}, σ₃: {xk[7]:.4f}")
        print(f"  Function value: {fval:.6f}")

    callback.count = 0
    return callback"""

set_up_backend("torch", data_type="float64")

h=[9,17,18] #,26,23,20,19,12,6,10,1,1,1]
l=[3,14,50] #,74,56,40,126,40,37,36,48,76,91]
p=[1,1,2] #,1,1,1,3,4,2,1,5,2,9]
t=[7,10,14] #,17,21,25,29,32,36,40,45,52,57]

def matrix_power(s0, s1, f2, t):
    # Handle f2 as a tensor
    a = torch.stack([torch.tensor([[0.0, 0.0, f],
                                   [s0, 0.0, 0.0],
                                   [0.0, s1, 0.0]]) for f in f2], dim=0)

    result = a
    for _ in range(t - 1):
        result = torch.bmm(result, a)

    return result

def p0(s0, s1, f2, h, l, p, t):
    a_n = matrix_power(s0, s1, f2, t)
    p0_result = torch.where(a_n[:, 0, 2] != 0, a_n[:, 0, 2] / h,
                            torch.where(a_n[:, 1, 2] != 0, a_n[:, 1, 2] / l,
                                        torch.where(a_n[:, 2, 2] != 0, a_n[:, 2, 2] / p, torch.nan)))
    return p0_result

def det(s0, s1, f2, t):
    a_n = matrix_power(s0, s1, f2, t)
    det_result = torch.where(a_n[:, 0, 2] != 0, 1 / a_n[:, 0, 2],
                             torch.where(a_n[:, 1, 2] != 0, 1 / a_n[:, 1, 2],
                                         torch.where(a_n[:, 2, 2] != 0, 1 / a_n[:, 2, 2], torch.nan)))
    return det_result

def distribution(variable, mu, sigma):
    coef = 1 / (sigma * torch.sqrt(torch.tensor(2 * torch.pi)))
    exponent = -0.5 * ((variable - mu) / sigma) ** 2
    return coef * torch.exp(exponent)

def fun1(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, h, l, p, t):
    def integrand(s):
        s0, s1, f2 = s[:, 0], s[:, 1], s[:, 2]

        p0_result = p0(s0, s1, f2, h, l, p, t)
        det_result = det(s0, s1, f2, t)

        return (distribution(s0, mu_0, sigma_0) *
                distribution(s1, mu_1, sigma_1) *
                distribution(f2, mu_2, sigma_2) *
                distribution(p0_result, mu_3, sigma_3) *
                det_result)

    integrator = MonteCarlo()
    for_range = torch.tensor([[-float('inf'), float('inf')],
                              [-float('inf'), float('inf')],
                              [-float('inf'), float('inf')]])

    result = integrator.integrate(integrand, dim=3, N=10000, integration_domain=for_range)
    return result

def main():
    fun1(1,0.5,1,0.5,1,0.5,1,0.5, 9, 3, 1, 7)

if __name__ == "__main__":
    main()