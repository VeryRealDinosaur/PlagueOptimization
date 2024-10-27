import cProfile
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import pstats
import io
import torch
from torchquad import MonteCarlo

#DEBUGING TOOLS
def create_callback(t, h, l, p, p0i):
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
    return callback

h=[9,17,18] #,26,23,20,19,12,6,10,1,1,1]
l=[3,14,50] #,74,56,40,126,40,37,36,48,76,91]
p=[1,1,2] #,1,1,1,3,4,2,1,5,2,9]
t=[7,10,14] #,17,21,25,29,32,36,40,45,52,57]

@lru_cache(maxsize=128)
def matrix_power(s0, s1, f2, t):
    a = torch.tensor([[0.0, 0.0, f2],
                      [s0, 0.0, 0.0],
                      [0.0, s1, 0.0]])

    result = a
    for _ in range(t - 1):
        result = torch.mm(result, a)

    return result

def p0(s0, s1, f2, h, l, p, t):
    a_n = matrix_power(s0, s1, f2, t)
    if a_n[0, 2] != 0:
        return a_n[0, 2] / h
    elif a_n[1, 2] != 0:
        return a_n[1, 2] / l
    elif a_n[2, 2] != 0:
        return a_n[2, 2] / p
    else:
        return torch.nan

def det(s0, s1, f2, t):
    a_n = matrix_power(s0, s1, f2, t)
    if a_n[0, 2] != 0:
        return 1 / a_n[0, 2]
    elif a_n[1, 2] != 0:
        return 1 / a_n[1, 2]
    elif a_n[2, 2] != 0:
        return 1 / a_n[2, 2]
    else:
        return torch.nan

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

def n_hat_lp(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, h, t):
    # Define integration limits (approximate for normal distribution)
    sigma_max = max(sigma_0, sigma_1, sigma_2, sigma_3)
    integration_limits = [[-4 * sigma_max, 4 * sigma_max], [-4 * sigma_max, 4 * sigma_max]]

    # Initialize Monte Carlo integrator
    mc_integrator = MonteCarlo()

    # Define the integrand function for `l` and `p`
    def integrand(tensor):
        l, p = tensor[:, 0], tensor[:, 1]
        return fun1(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, h, l, p, t)

    # Perform the integration
    result = mc_integrator.integrate(
        integrand=integrand,
        dim=2,
        N=10_000,  # Number of samples, adjust for accuracy as needed
        integration_domain=integration_limits
    )
    return result

def n_hat_hp(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, l, t):
    # Define integration limits (approximate for normal distribution)
    sigma_max = max(sigma_0, sigma_1, sigma_2, sigma_3)
    integration_limits = [[-4 * sigma_max, 4 * sigma_max], [-4 * sigma_max, 4 * sigma_max]]

    # Initialize Monte Carlo integrator
    mc_integrator = MonteCarlo()

    # Define the integrand function for `h` and `p`
    def integrand(tensor):
        h, p = tensor[:, 0], tensor[:, 1]
        return fun1(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, h, l, p, t)

    # Perform the integration
    result = mc_integrator.integrate(
        integrand=integrand,
        dim=2,
        N=10_000,  # Number of samples, adjust for accuracy as needed
        integration_domain=integration_limits
    )
    return result

def n_hat_hl(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, p, t):
    # Define integration limits (approximate for normal distribution)
    sigma_max = max(sigma_0, sigma_1, sigma_2, sigma_3)
    integration_limits = [[-4 * sigma_max, 4 * sigma_max], [-4 * sigma_max, 4 * sigma_max]]

    # Initialize Monte Carlo integrator
    mc_integrator = MonteCarlo()

    # Define the integrand function for `h` and `l`
    def integrand(tensor):
        h, l = tensor[:, 0], tensor[:, 1]
        return fun1(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, h, l, p, t)

    # Perform the integration
    result = mc_integrator.integrate(
        integrand=integrand,
        dim=2,
        N=10_000,  # Number of samples, adjust for accuracy as needed
        integration_domain=integration_limits
    )
    return result

def n_hat_p0(mu_3, sigma_3):
    # Define integration limits for `p0`
    integration_limits = [[-4 * sigma_3, 4 * sigma_3]]  # Approximate normal bounds

    # Initialize Monte Carlo integrator
    mc_integrator = MonteCarlo()

    # Define integrand for `p0`
    def integrand(tensor):
        p0 = tensor[:, 0]
        return distribution(p0, mu_3, sigma_3)  # Distribution function already returns PDF values

    # Perform the integration
    result = mc_integrator.integrate(
        integrand=integrand,
        dim=1,
        N=10_000,  # Number of samples
        integration_domain=integration_limits
    )
    return result

def exp_h(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, t):
    # Define integration limits for `h` (approximating infinite range)
    sigma_max = max(sigma_0, sigma_1, sigma_2, sigma_3)
    integration_limits = [[-4 * sigma_max, 4 * sigma_max]]

    # Initialize Monte Carlo integrator
    mc_integrator = MonteCarlo()

    # Define the integrand function for `h`
    def integrand(tensor):
        h = tensor[:, 0]
        # Call `n_hat_lp` with the provided `mu`, `sigma`, `h`, and `t` values
        return h * n_hat_lp(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, h, t)

    # Perform the integration
    result = mc_integrator.integrate(
        integrand=integrand,
        dim=1,
        N=10_000,  # Number of samples
        integration_domain=integration_limits
    )
    return result

def exp_l(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, t):
    # Define integration limits for `h` (approximating infinite range)
    sigma_max = max(sigma_0, sigma_1, sigma_2, sigma_3)
    integration_limits = [[-4 * sigma_max, 4 * sigma_max]]

    # Initialize Monte Carlo integrator
    mc_integrator = MonteCarlo()

    # Define the integrand function for `h`
    def integrand(tensor):
        l = tensor[:, 0]
        # Call `n_hat_lp` with the provided `mu`, `sigma`, `h`, and `t` values
        return l * n_hat_hp(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, l, t)

    # Perform the integration
    result = mc_integrator.integrate(
        integrand=integrand,
        dim=1,
        N=10_000,  # Number of samples
        integration_domain=integration_limits
    )
    return result

def exp_p(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, t):
    # Define integration limits for `h` (approximating infinite range)
    sigma_max = max(sigma_0, sigma_1, sigma_2, sigma_3)
    integration_limits = [[-4 * sigma_max, 4 * sigma_max]]

    # Initialize Monte Carlo integrator
    mc_integrator = MonteCarlo()

    # Define the integrand function for `h`
    def integrand(tensor):
        p = tensor[:, 0]
        # Call `n_hat_lp` with the provided `mu`, `sigma`, `h`, and `t` values
        return p * n_hat_hl(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, p, t)

    # Perform the integration
    result = mc_integrator.integrate(
        integrand=integrand,
        dim=1,
        N=10_000,  # Number of samples
        integration_domain=integration_limits
    )
    return result

def exp_p0(sigma_0, sigma_1, sigma_2, mu_3, sigma_3):
    # Define integration limits for `h` (approximating infinite range)
    sigma_max = max(sigma_0, sigma_1, sigma_2, sigma_3)
    integration_limits = [[-4 * sigma_max, 4 * sigma_max]]

    # Initialize Monte Carlo integrator
    mc_integrator = MonteCarlo()

    # Define the integrand function for `h`
    def integrand(tensor):
        p0 = tensor[:, 0]
        return p0 * n_hat_p0(mu_3,sigma_3)

    # Perform the integration
    result = mc_integrator.integrate(
        integrand=integrand,
        dim=1,
        N=10_000,  # Number of samples
        integration_domain=integration_limits
    )
    return result

def addition(params, t, h, l, p, p0i):
    mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3 = params

    with ProcessPoolExecutor() as executor:
        futures_info = {}

        for i in range(len(t)):
            future_h = executor.submit(exp_h, mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, t[i])
            future_l = executor.submit(exp_l, mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, t[i])
            future_p = executor.submit(exp_p, mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, t[i])

            futures_info[future_h] = ('h', i)
            futures_info[future_l] = ('l', i)
            futures_info[future_p] = ('p', i)

        suma = 0
        for future in as_completed(futures_info):
            func_type, i = futures_info[future]
            try:
                result = future.result()
                if func_type == 'h':
                    suma += (h[i] - result) ** 2
                elif func_type == 'l':
                    suma += (l[i] - result) ** 2
                elif func_type == 'p':
                    suma += (p[i] - result) ** 2
            except Exception as e:
                print(f"Error occurred for index {i}: {e}")

        suma += (p0i - exp_p0(mu_3, sigma_3)) ** 2

    return suma


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
    callback = create_callback(t, h, l, p, 10)
    x0=starting_points(h,l,p,t)
    result = addition(x0, t, h, l, p, 2)
    print("\nOptimización Completada:")
    print(result)

if __name__ == '__main__':
    pr = cProfile.Profile()
    pr.enable()

    main()

    pr.disable()

    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()

    print(s.getvalue())