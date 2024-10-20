import cProfile
import cupy as cp
import cupyx.scipy as sp  # Use cupyx for CuPy-compatible SciPy functions
from cupyx.scipy.stats import norm
from cupyx import optimize
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import pstats
import io

# Data
h = cp.array([0, 9, 17, 17, 26, 23, 20, 19, 12, 6, 10, 0, 0, 0])
l = cp.array([0, 3, 14, 50, 74, 56, 40, 126, 40, 37, 36, 48, 76, 91])
p = cp.array([0, 0, 0, 0, 0, 0, 0, 3, 4, 2, 0, 5, 2, 9])
t = cp.array([0, 7, 10, 14, 17, 21, 25, 29, 32, 36, 40, 45, 52, 57])

@lru_cache(maxsize=None)
def matrix_power(s0, s1, f2, t):
    a = cp.array([[0, 0, f2],
                   [s0, 0, 0],
                   [0, s1, 0]])
    return cp.linalg.matrix_power(a, t)

def p0(s0, s1, f2, h, l, p, t):
    a_n = matrix_power(s0, s1, f2, t)
    if a_n[0, 2] != 0:
        return a_n[0, 2] / h
    elif a_n[1, 2] != 0:
        return a_n[1, 2] / l
    elif a_n[2, 2] != 0:
        return a_n[2, 2] / p
    else:
        return cp.nan

def det(s0, s1, f2, t):
    a_n = matrix_power(s0, s1, f2, t)
    if a_n[0, 2] != 0:
        return 1 / a_n[0, 2]
    elif a_n[1, 2] != 0:
        return 1 / a_n[1, 2]
    elif a_n[2, 2] != 0:
        return 1 / a_n[2, 2]
    else:
        return cp.nan

def distribution(variable, mu, sigma):
    pdf_values = norm.pdf(variable, mu, sigma)
    return pdf_values

def fun1(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, h, l, p, t):
    def integrand(s0, s1, f2):
        p0_result = p0(s0, s1, f2, h, l, p, t)
        det_result = det(s0, s1, f2, t)
        return (distribution(s0, mu_0, sigma_0) *
                distribution(s1, mu_1, sigma_1) *
                distribution(f2, mu_2, sigma_2) *
                distribution(p0_result, mu_3, sigma_3) *
                det_result)

    for_range = [[-cp.inf, cp.inf], [-cp.inf, cp.inf], [-cp.inf, cp.inf]]
    return sp.integrate.nquad(integrand, for_range)[0]

def n_hat_lp(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, h, t):
    def integrand(l, p):
        return fun1(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, h, l, p, t)

    for_range = [[-cp.inf, cp.inf], [-cp.inf, cp.inf]]
    print("lp_done")
    return sp.integrate.nquad(integrand, for_range)[0]

def n_hat_hp(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, l, t):
    def integrand(h, p):
        return fun1(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, h, l, p, t)

    for_range = [[-cp.inf, cp.inf], [-cp.inf, cp.inf]]
    print("hp_done")
    return sp.integrate.nquad(integrand, for_range)[0]

def n_hat_hl(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, p, t):
    def integrand(h, l):
        return fun1(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, h, l, p, t)

    for_range = [[-cp.inf, cp.inf], [-cp.inf, cp.inf]]
    print("hl_done")
    return sp.integrate.nquad(integrand, for_range)[0]

def n_hat_p0(mu_3, sigma_3):
    def integrand(p0):
        return distribution(p0, mu_3, sigma_3)

    for_range = [[-cp.inf, cp.inf]]
    print("p0_done")
    return sp.integrate.nquad(integrand, for_range)[0]

def exp_h(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, t):
    def integrand(h):
        return h * n_hat_lp(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, h, t)

    for_range = [[-cp.inf, cp.inf]]
    return sp.integrate.nquad(integrand, for_range)[0]

def exp_l(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, t):
    def integrand(l):
        return l * n_hat_hp(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, l, t)

    for_range = [[-cp.inf, cp.inf]]
    return sp.integrate.nquad(integrand, for_range)[0]

def exp_p(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, t):
    def integrand(p):
        return p * n_hat_hl(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, p, t)

    for_range = [[-cp.inf, cp.inf]]
    return sp.integrate.nquad(integrand, for_range)[0]

def exp_p0(mu_3, sigma_3):
    def integrand(p0):
        return p0 * n_hat_p0(mu_3, sigma_3)

    for_range = [[-cp.inf, cp.inf]]
    return sp.integrate.nquad(integrand, for_range)[0]

def addition(params, t, h, l, p, p0i):
    mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3 = params

    with ProcessPoolExecutor() as executor:
        futures = {}

        for i in range(len(t)):
            futures[executor.submit(exp_h, mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, t[i])] = i
            futures[executor.submit(exp_l, mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, t[i])] = i
            futures[executor.submit(exp_p, mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, t[i])] = i

        suma = 0
        for future in as_completed(futures):
            i = futures[future]
            try:
                result = future.result()
                if future.fn == exp_h:
                    suma += (h[i] - result) ** 2
                elif future.fn == exp_l:
                    suma += (l[i] - result) ** 2
                elif future.fn == exp_p:
                    suma += (p[i] - result) ** 2
            except Exception as e:
                print(f"Error occurred for index {i}: {e}")

        suma += (p0i - exp_p0(mu_3, sigma_3)) ** 2
        return suma

def main():
    mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3 = (0, 1, 0, 1, 0, 1, 0, 1)  # Initial parameters
    p0i = 1.0  # Initial guess for p0

    # Profiling
    pr = cProfile.Profile()
    pr.enable()

    # Optimization
    result = optimize.minimize(addition, [mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3],
                                args=(t, h, l, p, p0i), method='Nelder-Mead')

    pr.disable()
    s = io.StringIO()
    sortby = pstats.SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()

    print(f"Result: {result.x}")
    print(f"Optimization success: {result.success}")
    print(f"Cumulative profiling results:\n{s.getvalue()}")

if __name__ == "__main__":
    main()
