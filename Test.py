import cProfile
import numpy as np
import scipy as sp
from scipy.stats import norm, variation
from scipy import optimize
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import pstats
import io

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


h=[9,17,17,26,23,20,19,12,6,10,1,1,1]
l=[3,14,50,74,56,40,126,40,37,36,48,76,91]
p=[1,1,1,1,1,1,3,4,2,1,5,2,9]
t=[7,10,14,17,21,25,29,32,36,40,45,52,57]

@lru_cache(maxsize=300)
def matrix_power(s0,s1,f2,t):
    a = np.array([[0, 0, f2],
                  [s0, 0, 0],
                  [0, s1, 0]])
    return np.linalg.matrix_power(a, t)

def p0(s0, s1, f2, h, l, p, t):
    a_n = matrix_power(s0,s1,f2,t)
    if a_n[0, 2] != 0:
        return a_n[0, 2] / h
    elif a_n[1, 2] != 0:
        return a_n[1, 2] / l
    elif a_n[2, 2] != 0:
        return a_n[2, 2] / p
    else:
        return np.nan

def det(s0, s1, f2, t):
    a_n = matrix_power(s0, s1, f2, t)
    if a_n[0, 2] != 0:
        return 1 / a_n[0, 2]
    elif a_n[1, 2] != 0:
        return 1 / a_n[1, 2]
    elif a_n[2, 2] != 0:
        return 1 / a_n[2, 2]
    else:
        return np.nan

def distribution(variable,mu,sigma):
    pdf_values = norm.pdf(variable,mu,sigma)
    return pdf_values

def fun1(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, h, l, p, t):
    def integrand(s0, s1, f2):

        p0_result = p0(s0,s1,f2,h,l,p,t)
        det_result = det(s0,s1,f2,t)

        return (distribution(s0, mu_0, sigma_0) *
                distribution(s1, mu_1, sigma_1) *
                distribution(f2, mu_2, sigma_2) *
                distribution(p0_result, mu_3, sigma_3) *
                det_result)

    for_range = [[-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf]]
    return sp.integrate.nquad(integrand, for_range)[0]

def n_hat_lp(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, h, t):
    def integrand(l,p):
        return fun1(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, h, l, p, t)

    for_range = [[-np.inf, np.inf], [-np.inf, np.inf]]
    return sp.integrate.nquad(integrand, for_range)[0]

def n_hat_hp(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, l, t):
    def integrand(h, p):
        return fun1(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, h, l, p, t)

    for_range = [[-np.inf, np.inf], [-np.inf, np.inf]]
    return sp.integrate.nquad(integrand, for_range)[0]

def n_hat_hl(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, p, t):
    def integrand(h,l):
        return fun1(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, h, l, p, t)

    for_range = [[-np.inf, np.inf], [-np.inf, np.inf]]
    return sp.integrate.nquad(integrand, for_range)[0]

def n_hat_p0(mu_3,sigma_3):
    def integrand(p0):
        return distribution(p0,mu_3,sigma_3)

    for_range = [[-np.inf, np.inf]]
    return sp.integrate.nquad(integrand, for_range)[0]

def exp_h(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, t):
    def integrand(h):
        return h*n_hat_lp(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, h, t)
    for_range = [[-np.inf, np.inf]]
    return sp.integrate.nquad(integrand, for_range)[0]

def exp_l(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, t):
    def integrand(l):
        return l*n_hat_hp(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, l, t)
    for_range = [[-np.inf, np.inf]]
    return sp.integrate.nquad(integrand, for_range)[0]

def exp_p(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, t):
    def integrand(p):
        return p*n_hat_hl(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, p, t)
    for_range = [[-np.inf, np.inf]]
    return sp.integrate.nquad(integrand, for_range)[0]

def exp_p0(mu_3,sigma_3):
    def integrand(p0):
        return p0*n_hat_p0(mu_3,sigma_3)
    for_range = [[-np.inf, np.inf]]
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

    return mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2

mu_sigma = starting_points(h,l,p,t)

x0 = np.concatenate((mu_sigma, [10, 1]))

def main():
    print(exp_h(x0, t))

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
