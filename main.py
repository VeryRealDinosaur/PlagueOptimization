import cProfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import scipy as sp
from scipy.stats import norm
import time
from scipy import optimize
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import numba
import pstats
import io

h=[0,9,17,17,26,23,20,19,12,6,10,0,0,0]
l=[0,3,14,50,74,56,40,126,40,37,36,48,76,91]
p=[0,0,0,0,0,0,0,3,4,2,0,5,2,9]
t=[0,7,10,14,17,21,25,29,32,36,40,45,52,57]

@lru_cache(maxsize=None)
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
    print("lp_done")
    return sp.integrate.nquad(integrand, for_range)[0]

def n_hat_hp(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, l, t):
    def integrand(h, p):
        return fun1(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, h, l, p, t)

    for_range = [[-np.inf, np.inf], [-np.inf, np.inf]]
    print("hp_done")
    return sp.integrate.nquad(integrand, for_range)[0]

def n_hat_hl(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, p, t):
    def integrand(h,l):
        return fun1(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, h, l, p, t)

    for_range = [[-np.inf, np.inf], [-np.inf, np.inf]]
    print("hl_done")
    return sp.integrate.nquad(integrand, for_range)[0]

def n_hat_p0(mu_3,sigma_3):
    def integrand(p0):
        return distribution(p0,mu_3,sigma_3)

    for_range = [[-np.inf, np.inf]]
    print("p0_done")
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

x0 = [np.mean(h), np.std(h), np.mean(l), np.std(l), np.mean(l), np.std(l), 10, 1]

def main():
    result_bfgs = optimize.minimize(fun=addition, x0=x0,
                                    args=(t, h, l, p, 10),
                                    method='BFGS')

    print(result_bfgs)

if __name__ == '__main__':
    pr = cProfile.Profile()
    pr.enable()

    main()

    pr.disable()

    # Create a stats object to analyze the profile
    s = io.StringIO()
    sortby = 'cumulative'  # You can change this to 'time', 'calls', etc.
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()

    print(s.getvalue())
