from concurrent.futures import ProcessPoolExecutor

import numpy as np
import scipy as sp
from scipy.stats import norm
import time

def p0(s0,s1,f2,h,l,p,t):
    a = np.array([[0,0,f2],
                  [s0,0,0],
                  [0,s1,0]])
    a_n = np.linalg.matrix_power(a, t)

    if a_n[0,2]!=0:
        return a_n[0,2]/h
    elif a_n[1,2]!=0:
        return a_n[1,2]/l
    elif a_n[2,2]!=0:
        return a_n[2,2]/p
    else:
        print("error p0")

def det(s0,s1,f2,t):
    a = np.array([[0, 0, f2],
                  [s0, 0, 0],
                  [0, s1, 0]])
    a_n = np.linalg.matrix_power(a, t)

    if a_n[0,2]!=0:
        return 1/a_n[0,2]
    elif a_n[1,2]!=0:
        return 1/a_n[1,2]
    elif a_n[2,2]!=0:
        return 1/a_n[2,2]
    else:
        print("error det")

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
    return sp.integrate.nquad(integrand, for_range)

def n_hat_lp(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, h, t):
    def integrand(l,p):
        return fun1(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, h, l, p, t)

    for_range = [[-np.inf, np.inf], [-np.inf, np.inf]]
    print("lp_done")
    return sp.integrate.nquad(integrand, for_range)

def n_hat_hp(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, l, t):
    def integrand(h, p):
        return fun1(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, h, l, p, t)

    for_range = [[-np.inf, np.inf], [-np.inf, np.inf]]
    print("hp_done")
    return sp.integrate.nquad(integrand, for_range)

def n_hat_hl(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, p, t):
    def integrand(h,l):
        return fun1(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, h, l, p, t)

    for_range = [[-np.inf, np.inf], [-np.inf, np.inf]]
    print("hl_done")
    return sp.integrate.nquad(integrand, for_range)

def n_hat_p0(mu_3,sigma_3):
    def integrand(p0):
        return distribution(p0,mu_3,sigma_3)

    for_range = [[-np.inf, np.inf]]
    print("p0_done")
    return sp.integrate.nquad(integrand, for_range)
"""
def exp_h(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, t):
    def integrand(h):
        return h*n_hat_lp(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, h, t)
    for_range = [[-np.inf, np.inf]]
    return sp.integrate.nquad(integrand, for_range)

def exp_l(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, t):
    def integrand(l):
        return l*n_hat_hp(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, l, t)
    for_range = [[-np.inf, np.inf]]
    return sp.integrate.nquad(integrand, for_range)

def exp_p(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, t):
    def integrand(p):
        return p*n_hat_hl(mu_0, sigma_0, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, p, t)
    for_range = [[-np.inf, np.inf]]
    return sp.integrate.nquad(integrand, for_range)

def exp_p0(mu_3,sigma_3):
    def integrand(p0):
        return p0*n_hat_p0(mu_3,sigma_3)
    for_range = [[-np.inf, np.inf]]
    return sp.integrate.nquad(integrand, for_range)

def run_exp_h():
    return exp_h(1, 0.5, 1, 0.5, 1, 0.5, 1, 0.5, 2)

def run_exp_l():
    return exp_l(1, 0.5, 1, 0.5, 1, 0.5, 1, 0.5, 2)

def run_exp_p():
    return exp_p(1, 0.5, 1, 0.5, 1, 0.5, 1, 0.5, 2)

def run_exp_p0():
    return exp_p0(1, 0.5)"""

# Start multithreading
if __name__ == "__main__":
    start_time = time.time()

    """with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(run_exp_h),
            executor.submit(run_exp_l),
            executor.submit(run_exp_p),
            executor.submit(run_exp_p0)
        ]
        results = [future.result() for future in futures]"""

    n_hat_lp(1, 0.5, 1, 0.5, 1, 0.5, 1, 0.5, 2, 2)
    n_hat_hp(1, 0.5, 1, 0.5, 1, 0.5, 1, 0.5, 2, 2)
    n_hat_hl(1, 0.5, 1, 0.5, 1, 0.5, 1, 0.5, 2, 2)
    n_hat_p0(1,0.5)

    print("--- %s seconds ---" % (time.time() - start_time))