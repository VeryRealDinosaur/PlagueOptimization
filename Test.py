from scipy.cluster.hierarchy import average
from torchquad import set_up_backend
import numpy as np
from scipy import optimize
from torchquad.integration.monte_carlo import MonteCarlo


from BaseComponents import *

set_up_backend("torch", data_type="float64")
MonteCarlo = MonteCarlo()

H=[9, 17, 18, 26, 23, 20, 19, 12, 6, 10, 1, 1, 1]
L=[3, 14, 50, 74, 56, 40, 126, 40, 37, 36, 48, 76, 91]
P=[1, 1, 2, 1, 1, 1, 3, 4, 2, 1, 5, 2, 9]
T=[7, 10, 14, 17, 21, 25, 29, 32, 36, 40, 45, 52, 57]


def fun1(params, t, case):

    mu_s0, sigma_s0, mu_s1, sigma_s1, mu_f2, sigma_f2, mu_h, sigma_h, mu_l, sigma_l, mu_p, sigma_p = params

    def integrand(x):
        s0, s1, f2, h, l, p = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5]

        h_matrix_p = matrix_product(s0, s1, f2, h, l, p, t, 0)
        l_matrix_p = matrix_product(s0, s1, f2, h, l, p, t, 1)
        p_matrix_p = matrix_product(s0, s1, f2, h, l, p, t, 2)


        marginal =  (distribution(h_matrix_p, mu_h, sigma_h)
                         * distribution(l_matrix_p, mu_l, sigma_l)
                         * distribution(p_matrix_p, mu_p, sigma_p)

                         * distribution(s0, mu_s0, sigma_s0)
                         * distribution(s1, mu_s1, sigma_s1)
                         * distribution(f2, mu_f2, sigma_f2)

                         * det(s0, s1, f2, t))

        #print(marginal)

        return marginal

    domain = [
        [mu_s0 - 6 * sigma_s0, mu_s0 + 6 * sigma_s0],
        [mu_s1 - 6 * sigma_s1, mu_s1 + 6 * sigma_s1],
        [mu_f2 - 6 * sigma_f2, mu_f2 + 6 * sigma_f2],
        [mu_h - 6 * sigma_h, mu_h + 6 * sigma_h],
        [mu_l - 6 * sigma_l, mu_l + 6 * sigma_l],
        [mu_p - 6 * sigma_p, mu_p + 6 * sigma_p],
    ]

    return (MonteCarlo.integrate(integrand, dim=6, N=20000000, integration_domain=domain)).item()



def main():

    """mu_s0, sigma_s0, mu_s1, sigma_s1, mu_f2, sigma_f2, mu_h, sigma_h, mu_l, sigma_l, mu_p, sigma_p = params"""
    initial = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    print(fun1(initial,0,1))


if __name__ == '__main__':
    main()