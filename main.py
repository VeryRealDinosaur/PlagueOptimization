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

        if case == 0:
            return h * marginal
        elif case == 1:
            return l * marginal
        elif case == 2:
            return p * marginal

    domain = [
        [mu_s0 - 6 * sigma_s0, mu_s0 + 6 * sigma_s0],
        [mu_s1 - 6 * sigma_s1, mu_s1 + 6 * sigma_s1],
        [mu_f2 - 6 * sigma_f2, mu_f2 + 6 * sigma_f2],
        [mu_h - 6 * sigma_h, mu_h + 6 * sigma_h],
        [mu_l - 6 * sigma_l, mu_l + 6 * sigma_l],
        [mu_p - 6 * sigma_p, mu_p + 6 * sigma_p],
    ]

    return (MonteCarlo.integrate(integrand, dim=6, N=10000000, integration_domain=domain)).item()


def addition(params, h, l, p, t):

    suma = torch.tensor(0.0, requires_grad=True)

    file_path = "/Users/jovany/PycharmProjects/PlagueOptimization/OptimumValues.txt"
    with open(file_path, "a") as file:

        for i in range(len(h)):
            term_h = (h[i] - fun1(params, t[i], 0)) ** 2
            term_l = (l[i] - fun1(params, t[i], 1)) ** 2
            term_p = (p[i] - fun1(params, t[i], 2)) ** 2

            #print(fun1(params, t[i], 0))
            #print(fun1(params, t[i], 1))
            #print(fun1(params, t[i], 2))


            suma = suma + term_h + term_l + term_p

        print_registry(params, suma, file)

    return suma.detach().numpy()


def print_registry(params, function_value, file):

    mu_s0, sigma_s0, mu_s1, sigma_s1, mu_f2, sigma_f2, mu_h, sigma_h, mu_l, sigma_l, mu_p, sigma_p = params

    print(f"Function value: {function_value}")
    print("Parameters:")
    print(f"  H (μ₀: {mu_s0:8.4f}, σ₀: {sigma_s0:8.4f})")
    print(f"  L (μ₁: {mu_s1:8.4f}, σ₁: {sigma_s1:8.4f})")
    print(f"  P (μ₂: {mu_f2:8.4f}, σ₂: {sigma_f2:8.4f})")
    print(f"  P0 (μ₃: {mu_h:8.4f}, σ₃: {sigma_h:8.4f})")
    print(f"  P0 (μ₃: {mu_l:8.4f}, σ₃: {sigma_l:8.4f})")
    print(f"  P0 (μ₃: {mu_p:8.4f}, σ₃: {sigma_p:8.4f})")
    print("-" * 60)

    file.write(f"Function value: {function_value}\n")
    file.write("Parameters:\n")
    file.write(f"  H (μ₀: {mu_s0:8.4f}, σ₀: {sigma_s0:8.4f})\n")
    file.write(f"  L (μ₁: {mu_s1:8.4f}, σ₁: {sigma_s1:8.4f})\n")
    file.write(f"  P (μ₂: {mu_f2:8.4f}, σ₂: {sigma_f2:8.4f})\n")
    file.write(f"  P0 (μ₃: {mu_h:8.4f}, σ₃: {sigma_h:8.4f})\n")
    file.write(f"  P0 (μ₃: {mu_l:8.4f}, σ₃: {sigma_l:8.4f})\n")
    file.write(f"  P0 (μ₃: {mu_p:8.4f}, σ₃: {sigma_p:8.4f})\n")
    file.write("-" * 60 + "\n")


def main():

    """mu_s0, sigma_s0, mu_s1, sigma_s1, mu_f2, sigma_f2, mu_h, sigma_h, mu_l, sigma_l, mu_p, sigma_p = params"""
    initial = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    #initial = np.array([1.8836, 13.2774, 25.3230, 2.1342, 5.9543, 96.5007, 10, 20, 10, 5, 60, 20])
    #initial = initial * np.random.uniform(0.5, 1.5, size=initial.shape)

    print("Valores")
    print(initial)


    result = optimize.minimize(
        fun=addition,
        x0=initial,
        args=(H, L, P, T),
        method='nelder-mead',
        options = {'adaptive': True}
    )
    print("\nOptimización Completada:")
    print(result)


if __name__ == '__main__':
    main()