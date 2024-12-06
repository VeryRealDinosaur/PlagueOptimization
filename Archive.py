"""from torchquad import set_up_backend
import numpy as np
from scipy import optimize
from torchquad.integration.gaussian import GaussLegendre

from BaseComponents import *

set_up_backend("torch", data_type="float64")
Gauss = GaussLegendre()

H=[9, 17, 18, 26, 23, 20, 19, 12, 6, 10, 1, 1, 1]
L=[3, 14, 50, 74, 56, 40, 126, 40, 37, 36, 48, 76, 91]
P=[1, 1, 2, 1, 1, 1, 3, 4, 2, 1, 5, 2, 9]
T=[7, 10, 14, 17, 21, 25, 29, 32, 36, 40, 45, 52, 57]

def mu(variable):
    sum = 0
    for i in range (len(variable)):
        sum += variable
    return sum/len(variable)

def fun1_h(mu_s0, sigma_s0, mu_s1, sigma_s1, mu_f2, sigma_f2, mu_h, sigma_h, mu_l, sigma_l, mu_p, sigma_p, t, domain):

    def integrand(x):
        s0, s1, f2, h, l, p = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5]

        h_matrix_p = matrix_product(s0, s1, f2, torch.atan(h), torch.atan(l), torch.atan(p), t, 1)
        l_matrix_p = matrix_product(s0, s1, f2, torch.atan(h), torch.atan(l), torch.atan(p), t, 2)
        p_matrix_p = matrix_product(s0, s1, f2, torch.atan(h), torch.atan(l), torch.atan(p), t, 3)


        marginalizada =  (distribution(h_matrix_p, mu_h, sigma_h)
                         * distribution(l_matrix_p, mu_l, sigma_l)
                         * distribution(p_matrix_p, mu_p, sigma_p)

                         * distribution(s0, mu_s0, sigma_s0)
                         * distribution(s1, mu_s1, sigma_s1)
                         * distribution(f2, mu_f2, sigma_f2))

        result = torch.atan(h) * marginalizada / (1 + h ** 2) * (1 + l ** 2) * (1 + p ** 2)

        return result

    result = Gauss.integrate(integrand, dim=6, integration_domain=domain)
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
            penalty = 1e5 * (float(abs(term_h) < 1e-10) + float(abs(term_l) < 1e-10) + float(abs(term_p) < 1e-10))
            suma = suma + term_h + term_l + term_p + penalty

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


    initial = np.array([1.8836, 13.2774, -250.3230, 2.1342, 5.9543, 96.5007, 60, 71.9338])



    print(p0(900,901,902,2,3,4,5))

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
    main()"""