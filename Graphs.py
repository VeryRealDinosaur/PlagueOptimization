from torchquad import set_up_backend
import numpy as np
from torchquad.integration.gaussian import GaussLegendre
import matplotlib.pyplot as plt
from BaseComponents_Graph import *
import torch
import plotly.graph_objects as go

set_up_backend("torch", data_type="float64")
Gauss = GaussLegendre()


def fun1_h(h, t):
    mu_0 = 1.8848
    sigma_0 = 13.2764
    mu_1 = -250.3746
    sigma_1 = 2.1980
    mu_2 = 5.9593
    sigma_2 = 96.4816
    mu_3 = 0.2244
    sigma_3 = 71.9353

    def integrand(x):
        s0, s1, f2, l, p = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4]

        # Now using single values h and t directly
        result = ((distribution(s0, mu_0, sigma_0)
                   * distribution(s1, mu_1, sigma_1) * distribution(f2, mu_2, sigma_2)
                   * distribution(p0(s0, s1, f2, h, torch.atan(l), torch.atan(p), t), mu_3, sigma_3)
                   * det(s0, s1, f2, t))) / ((1 + l ** 2) * (1 + p ** 2))

        return result

    domain = [
        [mu_0 - 6 * sigma_0, mu_0 + 6 * sigma_0],
        [mu_1 - 6 * sigma_1, mu_1 + 6 * sigma_1],
        [mu_2 - 6 * sigma_2, mu_2 + 6 * sigma_2],
        [-np.pi / 2, np.pi / 2],
        [-np.pi / 2, np.pi / 2]
    ]

    N = 300000
    result = Gauss.integrate(integrand, dim=5, N=N, integration_domain=domain)
    return result.item()


def main():
    # Create a finer mesh for better visualization
    x = np.linspace(-1, 1, 6)  # h values
    y = np.linspace(0, 6, 6)  # t values

    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            h_val = float(X[i, j])
            t_val = int(Y[i, j])
            Z[i, j] = fun1_h(h_val,t_val)
            print(f"Computing for h={h_val:.2f}, t={t_val:.2f}")

    # Plotting the result with better visualization
    fig = go.Figure()

    fig.add_trace(go.Surface(
        z=Z,
        x=X,
        y=Y,
        colorscale='Viridis',
        colorbar_title='Distribution'
    ))

    # Set axis labels and title
    fig.update_layout(
        scene=dict(
            xaxis_title='x',
            yaxis_title='y',
            zaxis_title='Distribution'
        ),
        title='Population Stage Distribution',
        width=800,
        height=600
    )

    # Show the interactive plot
    fig.show()


if __name__ == '__main__':
    main()