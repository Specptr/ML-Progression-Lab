import numpy as np
from logistic_regression import gradient_descent, compute_cost
from feature_mapping import map_feature
from visualize import plot_decision_boundary


def main():
    data = np.loadtxt("data/data2.txt", delimiter=",")
    x1 = data[:, 0]
    x2 = data[:, 1]
    y = data[:, 2]

    X = map_feature(x1, x2)
    y = y.astype(float)

    lam = 1  # regularization

    theta, cost_history = gradient_descent(
        X, y, lam=lam, alpha=0.1, iterations=100000
    )

    print("Final theta:", theta[:5], " ...")
    print("Final cost:", compute_cost(theta, X, y, lam=lam))

    # Plot nonlinear boundary
    plot_decision_boundary(theta, X, y, nonlinear=True, map_feature=map_feature)


if __name__ == "__main__":
    main()
