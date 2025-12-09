import numpy as np
import matplotlib as plt
from logistic_regression import gradient_descent, compute_cost
from visualize import plot_data, plot_decision_boundary

def main():
    data = np.loadtxt("data/data1.txt", delimiter=",")
    X = data[:, 0:2]
    y = data[:, 2]

    X_aug = np.column_stack([np.ones(len(X)), X])

    theta, cost_history = gradient_descent(X_aug, y, alpha=0.001, iterations=500000)

    plot_decision_boundary(theta, X_aug, y, nonlinear=False)

if __name__ == "__main__":
    main()
