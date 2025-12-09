# 2025 12 09
import numpy as np
from logistic_regression import gradient_descent, compute_cost
from visualize import plot_decision_boundary, plot_cost_history

def main():
    data = np.loadtxt("data/data1.txt", delimiter=",")
    X = data[:, 0:2]
    y = data[:, 2]

    X_aug = np.column_stack([np.ones(len(X)), X])

    theta, cost_history = gradient_descent(X_aug, y, alpha=0.001, iterations=500000)

    print("Final theta:", theta)
    print("Final cost:", compute_cost(theta, X_aug, y))

    plot_decision_boundary(theta, X_aug, y, nonlinear=False)
    plot_cost_history(cost_history)

if __name__ == "__main__":
    main()
