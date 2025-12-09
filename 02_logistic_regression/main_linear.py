import numpy as np
from logistic_regression import gradient_descent
from visualize import plot_decision_boundary, plot_cost_history

def main():
    data = np.loadtxt("data/data1.txt", delimiter=",")
    X = data[:, 0:2]
    y = data[:, 2]

    X_aug = np.column_stack([np.ones(len(X)), X])

    theta, cost_history = gradient_descent(X_aug, y, alpha=0.001, iterations=500000)

    plot_decision_boundary(theta, X_aug, y, nonlinear=False)
    plot_cost_history(cost_history)

if __name__ == "__main__":
    main()
