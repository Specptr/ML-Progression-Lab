# 2025.12.5
import numpy as np
from linear_regression import LinearRegressionGD
from visualize import plot_univariate

ALPHA = 0.01
ITERATIONS = 1500

def load_dataset():
    data = np.loadtxt("data/data1.txt", delimiter=",")
    X = data[:, 0]
    y = data[:, 1]
    return X, y

def main():
    X, y = load_dataset()

    # Train model
    model = LinearRegressionGD(alpha=ALPHA, iterations=ITERATIONS)
    model.fit(X, y)

    # Predictions
    y_pred = model.predict(X)

    plot_univariate(X, y, y_pred, model.cost_history, model, theta=model.theta)
    print("θ learned:", model.theta)

if __name__ == "__main__":
    main()
