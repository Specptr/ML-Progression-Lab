# 2025.12.5
import numpy as np
from feature_normalize import feature_normalize
from linear_regression import LinearRegressionGD
from visualize import plot_multivariate

ALPHA = 0.01
ITERATIONS = 1500

def load_dataset():
    data = np.loadtxt("data/data2.txt", delimiter=",")
    X = data[:, 0:2]
    y = data[:, 2]
    return X, y

def main():
    X, y = load_dataset()

    X_norm, _, _ = feature_normalize(X)

    model = LinearRegressionGD(alpha=ALPHA, iterations=ITERATIONS)
    model.fit(X_norm, y)
    y_pred = model.predict(X_norm)

    print("Learned parameters (theta):")
    print(model.theta)

    plot_multivariate(X_norm, y, y_pred, model.cost_history)

if __name__ == "__main__":
    main()
