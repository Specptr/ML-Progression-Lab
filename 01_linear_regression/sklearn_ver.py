# 2025 12 09
import numpy as np
from sklearn.linear_model import LinearRegression

def load_dataset(path="data/data1.txt"):
    """
    Load dataset
    Returns:
        X: shape (m, 1)
        y: shape (m,)
    """
    data = np.loadtxt(path, delimiter=",")
    X = data[:, 0].reshape(-1, 1)
    y = data[:, 1]
    return X, y

def train_model(X, y):
    """
    Fit sklearn's LR model
    sklearn model solves:
        θ = (XᵀX)^(-1) Xᵀ y
    """
    model = LinearRegression()
    model.fit(X, y)
    return model

def main():
    # 1. Load data
    X, y = load_dataset()

    # 2. Train model
    model = train_model(X, y)

    # 3. Retrieve
    theta_0 = model.intercept_
    theta_1 = model.coef_[0]

    print("RESULT")
    print(f"intercept: {theta_0:.6f}")
    print(f"slope: {theta_1:.6f}")

    # 4. predictions
    # skip

    # 5. score
    r2 = model.score(X, y)
    print(f"Model R^2 score: {r2:.6f}")

if __name__ == "__main__":
    main()