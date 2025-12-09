# 2025 12 09
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score

def main():
    # Load data
    data = np.loadtxt("data/data2.txt", delimiter=",")
    x1 = data[:, 0]
    x2 = data[:, 1]
    y = data[:, 2]

    # Feature mapping
    poly = PolynomialFeatures(degree=6, include_bias=True)
    X_mapped = poly.fit_transform(np.column_stack([x1, x2]))

    # Logistic Regression
    lam = 1
    C_value = 1 / lam

    model = LogisticRegression(
        penalty="l2",
        C=C_value,
        solver="lbfgs",
        max_iter=5000
    )
    model.fit(X_mapped, y)

    # Predictions
    preds = model.predict(X_mapped)

    # Accuracy
    acc = accuracy_score(y, preds)

    print("RESULT")
    print("Degree of polynomial: 6")
    print("Lambda:", lam)
    print("First 5 theta values:\n", model.coef_[0][5])
    print("Training accuracy:", acc)

if __name__ == "__main__":
    main()