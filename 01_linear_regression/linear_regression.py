import numpy as np

class LinearRegressionGD:
    """
    Linear Regression using Gradient Descent
    """

    def __init__(self, alpha=0.01, iterations=1500):
        self.alpha = alpha
        self.iterations = iterations
        self.theta = None
        self.cost_history = []

    @staticmethod
    def add_bias(X):
        """
        Add bias term to feature matrix: column of ones
        """
        return np.c_[np.ones(len(X)), X]

    @staticmethod
    def compute_cost(X, y, theta):
        """
        Compute the Mean Squared Error cost function:
            J(θ) = 1/(2m) * Σ(hθ(x_i) - y_i)^2
            J(θ) = 1/(2m) * Σ(Xθ - y)^2

        - X: an mx2 matrix
        - y: target vector
        - θ: parameter vector
        """
        m = len(y)
        predictions = X.dot(theta)
        errors = predictions - y

        return (1 / (2 * m)) * np.sum(errors ** 2)

    def fit(self, X, y):
        """
        Train the model using gradient descent
        Gradient descent update rule:
            θ := θ - α * 1/m * Xᵀ (Xθ - y)
        """
        X_b = self.add_bias(X)
        m, n = X_b.shape

        self.theta = np.zeros(n)

        for _ in range(self.iterations):
            gradients = (1 / m) * X_b.T.dot(X_b.dot(self.theta) - y)
            self.theta -= self.alpha * gradients
            self.cost_history.append(self.compute_cost(X_b, y, self.theta))

        return self

    def predict(self, X):
        """
        Predict using learned parameters θ
        """
        X_b = self.add_bias(X)
        return X_b.dot(self.theta)