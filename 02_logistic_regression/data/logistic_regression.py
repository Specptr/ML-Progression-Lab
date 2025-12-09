import numpy as np

def sigmoid(z):
    """
    The sigmoid function:
        g(z) = 1 / (1 + e^(-z))
    Map any real value to (0, 1) representing probability
    """
    return 1 / (1 + np.exp(-z))

def compute_cost(theta, X, y, lam=0):
    """
    Compute the cost function:
        J(θ) = 1/m * Σ [ -y log(hθ(x)) - (1-y) log(1-hθ(x)) ]

    With L2 regularization:
        J_reg = λ/(2m) * Σ θ_j^2   (j from 1...n, NOT including θ0)
    """
    m = len(y)
    h = sigmoid(X @ theta)

    cost = -(1/m) * (y @ np.log(h) + (1-y) @ np.log(1-h))

    reg = (lam/(2*m)) * np.sum(theta[1:] ** 2)

    return cost + reg

def compute_gradient(theta, X, y, lam=0):
    """
    Gradient of cost function

    Without regularization:
        grad_j = 1/m * Σ (hθ(x_i) - y_i) * x_i_j

    With regularization:
        grad_j += λ/m * θ_j     (j >= 1)
            BUT grad_0 never gets regularized.
    """
    m = len(y)
    h = sigmoid(X @ theta)
    error = h - y

    grad = (1/m) * (X.T @ error)

    grad[1:] += (lam/m) * theta[1:] # regularize

    return grad

def gradient_descent(X, y, lam=0, alpha=0.01, iterations=300000, tol=1e-7):
    """
    Gradient descent
    tol: convergence threshold
    """
    theta = np.zeros(X.shape[1])
    cost_history = []

    for _ in range(iterations):
        grad = compute_gradient(theta, X, y, lam)
        theta_new = theta - alpha * grad

        if np.linalg.norm(theta_new - theta) < tol:
            break

        theta = theta_new
        cost_history.append(compute_cost(theta, X, y, lam))

    return theta, cost_history