import numpy as np
import matplotlib.pyplot as plt

def plot_data(X, y):
    pos = y == 1
    neg = y == 0

    plt.scatter(X[pos, 0], X[pos, 1], c='blue', marker='+', label='Admitted')
    plt.scatter(X[neg, 0], X[neg, 1], c='red', marker='+', label='Not admitted')
    plt.legend()


def plot_decision_boundary(theta, X, y, nonlinear=False, map_feature=None):
    plt.style.use('dark_background')
    fig, ax = plt.subplots()
    ax.set_facecolor('black')
    ax.tick_params(colors='white')  # 坐标刻度颜色
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')

    plot_data(X[:, 1:3] if not nonlinear else X[:, 1:3], y)

    if not nonlinear:
        # θ0 + θ1*x1 + θ2*x2 = 0 → x2 = -(θ0 + θ1*x1)/θ2
        x_vals = np.linspace(min(X[:, 1]), max(X[:, 1]), 100)
        y_vals = -(theta[0] + theta[1] * x_vals) / theta[2]
        plt.plot(x_vals, y_vals, 'g-', label='Decision Boundary')
        plt.legend()
    else:
        # nonlinear decision boundary via contour
        u = np.linspace(-1, 1.5, 200)
        v = np.linspace(-1, 1.5, 200)
        z = np.zeros((len(u), len(v)))

        for i in range(len(u)):
            for j in range(len(v)):
                mapped = map_feature(np.array([u[i]]), np.array([v[j]]))
                z[i, j] = mapped @ theta

        plt.contour(u, v, z.T, levels=[0], linewidths=2.5, colors='green')

    plt.xlabel("Exam 1 Score")
    plt.ylabel("Exam 2 Score")
    plt.title("Decision Boundary")
    plt.show()
