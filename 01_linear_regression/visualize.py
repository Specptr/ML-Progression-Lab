import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def plot_univariate(X, y, y_pred, cost_history, model, theta=None):
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(12, 8))
    fig.patch.set_facecolor("black")
    gs = GridSpec(2, 2, figure=fig, wspace=0.5, hspace=0.5)

    # -------- 1. Linear Fit --------
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.spines['bottom'].set_color("#ffffff")
    ax1.spines['left'].set_color("#ffffff")
    ax1.scatter(X, y, c="#ff0000a0", s=4, label='data')
    ax1.plot(X, y_pred, color="#3300ff", linewidth=2, label='fit')
    if theta is not None:
        theta0, theta1 = theta
        textstr = f"$h_\\theta(x) = {theta0:.4f} + {theta1:.4f} x$"
        ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes,
                 fontsize=10, color="#fff000", verticalalignment='top',
                 bbox=dict(facecolor='black', alpha=0.5, edgecolor='white'))
    ax1.set_xlabel("x", color='white')
    ax1.set_ylabel("y", color='white')
    ax1.set_title("Linear Fit", color='white')
    ax1.legend()
    ax1.set_facecolor("black")
    ax1.tick_params(colors='white')

    # -------- 2. Cost Surface 3D --------
    t0 = np.linspace(-60, 60, 50)
    t1 = np.linspace(-8, 8, 50)
    T0, T1 = np.meshgrid(t0, t1)
    J_vals = np.zeros_like(T0)
    X_b = model.add_bias(X)
    for i in range(len(t0)):
        for j in range(len(t1)):
            theta_ij = np.array([T0[j, i], T1[j, i]])
            J_vals[j, i] = model.compute_cost(X_b, y, theta_ij)

    ax2 = fig.add_subplot(gs[1, 0], projection='3d')
    surf = ax2.plot_surface(T0, T1, J_vals, cmap="plasma", alpha=0.85)
    ax2.set_xlabel("θ0", color="#ff0000")
    ax2.set_ylabel("θ1", color="#0000ff")
    ax2.set_zlabel("J(θ)", color="#00ff00")
    gray = (0.2, 0.2, 0.2, 0.2)
    ax2.xaxis.set_pane_color(gray)
    ax2.yaxis.set_pane_color(gray)
    ax2.zaxis.set_pane_color(gray)
    ax2.xaxis.line.set_color("#FF0000")
    ax2.yaxis.line.set_color("#0000FF")
    ax2.zaxis.line.set_color("#00FF00")
    ax2.grid(False)
    ax2.set_title("Cost Surface", color='white')
    ax2.set_facecolor("black")
    ax2.tick_params(colors='white')

    # -------- 3. Cost Convergence --------
    ax3 = fig.add_subplot(gs[0, 1])
    ax3.spines['bottom'].set_color("#ffffff")
    ax3.spines['left'].set_color("#ffffff")
    x = np.arange(len(cost_history))
    y = np.array(cost_history)
    ax3.scatter(x, y, c=y, cmap="plasma", s=2)
    ax3.set_xlabel("Iteration", color='white')
    ax3.set_ylabel("Cost Function J(θ)", color='white')
    ax3.set_title("Cost Convergence", color='white')
    ax3.set_facecolor("black")
    ax3.tick_params(colors='white')


    # -------- 4. Cost Contour --------
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.spines['bottom'].set_color("#ffffff")
    ax4.spines['left'].set_color("#ffffff")
    CS = ax4.contour(T0, T1, J_vals, levels=np.logspace(-2, 3, 20), cmap="plasma")
    ax4.clabel(CS, inline=True, fontsize=8)
    ax4.scatter(model.theta[0], model.theta[1], c="#ff0000a0", marker='x', s=80)
    ax4.set_xlabel("θ0", color='white')
    ax4.set_ylabel("θ1", color='white')
    ax4.set_title("Cost Contour", color='white')
    ax4.set_facecolor("black")
    ax4.tick_params(colors='white')

    plt.show()

def plot_multivariate(X, y, y_pred, cost_history):
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(12, 5))
    fig.patch.set_facecolor("black")
    gs = GridSpec(1, 2, figure=fig, wspace=0.5, hspace=0.5)

    # -------- 1. Data vs Prediction (scatter) --------
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.spines['bottom'].set_color("#ffffff")
    ax1.spines['left'].set_color("#ffffff")
    ax1.scatter(X[:,0], y, c="#ff0000a0", s=4, label='true')
    ax1.scatter(X[:,0], y_pred, c="#000dffff", s=4, label='pred')
    ax1.set_xlabel("First feature", color='white')
    ax1.set_ylabel("Target", color='white')
    ax1.set_title("Data vs Prediction", color='white')
    ax1.legend()
    ax1.set_facecolor("black")
    ax1.tick_params(colors='white')

    # -------- 2. Cost Convergence --------
    iterations = np.arange(len(cost_history))
    y = np.array(cost_history)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.spines['bottom'].set_color("#ffffff")
    ax2.spines['left'].set_color("#ffffff")

    sc = ax2.scatter(iterations, y, c=y, cmap="plasma", s=8)

    ax2.set_xlabel("Iteration", color='white')
    ax2.set_ylabel("Cost Function J(θ)", color='white')
    ax2.set_title("Cost Convergence", color='white')
    ax2.set_facecolor("black")
    ax2.tick_params(colors='white')


    plt.show()
