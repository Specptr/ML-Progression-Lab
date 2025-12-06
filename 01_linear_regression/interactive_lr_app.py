import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QSlider
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from linear_regression import LinearRegressionGD

STYLESHEET = """
QWidget {
    background-color: #1C1C1C;
    color: #ffffff;
}
QPushButton {
    background-color: #3B3B3B;
    color: #FFFFFF;
    border-radius: 10px;
    border: 3px solid #0F0F0F;
    padding: 6px 12px;
}
QLineEdit {
    background-color: #181818;
    color: #ffffff;
    border: 1px solid #999999;
    border-radius: 10px;
    padding: 4px 8px;
    font-size: 20px;
    font-family: Consolas;
}
"""

class InteractiveLRApp(QMainWindow):
    def __init__(self, X, y):
        super().__init__()
        self.X = X
        self.y = y
        self.model = LinearRegressionGD()

        self.setWindowTitle("Interactive Linear Regression -- by EnoLaice")
        self.setGeometry(100, 100, 1200, 1200)

        central_widget = QWidget()
        central_widget.setStyleSheet(STYLESHEET)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        self.fig = plt.figure(figsize=(8, 12), dpi=120)
        self.fig.patch.set_facecolor("black")
        self.ax1 = self.fig.add_subplot(3, 2, 1)  # Linear Fit
        self.ax2 = self.fig.add_subplot(3, 2, 2)  # Cost Convergence
        self.ax3 = self.fig.add_subplot(3, 2, 5, projection='3d')  # Cost Surface
        self.ax4 = self.fig.add_subplot(3, 2, 6)  # Cost Contour

        self.canvas = FigureCanvas(self.fig)
        main_layout.addWidget(self.canvas)

        control_layout = QHBoxLayout()
        main_layout.addLayout(control_layout)

        control_layout.addWidget(QLabel("Learning Rate:"))
        self.alpha_entry = QLineEdit("0.01")
        control_layout.addWidget(self.alpha_entry)

        control_layout.addWidget(QLabel("Iterations:"))
        self.iter_entry = QLineEdit("1500")
        control_layout.addWidget(self.iter_entry)

        self.train_button = QPushButton("Train Model")
        self.train_button.clicked.connect(self.train_model)
        control_layout.addWidget(self.train_button)

        self.output_label = QLabel("")
        control_layout.addWidget(self.output_label)

        self.draw_initial_plot()

    def draw_initial_plot(self):
        # Linear Fit
        self.ax1.clear()
        self.ax1.set_title("Linear Fit", color='white')
        self.ax1.set_facecolor("black")
        self.ax1.tick_params(colors='white')
        self.ax1.spines['bottom'].set_color("#ffffff")
        self.ax1.spines['left'].set_color("#ffffff")
        self.ax1.scatter(self.X, self.y, c="#ff0000a0", s=4, label='data')
        self.ax1.set_xlabel("x", color='white')
        self.ax1.set_ylabel("y", color='white')
        self.ax1.set_facecolor("black")
        self.ax1.tick_params(colors='white')

        # Cost Convergence
        self.ax2.clear()
        self.ax2.set_title("Cost Convergence", color='white')
        self.ax2.set_facecolor("black")
        self.ax2.tick_params(colors='white')
        self.ax2.spines['bottom'].set_color("#ffffff")
        self.ax2.spines['left'].set_color("#ffffff")

        # Cost Surface
        self.ax3.clear()
        self.ax3.set_title("Cost Surface", color='white')
        self.ax3.set_facecolor("black")
        self.ax3.tick_params(colors='white')
        self.ax3.spines['bottom'].set_color("#ffffff")
        self.ax3.spines['left'].set_color("#ffffff")
        gray = (0.2, 0.2, 0.2, 0.2)
        self.ax3.xaxis.set_pane_color(gray)
        self.ax3.yaxis.set_pane_color(gray)
        self.ax3.zaxis.set_pane_color(gray)
        self.ax3.set_xlabel("θ0", color='white')
        self.ax3.set_ylabel("θ1", color='white')
        self.ax3.set_zlabel("J(θ)", color='white')
        self.ax3.set_title("Cost Surface", color='white')
        self.ax3.set_facecolor("black")
        self.ax3.tick_params(colors='white')
        self.ax3.set_xlabel("θ0", color="#ff0000")
        self.ax3.set_ylabel("θ1", color="#0000ff")
        self.ax3.set_zlabel("J(θ)", color="#00ff00")
        self.ax3.xaxis.line.set_color("#FF0000")
        self.ax3.yaxis.line.set_color("#0000FF")
        self.ax3.zaxis.line.set_color("#00FF00")
        self.ax3.grid(False)

        # Cost Contour
        self.ax4.clear()
        self.ax4.set_title("Cost Contour", color='white')
        self.ax4.set_facecolor("black")
        self.ax4.tick_params(colors='white')
        self.ax4.spines['bottom'].set_color("#ffffff")
        self.ax4.spines['left'].set_color("#ffffff")

        self.canvas.draw()

    def train_model(self):
        try:
            alpha = float(self.alpha_entry.text())
            iterations = int(self.iter_entry.text())
        except:
            self.output_label.setText("Invalid input!")
            return

        self.model = LinearRegressionGD(alpha=alpha, iterations=iterations)
        self.model.fit(self.X, self.y)
        y_pred = self.model.predict(self.X)

        t0, t1 = self.model.theta

        # -------- 1. Linear Fit --------
        self.ax1.clear()
        self.ax1.scatter(self.X, self.y, c="#ff0000a0", s=4, label='data')
        self.ax1.plot(self.X, y_pred, color="#3300ff", linewidth=2, label='fit')
        self.ax1.set_title("Linear Fit", color='white')
        if self.model.theta is not None:
            textstr = f"$h_\\theta(x) = {t0:.4f} + {t1:.4f} x$"
            self.ax1.text(0.05, 0.95, textstr, transform=self.ax1.transAxes,
                    fontsize=10, color="#fff000", verticalalignment='top',
                    bbox=dict(facecolor='black', alpha=0.5, edgecolor='white'))
        self.ax1.set_facecolor("black")
        self.ax1.tick_params(colors='white')
        self.ax1.legend()

        # -------- 2. Cost Convergence --------
        self.ax2.clear()
        x = np.arange(len(self.model.cost_history))
        y_cost = np.array(self.model.cost_history)
        self.ax2.scatter(x, y_cost, c=y_cost, cmap="plasma", s=2)
        self.ax2.set_xlabel("Iteration", color='white')
        self.ax2.set_ylabel("Cost J(θ)", color='white')
        self.ax2.set_title("Cost Convergence", color='white')
        self.ax2.set_facecolor("black")
        self.ax2.tick_params(colors='white')

        # -------- 3. Cost Surface 3D --------
        t0 = np.linspace(-50, 50, 50)
        t1 = np.linspace(-5, 5, 50)
        T0, T1 = np.meshgrid(t0, t1)
        J_vals = np.zeros_like(T0)
        X_b = self.model.add_bias(self.X)
        for i in range(len(t0)):
            for j in range(len(t1)):
                theta_ij = np.array([T0[j, i], T1[j, i]])
                J_vals[j, i] = self.model.compute_cost(X_b, self.y, theta_ij)

        self.ax3.clear()
        self.ax3.plot_surface(T0, T1, J_vals, cmap="plasma", alpha=0.85)
        gray = (0.2, 0.2, 0.2, 0.2)
        self.ax3.xaxis.set_pane_color(gray)
        self.ax3.yaxis.set_pane_color(gray)
        self.ax3.zaxis.set_pane_color(gray)
        self.ax3.set_xlabel("θ0", color='white')
        self.ax3.set_ylabel("θ1", color='white')
        self.ax3.set_zlabel("J(θ)", color='white')
        self.ax3.set_title("Cost Surface", color='white')
        self.ax3.set_facecolor("black")
        self.ax3.tick_params(colors='white')
        self.ax3.set_xlabel("θ0", color="#ff0000")
        self.ax3.set_ylabel("θ1", color="#0000ff")
        self.ax3.set_zlabel("J(θ)", color="#00ff00")
        self.ax3.xaxis.line.set_color("#FF0000")
        self.ax3.yaxis.line.set_color("#0000FF")
        self.ax3.zaxis.line.set_color("#00FF00")
        self.ax3.grid(False)

        # -------- 4. Cost Contour --------
        self.ax4.clear()
        CS = self.ax4.contour(T0, T1, J_vals, levels=np.logspace(-2, 3, 20), cmap="plasma")
        self.ax4.clabel(CS, inline=True, fontsize=8)
        self.ax4.scatter(self.model.theta[0], self.model.theta[1], c="#ff0000a0", marker='x', s=80)
        self.ax4.set_xlabel("θ0", color='white')
        self.ax4.set_ylabel("θ1", color='white')
        self.ax4.set_title("Cost Contour", color='white')
        self.ax4.set_facecolor("black")
        self.ax4.tick_params(colors='white')

        self.canvas.draw()
