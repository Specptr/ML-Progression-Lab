# 2025.12.6
import sys
import numpy as np
from PyQt5.QtWidgets import QApplication
from interactive_lr_app import InteractiveLRApp

if __name__ == "__main__":
    data = np.loadtxt("data/data1.txt", delimiter=',')
    X = data[:, 0]
    y = data[:, 1]

    app = QApplication(sys.argv)
    window = InteractiveLRApp(X, y)
    window.show()
    sys.exit(app.exec_())
