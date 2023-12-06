import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QWidget,
    QSlider,
    QLabel,
    QHBoxLayout,
)
from PyQt5.QtCore import Qt


class MyGUI(QMainWindow):
    def __init__(self, point_clouds, circles, images):
        super(MyGUI, self).__init__()

        self.point_clouds = point_clouds
        self.circles = circles
        self.images = images
        self.index = 0

        self.initUI()

    def initUI(self):
        self.setGeometry(100, 100, 800, 600)
        self.setWindowTitle("QT GUI with Matplotlib Subplots")

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        layout = QVBoxLayout(self.central_widget)

        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(8, 4))

        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        slider_layout = QHBoxLayout()
        self.slider_label = QLabel("Index:", self)
        slider_layout.addWidget(self.slider_label)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMaximum(len(self.point_clouds) - 1)
        self.slider.valueChanged.connect(self.updatePlot)
        slider_layout.addWidget(self.slider)

        layout.addLayout(slider_layout)

        self.updatePlot()

    def updatePlot(self):
        self.index = self.slider.value()
        self.slider_label.setText(f"Index: {self.index}")

        # Plot left subplot with point clouds and circles
        try:
            self.ax1.clear()
            self.ax1.scatter(*zip(*self.point_clouds[self.index]), label="Point Clouds")
            for circle in self.circles[self.index]:
                self.ax1.add_patch(
                    plt.Circle((circle[0], circle[1]), circle[2], color="r", fill=False)
                )
            self.ax1.set_title("Point Clouds and Circles")
            self.ax1.set_aspect("equal")  # Set aspect ratio to be equal
            self.ax1.legend()
        except Exception as e:
            print(f"Error updating left subplot: {e}")

        # Plot right subplot with image
        self.ax2.clear()
        self.ax2.imshow(self.images[self.index], cmap="gray")
        self.ax2.set_title("Image")

        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Example data
    np.random.seed(42)
    point_clouds = [np.random.rand(10, 2) for _ in range(5)]
    circles = [[(0.5, 0.5, 0.2), (0.3, 0.7, 0.1), (0.8, 0.2, 0.15)] for _ in range(5)]
    images = [np.random.rand(100, 100) for _ in range(5)]

    my_gui = MyGUI(point_clouds, circles, images)
    my_gui.show()

    sys.exit(app.exec_())

# IN DEBUG MODE:
"""
point_clouds = [circ["points"] for circ in cirlce_stack]
circles = [(circ["circle"].x, circ["circle"].y, circ["circle"].radius) for circ in cirlce_stack]
images = [circ["hough_points"] for circ in cirlce_stack]
if len(point_clouds) > 0:
    from PyQt5.QtWidgets import QApplication
    from digiforest_analysis.utils.debug_gui import MyGUI
    import sys
    app = QApplication(sys.argv)
    my_gui = MyGUI(point_clouds, circles, images)
    my_gui.show()
    app.exec_()
else:
    print("No circles found.")
"""
