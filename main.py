from PyQt5 import QtWidgets, QtGui
from PyQt5.QtGui import QPixmap,QImage
from PyQt5.QtCore import Qt
from task import Ui_MainWindow
import sys
import random
import cv2
import pyqtgraph as pg
import numpy as np
import harris


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ApplicationWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        path = 'squares.png'
        self.img = cv2.imread(path, 0)
        
        self.ui.original.setPixmap(QPixmap(path))
        self.ui.comboBox.currentIndexChanged[int].connect(self.harris_operator)


    def harris_operator(self):
        if self.ui.comboBox.currentIndex() == 1:
            
            #Read Image
            firstimagename = 'squares.png'
            
            # Get the first image
            firstimage = cv2.imread(firstimagename, 0)
            w, h = firstimage.shape

            # Covert image to color to draw colored circles on it
            bgr = cv2.cvtColor(firstimage, cv2.COLOR_GRAY2RGB)

            # Corner detection
            harris_output = harris.HarrisCornerDetection(firstimage)

            #Parameter

            CornerStrengthThreshold = 500

            # Plot detected corners on image
            radius = 1
            color = (0, 255, 0)
            thickness = 1

            PointList = []
            # Look for Corner strengths above the threshold
            for row in range(w):
                for col in range(h):
                    if harris_output[row][col] > CornerStrengthThreshold:
                        # print(R[row][col])
                        max = harris_output[row][col]

                        # Local non-maxima suppression
                        skip = False
                        for nrow in range(5):
                            for ncol in range(5):
                                if row + nrow - 2 < w and col + ncol - 2 < h:
                                    if harris_output[row + nrow - 2][col + ncol - 2] > max:
                                        skip = True
                                        break

                        if not skip:
                            # Point is expressed in x, y which is col, row
                            cv2.circle(bgr, (col, row), radius, color, thickness)
                            PointList.append((row, col))

        self.my_img = pg.ImageItem(harris_output)
        self.ui.image.addItem(self.my_img)
        

def main():
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    app.exec_()


if __name__ == "__main__":
    main()
