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
import time


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
        t0= time.process_time()
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

            #Parameter setting depending on the image
            CornerStrengthThreshold = 5

            # Plot detected corners on image
            radius = 3
            color = (255, 255, 255) #white
            thickness = 1
            PointList = []

            # Look for Corner strengths above the threshold
            for row in range(w):
                for col in range(h):
                    if harris_output[row][col] > CornerStrengthThreshold:
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
        
        t1 = time.process_time() - t0
        print("Computation time: ", t1)
        self.my_img = pg.ImageItem(harris_output)
        self.ui.image.addItem(self.my_img)
        

def main():
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    app.exec_()


if __name__ == "__main__":
    main()
