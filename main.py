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
            threshold_log10 = 3
            threshold = 10**(threshold_log10)
            #img = load_image('squares.png')[:,:,np.newaxis]
            img = cv2.imread('squares.png')
            responses = harris.get_responses(img)
            responses = np.where(responses>threshold,responses,-1)
            #rows, cols = harris.non_maxima_suppression(responses, 13)
            harris_image = harris.non_maxima_suppression(responses, 13)
            color_img = cv2.cvtColor(img[:,:,0], cv2.COLOR_GRAY2BGR)
        self.my_img = pg.ImageItem(harris_image)
        self.ui.image.addItem(self.my_img)
        

                #for each_corner in range(len(rows)):
                 #   cv2.circle(color_img, (cols[each_corner], rows[each_corner]), 3, (0,0,255), -1)
                #cv2.imwrite('Q3-Output/corners.jpg', color_img)


    


def main():
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    app.exec_()


if __name__ == "__main__":
    main()
