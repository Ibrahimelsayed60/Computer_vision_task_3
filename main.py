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
import pyqtgraph as pg
import pysift
import features_matching
import SIFT_Descriptor

class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ApplicationWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        path = 'squares.png'
        self.img = cv2.imread(path, 0)
        self.image1 =  cv2.rotate(cv2.imread('SIFT_images/box.png',0),cv2.ROTATE_90_CLOCKWISE)
        self.image2 = cv2.rotate(cv2.imread('SIFT_images/box_in_scene.png',0),cv2.ROTATE_90_CLOCKWISE)
        self.image3 = cv2.rotate(cv2.imread('book.jpg', 0),cv2.ROTATE_90_CLOCKWISE )
        self.image4 = cv2.rotate(cv2.imread('table.jpg', 0),cv2.ROTATE_90_CLOCKWISE )
        
        self.ui.original.setPixmap(QPixmap(path))
        self.ui.comboBox_2.currentIndexChanged[int].connect(self.Features_Matching)
        self.ui.comboBox.currentIndexChanged[int].connect(self.harris_operator)

        self.ui.widget_2.getPlotItem().hideAxis('bottom')
        self.ui.widget_2.getPlotItem().hideAxis('left')
        self.ui.widget_3.getPlotItem().hideAxis('bottom')
        self.ui.widget_3.getPlotItem().hideAxis('left')
        # self.ui.widget.getPlotItem().hideAxis('bottom')
        # self.ui.widget.getPlotItem().hideAxis('left')

        self.ui.pushButton_2.clicked.connect(self.featured_result_image)
        #self.ui.pushButton_1.clicked.connect(self.matching_image)

    def Features_Matching(self):
        if self.ui.comboBox_2.currentIndex() == 0:
            self.ui.widget.clear()
        else:
            sift = cv2.xfeatures2d.SIFT_create()
            keypoints_1, descriptors_1 = sift.detectAndCompute(self.image3,None)
            keypoints_2, descriptors_2 = sift.detectAndCompute(self.image4,None)
            # keypoints_1, descriptors_1 = pysift.computeKeypointsAndDescriptors(self.image3)
            # keypoints_2, descriptors_2 = pysift.computeKeypointsAndDescriptors(self.image4)
            t0= time.process_time()
            if self.ui.comboBox_2.currentIndex() == 1:
                matching, trainIdx = features_matching.SSD(descriptors_1, descriptors_2)
            if self.ui.comboBox_2.currentIndex() == 2:
                matching, trainIdx = features_matching.NCC(descriptors_1, descriptors_2)
            newimg = features_matching.get_matching_image(self.image3, self.image4, keypoints_1, keypoints_2, trainIdx)
            img = pg.ImageItem(newimg)
            self.ui.widget_2.addItem(img)
            t1 = time.process_time() - t0
            self.ui.widget_2.setTitle("Computation time = {}".format(t1))
            print("Computation time: ", t1)

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
        self.ui.image.setTitle("Computation time = {}".format(t1))        

    def featured_result_image(self):
        #newimage = SIFT_Descriptor.featured_image(self.image1)
        newimage = pysift.featured_image(self.image1)
        img = pg.ImageItem(newimage)
        self.ui.widget_3.addItem(img)

    # def matching_image(self):
    #     newimage = pysift.SIFT_matching_keypoint(self.image1,self.image2)
    #     img = pg.ImageItem(newimage)
    #     self.ui.widget_2.addItem(img)



def main():
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    app.exec_()


if __name__ == "__main__":
    main()
