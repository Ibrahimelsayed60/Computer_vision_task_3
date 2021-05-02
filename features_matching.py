import cv2
import numpy as np

def SSD(descriptors_1, descriptors_2):
    matching = []
    trainIdx = []
    for k in range(descriptors_1.shape[0]):
        ssd = []
        for i in range(0 , descriptors_2.shape[0]):
            x = 0
            for j in range(0, descriptors_1.shape[1]):
                diff = descriptors_1[k][j] - descriptors_2[i][j]
                x += diff*diff
            ssd.append(x/(128))
        # print(int(min(ssd)))
        trainIdx.append(ssd.index(min(ssd)))
        matching.append(int(min(ssd)))
    return matching, trainIdx

def NCC(descriptors_1, descriptors_2):
    mean_1 = ((descriptors_1.mean(axis=-1)).mean(axis=-1)).reshape(-1,1,1)
    mean_2 = ((descriptors_2.mean(axis=-1)).mean(axis=-1)).reshape(-1,1,1)

    mean_1_sub = descriptors_1 - mean_1
    mean_2_sub = descriptors_2 - mean_2

    matching = []
    trainIdx = []
    for k in range(descriptors_1.shape[0]):
        NCC = []
        for i in range(0 , descriptors_2.shape[0]):
            x = 0
            y1 = 0
            y2 = 0
            for j in range(0, descriptors_1.shape[1]):
                x += mean_1_sub[0][k][j] * mean_2_sub[0][i][j]
                y1 += mean_1_sub[0][k][j]**2
                y2 += mean_2_sub[0][i][j]**2
            NCC.append(x/np.sqrt(y1*y2))
        trainIdx.append(NCC.index(max(NCC)))
        matching.append(int(max(NCC)))
    return matching, trainIdx

def get_matching_image(img1, img2, keypoints_1, keypoints_2, trainIdx):
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    nWidth = w1 + w2
    nHeight = max(h1, h2)
    hdif = int((h2 - h1) / 2)
    newimg = np.zeros((nHeight + 20, nWidth, 3), np.uint8)
    for i in range(3):
        newimg[hdif:hdif + h1, :w1, i] = img1
        newimg[:h2, w1:w1 + w2, i] = img2
    # Draw SIFT keypoint matches
    for m in range(80):
        pt1 = (int(keypoints_1[m].pt[0]), int(keypoints_1[m].pt[1] + hdif))
        pt2 = (int(keypoints_2[trainIdx[m]].pt[0] + w1), int(keypoints_2[trainIdx[m]].pt[1]))
        cv2.line(newimg, pt1, pt2, (255, 0, 0))
    return newimg
