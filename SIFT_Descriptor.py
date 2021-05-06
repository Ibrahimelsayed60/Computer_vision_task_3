import numpy as np
import cv2
from scipy import signal
#import Harris_Corner_Detector as HC
import harris
import time
import matplotlib.pyplot as plt



def sobel_edge_of_image(image):
    kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    ky = (kx.transpose())

    Int_x = signal.convolve2d(image,kx,mode='same',boundary='symm')
    Int_y = signal.convolve2d(image, ky,mode='same',boundary = 'symm')

    ########### Get the magnitude and phase image #######################
    mag = np.sqrt( (Int_x*Int_x) + (Int_y*Int_y))
    phase = np.rad2deg( np.arctan2(Int_y, Int_x)) % 360

    return mag, phase

def gaussian_filter(shape,sigma):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h



# Descriptor Generation
def feature_descriptor(kepy_points, img, time_start, num_subregion=4, num_bin=8):
    descriptors = []
    sigma = 1.5
    kernel = gaussian_filter((16, 16), sigma)
    for kp in kepy_points:
        x, y, dominant_angle = int(kp[0]), int(kp[1]), kp[2]

        # 16 X 16 window around the key point
        sub_img = img [x - 8:x + 8, y - 8:y + 8]
        mag, dir = sobel_edge(sub_img)
        weighted_mag = np.multiply(mag, kernel)
        # subtract the dominant direction
        dir = (((dir - dominant_angle) % 360) * num_bin / 360.).astype(int)
        features = []
        for sub_i in range(num_subregion):
            for sub_j in range(num_subregion):
                sub_weights = weighted_mag[sub_i * 4:(sub_i + 1) * 4, sub_j * 4:(sub_j + 1) * 4]
                sub_dir_idx = dir[sub_i * 4:(sub_i + 1) * 4, sub_j * 4:(sub_j + 1) * 4]
                hist = np.zeros(num_bin, dtype=np.float32)
                for bin_idx in range(num_bin):
                    hist[bin_idx] = np.sum(sub_weights[sub_dir_idx == bin_idx])
                features.extend(hist.tolist())
        features /= (np.linalg.norm(np.array(features)))
        features = np.clip(features, np.finfo(np.float16).eps, 0.2)
        features /= (np.linalg.norm(features))
        descriptors.append(features)
    feature_desc_time_end = time.time()
    print(f"Execution time of the feature descriptor generation is {feature_desc_time_end - time_start}  sec")
    return descriptors


def orientation_keypoints(keypoints,img,bins_num = 36):
    #sigma = 1.5
    new_keypoint = []

    #### Divide image to sub images 
    Max_size_window = 3
    SIGMA = 1.6
    sigma = SIGMA * 1.5
    w = int(2 * np.ceil(sigma) + 1)
    g_kernel =gaussian_filter((w,w), sigma)
    bins_width = 360 // bins_num
    

    for kp in keypoints:
        x = int(kp[0])
        y = int(kp[1])
        hist = np.zeros(bins_num, dtype=np.float32)

        #sub_image_result =sub_image(image, x, y)
        sub_image_result = np.array([[img[x-3][y-3], img[x-3][y-2], img[x-3][y-1], img[x-3][y], img[x-3][y+1], img[x-3][y+2], img[x-3][y+3]],
                       [img[x-2][y-3], img[x-2][y-2], img[x-2][y-1], img[x-2][y], img[x-2][y+1], img[x-2][y+2], img[x-2][y+3]],
                       [img[x-1][y-3], img[x-1][y-2], img[x-1][y-1], img[x-1][y], img[x-1][y+1], img[x-1][y+2], img[x-1][y+3]],
                       [img[x][y-3],   img[x][y-2],   img[x][y-1],   img[x][y],   img[x][y+1],   img[x][y+2],   img[x][y+3]],
                       [img[x+1][y-3], img[x+1][y-2], img[x+1][y-1], img[x+1][y], img[x+1][y+1], img[x+1][y+2], img[x+1][y+3]],
                       [img[x+2][y-3], img[x+2][y-2], img[x+2][y-1], img[x+2][y], img[x+2][y+1], img[x+2][y+2], img[x+2][y+3]],
                       [img[x+3][y-3], img[x+3][y-2], img[x+3][y-1], img[x+3][y], img[x+3][y+1], img[x+3][y+2], img[x+3][y+3]]
                      ])

        #print(sub_image_result.shape)
        magnitude, phase = sobel_edge_of_image(sub_image_result)
        #print(magnitude.shape)
        #print(g_kernel.shape)

        weighted_image = np.multiply(magnitude , g_kernel)


        for i in range(0, len(phase)):
            for j in range(0, len(magnitude)):
                bin_indx = int(np.floor(phase[i][j]) // bins_width)
                hist[bin_indx] += weighted_image[i][j]

        max_bin = np.argmax(hist)
        new_keypoint.append([kp[0], kp[1], hist[max_bin]])
        # finding new key points (value > 0.8 * max_val)
        max_val = np.max(hist)
        for bin_no, val in enumerate(hist):
            if bin_no == max_bin: # dominant direction
                continue
            if .8 * max_val <= val:
                new_keypoint.append([kp[0], kp[1], hist[bin_no]])

    n_keypoints = np.array(new_keypoint)

    return n_keypoints

def featured_image(image):
    start_time = time.process_time()
    kp1 = orientation_keypoints(R,image)

    image_2 = image.copy()
    image_2_point = cv2.drawKeypoints(image,kp1,outImage = image_2,color=(255,0,0))

    t1 = time.process_time() - start_time
    print("Computation time: ", t1)

    return image_2_point


# tests
img = cv2.imread('Cow.png', cv2.IMREAD_GRAYSCALE)

# R = HC.HarrisCornerDetection(img)
R = harris.HarrisCornerDetection(img)
kps = orientation_keypoints(R, img)

new_image = featured_image(img)

plt.figure(figsize=(20,20))
plt.imshow(new_image)

# desc = feature_descriptor(kps, img, time_start)
print(kps)

