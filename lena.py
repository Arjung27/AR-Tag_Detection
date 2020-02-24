import numpy as np
import cv2
import matplotlib.pyplot as plt
from copy import deepcopy
from rectify import *
import glob
from P1 import *

if __name__=="__main__":
    # files = open('fileList.txt', 'r')
    # lines = [line.rstrip() for line in files.readlines()]
    # count = 0
    # for file in lines:
    img = cv2.imread("/home/vishnuu/UMD/ENPM673/Perception_Projects/Project1/AR-Tag-Detection-and-Tracking/VideoFrames/vid150.jpg",cv2.IMREAD_GRAYSCALE)
    ctr = contourDetection(img)
    crnr = getCorners(ctr)
    crnr = np.array(crnr[1])
    crnr = np.squeeze(crnr, 1)
    # if crnr.shape[0] < 4:
    #     continue
    rect_img = rectify(img,crnr[0:4])
    num_rot, rect_img = orient_img(rect_img)
    lenaImg = cv2.imread("/home/vishnuu/UMD/ENPM673/Perception_Projects/Project1/AR-Tag-Detection-and-Tracking/reference_images/Lena.png")
    for i in range(num_rot):
        lenaImg = cv2.rotate(lenaImg, cv2.ROTATE_90_CLOCKWISE)
    raw_lena_img = cv2.resize(lenaImg, (256,256))  
    
    warped_im = warp_lena(cv2.imread("/home/vishnuu/UMD/ENPM673/Perception_Projects/Project1/AR-Tag-Detection-and-Tracking/VideoFrames/vid150.jpg"),raw_lena_img,crnr[0:4])
    cv2.imshow("warped image", warped_im)
    cv2.waitKey(0)  