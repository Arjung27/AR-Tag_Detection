import numpy as np
import cv2
import matplotlib.pyplot as plt
from copy import deepcopy
from rectify import *
import glob
from P1 import *

if __name__=="__main__":
    files = open('fileList.txt', 'r')
    lines = [line.rstrip() for line in files.readlines()]
    count = 0
    for file in lines:
        img = cv2.imread(file,cv2.IMREAD_GRAYSCALE)
        ctr = contourDetection(img)
        crnr = getCorners(ctr)
        crnr = np.array(crnr[1])
        crnr = np.squeeze(crnr, 1)
        # if crnr.shape[0] < 4:
        #     continue
        rect_img = rectify(img,crnr[0:4])
        num_rot, rect_img = orient_img(rect_img)
        refArtag = cv2.imread("./reference_images/ref_marker.png")
        for i in range(num_rot):
            refArtag = cv2.rotate(refArtag, cv2.ROTATE_90_CLOCKWISE)
        refImgShape = refArtag.shape
        refImgShape = [512, 512]
        print (refImgShape)
        imgColor = cv2.imread(file)
        cubeHeight = 512
        calib = np.array( [ [1406.08415449821, 0, 0], [2.20679787308599, 1417.99930662800,  0], [1014.13643417416, 566.347754321696,  1] ])
        calib = np.transpose(calib)
        #print(calib)
        reqdPts = np.array([[0, 0, -cubeHeight, 1], [0, refImgShape[1], -cubeHeight, 1],
                            [refImgShape[0], refImgShape[1], -cubeHeight, 1], [refImgShape[0], 0  , -cubeHeight, 1]])
        imgPts = find_cube_pts(imgColor, reqdPts, crnr[0:4], refImgShape, calib)
        # print(imgPts[:]/imgPts[2,:])
        imgPts = imgPts[:]/imgPts[2,:]
        for c in imgPts.T:
            print(c.ravel())
            x, y, _ = c.ravel()
        #     center = (int(imgPts[0,0]), int(imgPts[1,0]))
            print(x, y)
            cv2.circle(imgColor,(int(x), int(y)),10,(0,255,0)) 
        
        draw_cubes(imgColor, crnr[0:4], imgPts)
        cv2.imshow("test",imgColor)
        cv2.waitKey(0)
        # exit(-1)

    	# raw_lena_img = cv2.resize(lenaImg, (256,256)) 