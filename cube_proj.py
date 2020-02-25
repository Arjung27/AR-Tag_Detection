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
		refImgShape = refArtag.shape()
		imgColor = cv2.imread(file)
		cubeHeight = 20
		reqdPts = np.array([[refImgShape[0], refImgShape[0], -cubeHeight ], [refImgShape[0], refImgShape[1], -cubeHeight],
						 [refImgShape[1], refImgShape[1], -cubeHeight], [refImgShape[1], refImgShape[0], -cubeHeight]])
		imgPts = find_cube_pts(imgColor, reqdPts, crnr[0:4], refImgShape)
		

    	# raw_lena_img = cv2.resize(lenaImg, (256,256)) 