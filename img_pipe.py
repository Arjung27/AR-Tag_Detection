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
        if crnr.shape[0] < 4:
            continue
        rect_img = rectify(img,crnr[0:4])
        num_rot, rect_img = orient_img(rect_img)
        id = find_id(rect_img)
        rect_img = rect_img.astype(np.uint8)
        print(id)
        if id != 15:
            count += 1
        cv2.imshow("rectified image",rect_img)
        cv2.waitKey(0)

print('-----------------')
print(count)