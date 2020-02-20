# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 21:21:12 2020

@author: abhishek
"""

import numpy as np
import cv2

def captureVideo(fname):

    cap = cv2.VideoCapture(fname)
    i=0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        cv2.imwrite('VideoFrames/vid'+str(i)+'.jpg',frame)
        i+=1
    cap.release()
    cv2.destroyAllWindows()
    
def tagDetection(img):
    im = cv2.imread(img)
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    m = np.mean(imgray)+100
    ret, thresh = cv2.threshold(imgray, m, 255, 0)
    contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #print(contours)
    dst = cv2.drawContours(im, contours, -1, (0,255,0), 3)
    cv2.imshow('contour',dst)
    cv2.waitKey(0)
    
def main():
    
    #filename = 'Video_dataset/Tag0.mp4'
    vname = 'VideoFrames/vid50.jpg'
    #captureVideo(filename)
    tagDetection(vname)
    
if __name__ == "__main__":
    main()
    
    
