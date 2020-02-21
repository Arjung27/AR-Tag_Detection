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
    
def contourDetection(img):
    im = cv2.imread(img)
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    m = np.mean(imgray)+110
    ret, thresh = cv2.threshold(imgray, m, 255, 0)
    contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    '''
    dst = cv2.drawContours(im, contours, -1, (0,255,0), 3)
    cv2.imshow('contour',dst)
    cv2.waitKey(0)
    '''
    return contours
    
def getCorners(cntr):
    for cnt in cntr:
        cnt = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
        hull = cv2.convexHull(cnt)
        if len(hull) == 4 and cv2.contourArea(hull)<8000:
            return hull
            
            
#cv2.contourArea(hull)<8000        
            
     
        
    
def main():
    
    #filename = 'Video_dataset/Tag0.mp4'
    vname = 'VideoFrames/vid44.jpg'
    #captureVideo(filename)
    img = cv2.imread(vname)
    ctr = contourDetection(vname)
    crnr = getCorners(ctr)
    print(crnr)
    #crnr = np.array(crnr[0])
    #crnr = np.reshape(crnr,(crnr.shape[0],crnr.shape[2]))
   
    for c in crnr:
        x,y = c.ravel()
        cv2.circle(img,(x,y),5,255,2)
    cv2.imshow('coroner',img)
    cv2.waitKey(0)    
    
    
if __name__ == "__main__":
    main()
    
    
