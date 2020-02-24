# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 21:21:12 2020

@author: abhishek
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from copy import deepcopy

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
    _,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    '''
    dst = cv2.drawContours(im, contours, -1, (0,255,0), 3)
    cv2.imshow('contour',dst)
    cv2.waitKey(0)
    '''
    return contours

def getCrs(cntr,_img):
    for cnt in cntr:
        per = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
    
def getCorners(cntr):
    hullist = []
    for cnt in cntr:
        cnt = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
        hull = cv2.convexHull(cnt)
        hullist.append(hull)
    return hullist

def getC(cntr,_img):
    contours = sorted(cntr, key = cv2.contourArea, reverse = True)
    c1 = contours[1]
    rect = cv2.minAreaRect(c1)
    box = cv2.boxPoints(rect)
    box = box.astype('int')
    img_copy = deepcopy(_img)
    #img_box = cv2.rectangle(img_copy, (x, y), (x+w, y+h), color = (255, 0, 0), thickness = 2)
    final = cv2.drawContours(img_copy, contours=[box], contourIdx = -1, 
                         color = (255, 0, 0), thickness = 2)
    plt.imshow(final)
    plt.show()

#def getCr(cntr,_img):

            
def getCornersAlt(cntr,_img):
    box = []
    for i in range(len(cntr)):
        hull = cv2.convexHull(cntr[i])
        box.append(hull)
    cv2.drawContours(_img, box[0], -1, (255, 0, 0), 2)
    # Display the final convex hull image
    cv2.imshow('ConvexHull', _img)
    cv2.waitKey(0)
    
def cornerDetectCustom(fname):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    corners = cv2.goodFeaturesToTrack(gray,25,0.01,10)
    corners = np.int0(corners)
    
    for i in corners:
        x,y = i.ravel()
        cv2.circle(img,(x,y),3,255,-1)
    
    cv2.imshow('coroner',img)
    cv2.waitKey(0)
    
    
            
            
#cv2.contourArea(hull)<8000        
            
     
        
    
def main():
    
    #filename = 'Video_dataset/Tag0.mp4'
    vname = 'VideoFrames/vid447.jpg'
    #captureVideo(filename)
    img = cv2.imread(vname)
    ctr = contourDetection(vname)
    #getCornersAlt(ctr,img)
    #getC(ctr,img)

    crnr = getCorners(ctr)
    print(len(crnr))
    crnr = np.array(crnr[1])
    #crnr = np.reshape(crnr,(crnr.shape[0],crnr.shape[2]))
    #cornerDetectCustom(vname)
       
    for c in crnr:
        x,y = c.ravel()
        cv2.circle(img,(x,y),5,255,2)
        print((x,y))
    cv2.imshow('coroner',img)
    cv2.waitKey(0)
        
    
    
    
if __name__ == "__main__":
    main()
    
    
