import numpy as np
import cv2
import matplotlib.pyplot as plt
from copy import deepcopy
from rectify import *
import glob
from project_contour import *

if __name__=="__main__":
    path = "/home/vishnuu/UMD/ENPM673/Perception_Projects/Project1/AR-Tag-Detection-and-Tracking/Video_dataset/"
    filename = "multipleTags.mp4" # Tag1.mp4, Tag2.mp4, Tag3.mp4    
    cap = cv2.VideoCapture(path+filename)
    vidWriter = cv2.VideoWriter(path+"output_"+filename, cv2.VideoWriter_fourcc(*'mp4v'), 24, (1920,1080))
    i = 0
    print("Running....")
    while(cap.isOpened()):
        # print(i)
        i+=1
        # if i>20:
        #     break
        ret, img = cap.read()
        if ret == False:
            break
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ctr, heirarchy = contourDetection(img_gray)
        crnr = getCornersReloaded(ctr, heirarchy)
        crnr = np.asarray(crnr)
        if (crnr.shape == ()):  
            continue
        crnr = np.squeeze(crnr, axis = 2)
        crnr = crnr.astype(np.int32)
        for j in range(0,crnr.shape[0]):
            rect_img = rectify(img_gray,crnr[j])
            num_rot, rect_img = orient_img(rect_img)
            tagCorner = (int(crnr[j,0,0]), int(crnr[j,0,1]))
            id = find_id(rect_img)
            rect_img = rect_img.astype(np.uint8)
            # cv2.imshow("rect", rect_img)
            cv2.putText(img, "TagID: "+str(id), tagCorner, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255))    
        vidWriter.write(img)
print("Completed")        
cap.release()
vidWriter.release()
