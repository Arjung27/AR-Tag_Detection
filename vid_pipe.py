import numpy as np
import cv2
import matplotlib.pyplot as plt
from copy import deepcopy
from rectify import *
import glob
from project_contour import *
np.random.seed(0)

if __name__=="__main__":
    path = "/home/vishnuu/UMD/ENPM673/Perception_Projects/Project1/AR-Tag-Detection-and-Tracking/Video_dataset/"
    filename = "Tag2.mp4" # Tag1.mp4, Tag2.mp4, Tag3.mp4    
    cap = cv2.VideoCapture(path+filename)
    vidWriter = cv2.VideoWriter(path+"output_"+filename, cv2.VideoWriter_fourcc(*'mp4v'), 24, (1920,1080))
    i = 0
    print("Running....")
    while(cap.isOpened()):
        print(i)
        i+=1
        ret, img = cap.read()
        if ret == False:
            break
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ctr, heirarchy = contourDetection(img_gray)
        crnr = getCornersReloaded(ctr, heirarchy)
        crnr = np.asarray(crnr)
        # print (crnr.shape)``
        if (crnr.shape == ()):  
            continue
        num_tags = int(crnr.shape[0]/4)
        crnr = crnr[0:num_tags*4]
        tagCrnrs = np.zeros([num_tags,4,1,2])
        if crnr.shape[0]%4 != 0:
            continue
        for j in range(0,len(crnr), 4):
            tagCrnrs[int(j/4),:,:,:] = crnr[j:(j+4)]
        tagCrnrs = np.squeeze(tagCrnrs, axis = 2)
        tagCrnrs = tagCrnrs.astype(np.int32)
        for j in range(0,tagCrnrs.shape[0]):
            rect_img = rectify(img_gray,tagCrnrs[j])
            num_rot, rect_img = orient_img(rect_img)
            tagCorner = (int(tagCrnrs[j,0,0]), int(tagCrnrs[j,0,1]))
            id = find_id(rect_img)
            rect_img = rect_img.astype(np.uint8)
            # cv2.imshow("rect", rect_img)
            cv2.putText(img, "TagID: "+str(id), tagCorner, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255))    
        vidWriter.write(img)
print("Completed")        
cap.release()
vidWriter.release()