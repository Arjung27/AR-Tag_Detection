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
    vidWriter = cv2.VideoWriter(path+"CubeProjection_"+filename, cv2.VideoWriter_fourcc(*'mp4v'), 24, (1920,1080))
    i = 0
    print("Running...")
    while(cap.isOpened()):
        # print(i)
        i+=1
        ret, img = cap.read()
        if ret == False:
            break
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ctr, heirarchy = contourDetection(img_gray)
        crnr = getCornersReloaded(ctr, heirarchy)
        crnr = np.asarray(crnr)
        if (crnr.shape == ()):  
            continue
        num_tags = int(crnr.shape[0])
        crnr = np.squeeze(crnr, axis = 2)
        crnr = crnr.astype(np.int32)
        for j in range(0,crnr.shape[0]): 
            rect_img = rectify(img_gray,crnr[j])
            num_rot, rect_img = orient_img(rect_img)
            refArtag = cv2.imread("./reference_images/ref_marker.png")
            refImgShape = [512, 512]
            cubeHeight = 512
            calib = np.array( [ [1406.08415449821, 0, 0], [2.20679787308599, 1417.99930662800,  0], [1014.13643417416, 566.347754321696,  1] ])
            calib = np.transpose(calib)
            reqdPts = np.array([[0, 0, -cubeHeight, 1], [0, refImgShape[1], -cubeHeight, 1],
                                [refImgShape[0], refImgShape[1], -cubeHeight, 1], [refImgShape[0], 0  , -cubeHeight, 1]])
            imgPts = find_cube_pts(img, reqdPts, crnr[j], refImgShape, calib)
            imgPts = imgPts[:]/imgPts[2,:]
            for c in imgPts.T:
                x, y, _ = c.ravel()
                cv2.circle(img,(int(x), int(y)),10,(0,255,0)) 
            draw_cubes(img, crnr[j], imgPts)
        vidWriter.write(img)
print("Ended...")
cap.release()
vidWriter.release()
