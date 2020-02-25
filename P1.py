import numpy as np
import cv2
import matplotlib.pyplot as plt

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

def contourDetection(imgray):
    imgray = cv2.GaussianBlur(imgray,(5,5),cv2.BORDER_DEFAULT)
    m = np.mean(imgray)+110
    ret, thresh = cv2.threshold(imgray, m, 255, 0)
    _,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    '''
    dst = cv2.drawContours(im, contours, -1, (0,255,0), 3)
    cv2.imshow('contour',dst)
    cv2.waitKey(0)
    '''
    return contours

def getCorners(cntr):
    hullist = []
    for cnt in cntr:
        cnt = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
        hull = cv2.convexHull(cnt)
	if cv2.contourArea(hull)>800:
        	hullist.append(hull)
    return hullist

def main():
    

    vname = 'VideoFrames/vid150.jpg'

    img = cv2.imread(vname)
    ctr = contourDetection(vname)

    crnr = getCorners(ctr)
    print(len(crnr))

    crnr = np.array(crnr[len(crnr)-2])

       
    for c in crnr:
        x,y = c.ravel()
        cv2.circle(img,(x,y),5,255,2)
        print((x,y))
    cv2.imshow('coroner',img)
    cv2.waitKey(0)
    
if __name__ == "__main__":
    main()
