import numpy as np
import cv2
import matplotlib.pyplot as plt

def contourDetection(img):
    im = cv2.imread(img)
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    imgray = cv2.GaussianBlur(imgray,(3,3),cv2.BORDER_DEFAULT)
    m = np.mean(imgray)+110
    ret, thresh = cv2.threshold(imgray, m, 255, 0)
    contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hierarchy = hierarchy[0]
    return contours, hierarchy


def getCornersReloaded(_contour,_heirarchy):
    contr = []
    for i,h in enumerate(_heirarchy):
    	if h[2] != -1 and h[3] != -1:
    		contr.append((i))
    ctr = np.asarray(_contour)
    cnt = ctr[contr[0]]
    cnt = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
    hull = cv2.convexHull(cnt)
    return hull


def main():
    vname = 'VideoFrames/vid509.jpg'
    img = cv2.imread(vname)
    ctr,heir = contourDetection(vname)

    hullk = getCornersReloaded(ctr,heir)

    for c in hullk:
        x,y = c.ravel()
        cv2.circle(img,(x,y),5,255,2)
        print((x,y))
    plt.imshow(img)
    plt.show()
    
if __name__ == "__main__":
    main()