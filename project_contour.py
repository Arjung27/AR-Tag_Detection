import numpy as np
import cv2
import matplotlib.pyplot as plt

def contourDetection(imgray):
    imgray = cv2.GaussianBlur(imgray,(3,3),cv2.BORDER_DEFAULT)
    m = np.mean(imgray)+110
    ret, thresh = cv2.threshold(imgray, m, 255, 0)
    _,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hierarchy = hierarchy[0]
    return contours, hierarchy


def getCornersReloaded(_contour,_heirarchy):
    contr = []
    for i,h in enumerate(_heirarchy):
    	if h[2] != -1 and h[3] != -1:
    		contr.append((i))
    #print(contr)

    cour = np.asarray(_contour)
    arr = []
    for i in contr:
        cont = cour[i]
        cont = cv2.approxPolyDP(cont,0.01*cv2.arcLength(cont,True),True)
        cont = cv2.convexHull(cont)
        arr.append((i,cv2.contourArea(cont)))
        #print(cv2.contourArea(cont))
    arr = np.asarray(arr)
    # print(arr)
    if len(arr)==0:
        return None
    
    arr = arr[arr[:,1].argsort()[::-1]]
    
    arr_ind = arr[:,0]

    connor = []
    for i in arr_ind:
        connor.append((i))
    connor = np.asarray(connor)
    # print(len(connor))
    if len(connor)>2:
        rng = 3
    else:
        rng = 1

    
    chull = []
    for i in range(0,rng):
        ctr = np.asarray(_contour)

        cnt = ctr[int(connor[i])]
        #print(contr[i])
        coeff = 0.01
        cnt = cv2.approxPolyDP(cnt,coeff*cv2.arcLength(cnt,True),True)
        hull = cv2.convexHull(cnt)
        
            
        thull = np.reshape(hull,(hull.shape[0],hull.shape[2]))
        # print(thull)
        ymin= np.where(thull[:,1]==np.amin(thull[:,1]))
        ymax= np.where(thull[:,1]==np.amax(thull[:,1]))
        ymin = ymin[0][0]
        ymax = ymax[0][0]
        xmin= np.where(thull[:,0]==np.amin(thull[:,0]))
        xmax= np.where(thull[:,0]==np.amax(thull[:,0]))
        xmin = xmin[0][0]
        xmax = xmax[0][0]
        temp = []
        temp.append(hull[ymin,:,:])
        temp.append(hull[xmax,:,:])
        temp.append(hull[ymax,:,:])
        temp.append(hull[xmin,:,:])
        temp = np.asarray(temp)
            #print(temp.shape)

        #if len(hull)==4:
        chull.append(temp)

    return chull


def main():

    vname = 'VideoFrames3/vid572.jpg'
    img = cv2.imread(vname)
    ctr,heir = contourDetection(vname)

    hullk = getCornersReloaded(ctr,heir)
    if len(hullk) == 0:
        print("No corner groups found")
    

    for i in range(len(hullk)):
        for c in hullk[i]:
            x,y = c.ravel()
            cv2.circle(img,(x,y),5,255,2)
            print((x,y))
        plt.imshow(img)
    plt.show()

    
if __name__ == "__main__":
    main()