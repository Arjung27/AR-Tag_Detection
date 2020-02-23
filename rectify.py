import cv2 as cv2
import numpy as np
from scipy.spatial import Delaunay

def rectify(img, corners):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    desired_corners = np.array([[0,0],[128, 0], [128, 128] ,[0, 128]], dtype=np.float32)
    H = find_svd(corners, desired_corners)
    hull = Delaunay(corners) ## creates a delaunay diagram from the corners of artag. Required to find points inside the ar tag
    rect_img = np.zeros((17, 17))
    print(img_gray.shape)
    rect_img = cv2.warpPerspective(img_gray, H, (128, 128))
    # H = np.linalg.inv(H)
    # for i in range(img.shape[0]):
    #     for j in range(img.shape[1]):
    #         if (in_hull([i,j], hull)):
    #             x,y,_ = (np.matmul(H,np.array([i,j,1]).T)).T
    #             x,y = int(x), int(y)
    #             print(str(x) + ","+  str(y))
    #             rect_img[x, y] = img_gray[i, j]

    return rect_img

def find_svd(c1,c2):
    # c1 : artag corners
    # c2 : desired corners
    print(c1, c2)
    [x1,y1],[x2,y2],[x3,y3],[x4,y4] = c2
    [xp1, yp1], [xp2, yp2], [xp3, yp3], [xp4, yp4] = c1
    A = np.array([[-x1,-y1,-1,0,0,0,x1*xp1,y1*xp1,xp1],[0,0,0,-x1,-y1,-1,x1*yp1,y1*yp1,yp1],[-x2,-y2,-1,0,0,0,x2*xp2,y2*xp2,xp2],\
                [0,0,0,-x2,-y2,-1,x2*yp2,y2*yp2,yp2],[-x3,-y3,-1,0,0,0,x3*xp3,y3*xp3,xp3],[0,0,0,-x3,-y3,-1,x3*yp3,y3*yp3,yp3],\
                [-x4,-y4,-1,0,0,0,x4*xp4,y4*xp4,xp4],[0,0,0,-x4,-y4,-1,x4*yp4,y4*yp4,yp4]], dtype=np.float32)
                    
    A_trans = A.transpose()

    A_prod = np.dot(A_trans,A)
    # print(type(a))
    w,v = np.linalg.eig(A_prod)
    H = v[:,-1]
    H = np.reshape(H,(3,3))
    H = H/H[2,2]
    H = np.linalg.inv(H)
    H = H/H[2,2]
    print(H)
    H_ = cv2.getPerspectiveTransform(np.asarray(c1).astype(np.float32), c2.astype(np.float32))
    print(H_)
    # exit(-1)
    return H


def orient_img(img):
    sizex = img.shape[0] # pixel dimensions in x
    sizey = img.shape[1] # pixel dimensions in y
    keypts = np.array([[(2.5*sizex)/8, (2.5*sizey)/8], [(5.5*sizex)/8, (2.5*sizey)/8], [(5.5*sizex)/8, (5.5*sizey)/8], [(2.5*sizex)/8, (5.5*sizey)/8]]) # key points in the 5x5 grid to check for orientation
    # for i in range(len(keypts)):
    #     if(img[int(keypts[i][0])][int(keypts[i][1])] > 220 and img[int(keypts[i][0])+5][int(keypts[i][1])+5] > 220 and img[int(keypts[i][0])-5][int(keypts[i][1])-5] > 220): # check if particular grid point is white
    #         key_pt = i                              # store that particular keypoint index
    #         print (i)
    #         break
    print(img[0,0])
    inds = np.where(img >= 252)
    xmin = np.min(inds[0])
    xmax = np.max(inds[0])
    ymin = np.min(inds[1])
    ymax = np.max(inds[1])
    topLeft = [xmin, ymin]
    topRight = [xmin, ymax]
    bottomLeft = [xmax, ymin]
    bottomRight = [xmax, ymax]
    keypts = np.array([topLeft, topRight, bottomLeft, bottomRight])
    if (img[topLeft[0]+8, topLeft[1]+8] >= 252):
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif (img[topRight[0]+8, topRight[1]-8] >= 252):
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif (img[bottomLeft[0]-8, bottomLeft[1]+8] >= 252):
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # num_rot = {                                     # num of clockwise rotation based on which index is white in color. 
    #     0 : 2,
    #     1 : 1,
    #     2 : 0,
    #     3 : 3
    # }
    # for i in range(num_rot.get(key_pt)): 
    #     img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return img  
    

def find_id(img):
    sizex = img.shape[0]
    sizey = img.shape[1]
    inds = np.where(img >= 252)
    xmin = np.min(inds[0])
    xmax = np.max(inds[0])
    ymin = np.min(inds[1])
    ymax = np.max(inds[1])
    topLeft = [xmin, ymin]
    topRight = [xmin, ymax]
    bottomLeft = [xmax, ymin]
    bottomRight = [xmax, ymax]
    keypts = np.array([np.add(topLeft,0.375*np.add(bottomRight,np.multiply(-1,topLeft))),np.add(topRight,0.375*np.add(bottomLeft,np.multiply(-1,topRight))),
                     np.add(bottomLeft, 0.375*np.add(topRight, np.multiply(-1,bottomLeft))), np.add(bottomRight, 0.375*np.add(topLeft, np.multiply(-1,bottomRight)))])
    id = 0
    for i in range(len(keypts)):
        if(img[int(keypts[i][0])][int(keypts[i][1])] > 240):
            id = (id << 1) | int('00000001', 2)
        else:
            id = (id << 1) | int('00000000', 2)
    return id               

def in_hull(p, hull):
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)
    res = hull.find_simplex(p)>=0
    # print(res)
    return res
    
if __name__=="__main__":
    index = [44, 250, 399]
    corners = [
                np.array([[1145, 567], [1074, 598], [1033, 537], [1104, 508]], dtype=np.float32),
                np.array([[1099, 625], [1037, 642], [1004, 582], [1067, 566]], dtype=np.float32),
                np.array([[1158, 540], [1134, 597], [1057, 558], [1086, 498]], dtype=np.float32)
            ]

    for i in range(len(index)):
        img = cv2.imread("VideoFrames/vid"+ str(index[i])+".jpg")
        imgCorner = corners[i]
        rect_img = rectify(img, imgCorner)
        cv2.imshow("rectified image", rect_img)
        oriented_image = orient_img(rect_img)
        cv2.imshow("oriented image", oriented_image)
        id = find_id(oriented_image)
        print (id)
        cv2.waitKey(0)
    