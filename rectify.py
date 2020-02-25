import cv2 as cv2
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

def rectify(img_gray, corners):


    desired_corners = np.array([[0,0],[128, 0], [128, 128] ,[0, 128]], dtype=np.float32)
    H = find_svd(corners, desired_corners)
    shape = [128, 128]
    rect_img = getPerspectiveTransform(img_gray,H, shape)
    
    return rect_img

def warp_lena(img, img_lena, corners):
    desired_corners = np.array([[0,0],[img_lena.shape[0], 0], [img_lena.shape[0], img_lena.shape[0]] ,[0, img_lena.shape[0]]], dtype=np.float32)
    #print(corners)
    H = find_svd(corners, desired_corners)
    shape = [img_lena.shape[0],img_lena.shape[0]]
    warped_img = getPerspectiveTransform_Lena(img,img_lena, H, shape)
    return warped_img

def find_cube_pts(img, reqdPts, corners, shape, calib):
    desired_corners = np.array([[0, 0],[0, shape[1]],[shape[0], shape[1]], [shape[0], 0]])
    H = find_svd(corners, desired_corners)
    # print(H)
    H = np.linalg.inv(H)
    H = H/H[2,2]
    E = np.zeros([3, 4])
    calib_inv = np.linalg.inv(calib)
    E_ = np.matmul(calib_inv, H)
    lamda = (np.linalg.norm(np.matmul(calib_inv, H[:, 0])) + np.linalg.norm(np.matmul(calib_inv, H[:, 1])))/2
    B = np.linalg.det(E_)
    if B < 0:
        E_ = -E_
        
    E_ = E_/lamda
    E[:,0] = (E_[:,0]/lamda).T
    E[:,1] = (E_[:,1]/lamda).T
    E[:,2] = (np.cross(E[:,0], E[:,1])*lamda).T
    E[:,3] = (E_[:,2]/lamda).T
    E = E[:]/E[2,3]
    imgPts = np.matmul(calib,np.matmul(E,reqdPts.T))
    # print(E)
    # print(imgPts)
    # print(np.dot(E,reqdPts.T))
    # print(reqdPts.T)
    # exit(-1)
    return imgPts


def find_svd(c1,c2):
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
    return H

def getPerspectiveTransform(img, H, shape):
    Hinv = np.linalg.inv(H)
    Hinv = Hinv/Hinv[2,2]
    rect_img = np.zeros((shape[0], shape[1], 1))
    img_ = img.astype(np.float32)

    for i in range(shape[0]): # x? to change
        for j in range(shape[1]): #y?
            [x, y, z] = np.dot(Hinv, np.transpose([j, i, 1]))
            x = x/z
            y = y/z       
            if (x < 1920 and y < 1080 and x >= 0 and y >= 0):
                rect_img[i,j] = (img_[int(np.floor(y)),int(np.floor(x))] + img_[int(np.floor(y)),int(np.ceil(x))]
                                    + img_[int(np.ceil(y)), int(np.ceil(x))]+ img_[int(np.ceil(y)) , int(np.floor(x))])/4.0
            
    return rect_img

def getPerspectiveTransform_Lena(img, img_lena, H, shape):
    Hinv = np.linalg.inv(H)
    Hinv = Hinv/Hinv[2,2]
    img_ = np.zeros((img.shape[0], img.shape[1], 4))
    img_[:,:,0:3] = img
    
    for i in range(shape[0]): # x? to change
        for j in range(shape[1]): #y?
            [x, y, z] = np.dot(Hinv, np.transpose([j, i, 1]))
            x = x/z
            y = y/z
            #print(x, y)
            index_x = [int(np.floor(y)), int(np.floor(y)), int(np.ceil(y)), int(np.ceil(y))]
            index_y = [int(np.floor(x)), int(np.floor(x)), int(np.ceil(x)), int(np.ceil(x))]
            if(x < 1920 and y < 1080 and x>=0 and y>=0):
                img_[int(np.floor(y)), int(np.floor(x)), 0:3] = (img_[int(np.floor(y)), int(np.floor(x)), 0:3]*img_[int(np.floor(y)), int(np.floor(x)), 3] 
                                                                    + img_lena[i,j,0:3])/(img_[int(np.floor(y)), int(np.floor(x)), 3] + 1)
                img_[int(np.floor(y)), int(np.floor(x)), 3] += 1
                # #print(img_[int(np.floor(y)), int(np.floor(x)), 0:3]) 

                # img_[int(np.floor(y)), int(np.ceil(x)), 0:3] = (img_[int(np.floor(y)), int(np.ceil(x)), 0:3]*img_[int(np.floor(y)), int(np.ceil(x)), 3] 
                #                                                     + img_lena[i,j,0:3])/(img_[int(np.floor(y)), int(np.ceil(x)), 3] + 1)
                # img_[int(np.floor(y)), int(np.ceil(x)), 3] += 1
                # #print(img_[int(np.floor(y)), int(np.ceil(x)), 0:3])

                # img_[int(np.ceil(y)), int(np.ceil(x)), 0:3] = (img_[int(np.ceil(y)), int(np.ceil(x)), 0:3]*img_[int(np.ceil(y)), int(np.ceil(x)), 3] 
                #                                                     + img_lena[i,j,0:3])/(img_[int(np.ceil(y)), int(np.ceil(x)), 3] + 1)
                # img_[int(np.ceil(y)), int(np.ceil(x)), 3] += 1 
                # #print(img_[int(np.ceil(y)), int(np.ceil(x)), 0:3])

                # img_[int(np.ceil(y)), int(np.floor(x)), 0:3] = (img_[int(np.ceil(y)), int(np.floor(x)), 0:3]*img_[int(np.ceil(y)), int(np.floor(x)), 3] 
                #                                                     + img_lena[i,j,0:3])/(img_[int(np.ceil(y)), int(np.floor(x)), 3] + 1)
                # img_[int(np.ceil(y)), int(np.floor(x)), 3] += 1
                #print(img_[int(np.ceil(y)), int(np.floor(x)), 0:3])
                # img_[index_x, index_y, 0:3] = (img_[index_x, index_y, 0:3]*img_[index_x, index_y,3] + img_lena[i,j,0:3])/(img_[index_x, index_y,3] + 1)
                # img_[index_x, index_y, 3] += 1

    
    # inds = np.where(img_[:,:,3]==0)
    # img_[inds[0], inds[1], 3] = 1
    #img_ = img_[:,:,0:3]/img_[:,:,3, np.newaxis] 
    return img_[:,:,0:3].astype(np.uint8)                                                                              
        
def orient_img(img):
    scale = 5
    scaleEnd = -3
    num_rot = 0
    img_ = np.asarray(img[scale:scaleEnd, scale:scaleEnd]).astype(np.int32)
    inds = np.where(img_>= np.max(img_) - 15)
    xmin = np.min(inds[0])+scale
    xmax = np.max(inds[0])+scale
    ymin = np.min(inds[1])+scale
    ymax = np.max(inds[1])+scale
    topLeft = [xmin, ymin]
    topRight = [xmin, ymax]
    bottomLeft = [xmax, ymin]
    bottomRight = [xmax, ymax]
    keypts = np.array([np.add(topLeft,0.125*np.add(bottomRight,np.multiply(-1,topLeft))),np.add(topRight,0.125*np.add(bottomLeft,np.multiply(-1,topRight))),
                     np.add(bottomLeft, 0.125*np.add(topRight, np.multiply(-1,bottomLeft))), np.add(bottomRight, 0.125*np.add(topLeft, np.multiply(-1,bottomRight)))])
    if (img[topLeft[0]+8, topLeft[1]+8] >= 252):
        num_rot = 2
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif (img[topRight[0]+8, topRight[1]-8] >= 252):
        num_rot = 3
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif (img[bottomLeft[0]-8, bottomLeft[1]+8] >= 252):
        num_rot = 1
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return num_rot, img  
    

def find_id(img):
    sizex = img.shape[0]
    sizey = img.shape[1]
    scale = 5
    scaleEnd = -3
    img_ = np.asarray(img[scale:scaleEnd, scale:scaleEnd]).astype(np.int32)
    inds = np.where(img_>= np.max(img_) - 15)
    xmin = np.min(inds[0])+scale
    xmax = np.max(inds[0])+scale
    ymin = np.min(inds[1])+scale
    ymax = np.max(inds[1])+scale
    topLeft = [xmin, ymin]
    topRight = [xmin, ymax]
    bottomLeft = [xmax, ymin]
    bottomRight = [xmax, ymax]
    keypts = np.array([np.add(topLeft,0.45*np.add(bottomRight,np.multiply(-1,topLeft))),np.add(topRight,0.45*np.add(bottomLeft,np.multiply(-1,topRight))),
                     np.add(bottomLeft, 0.45*np.add(topRight, np.multiply(-1,bottomLeft))), np.add(bottomRight, 0.45*np.add(topLeft, np.multiply(-1,bottomRight)))])
    id = 0
    cv2.rectangle(img,(ymin,xmin),(ymax,xmax),(0,255,0),thickness=1)
    for i in range(len(keypts)):
        if(img[int(keypts[i][0])][int(keypts[i][1])] > 230):
            id = (id << 1) | int('00000001', 2)
        else:
            id = (id << 1) | int('00000000', 2)
    return id

def draw_cubes(img, corners, imgPts):
    
    for i, pt in enumerate(corners):        
        cv2.line(img, tuple(corners[i%4]),tuple(corners[(i+1)%4]),(0,255,255),3)
        cv2.line(img, tuple(imgPts[0:2, i%4].astype(np.int32)),tuple(imgPts[0:2, (i+1)%4].astype(np.int32)),(0,255,255),3)
        cv2.line(img, tuple(corners[i%4]),tuple([int(imgPts[0,i%4]),int(imgPts[1,i%4])]),(255,0,0),3)           

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
    