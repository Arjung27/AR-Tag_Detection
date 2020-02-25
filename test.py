import cv2
 
# Opens the Video file
cap= cv2.VideoCapture('/home/vishnuu/UMD/ENPM673/Perception_Projects/Project1/AR-Tag-Detection-and-Tracking/Video_dataset/Tag0.mp4')
vid_create = cv2.VideoWriter('/home/vishnuu/UMD/ENPM673/Perception_Projects/Project1/AR-Tag-Detection-and-Tracking/Tag0.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 24, (1920,1080))
i=0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    vid_create.write(frame)
    i+=1
    print (i)
 
cap.release()
cv2.ReleaseVideoWriter(vid_create)
cv2.destroyAllWindows()