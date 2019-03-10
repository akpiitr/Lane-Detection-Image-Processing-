import cv2 as cv
import numpy as np

#vdosrc= "C:\\Users\\Abhishek\\Desktop\\ML\\data\\road_car_view.mp4"
video = cv.VideoCapture("C:\\Users\\Abhishek\\Desktop\\ML\\data\\road_car_view.mp4")

while True:
    ret, frame = video.read()
    if not ret:
        video = cv.VideoCapture("C:\\Users\\Abhishek\\Desktop\\ML\\data\\road_car_view.mp4")
        continue
    hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)
    #cv.imshow("HSV", hsv)
    hsv[0:500,0:1280] = [0,0,0]
    blurimg = cv.blur(hsv, (2, 2),0)

    low_yellow = np.array([18,94,140])
    high_yellow= np.array([48,255,255])

    mask = cv.inRange(blurimg,low_yellow,high_yellow)
    #cv.imshow("Mask", mask)
    edge = cv.Canny(mask,75,150)
    #cv.imshow("Edge", edge)
    #print(edge.shape)

    lines = cv.HoughLinesP(edge,1,np.pi/180,50,maxLineGap=50)
    if lines is not None:
       for line in lines:
            x1,y1,x2,y2 = line[0]
            cv.line(frame,(x1,y1),(x2,y2),(0,255,0),3)

    cv.imshow("Frame",frame)
    #cv.imshow("Cropped",cropped)
    key = cv.waitKey(1)
    if key == 27:
        break
video.release()
cv.destroyAllWindows()
