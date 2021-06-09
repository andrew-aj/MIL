import cv2
import numpy as np
from collections import deque
import imutils

lower = (15, 50, 100)
upper = (30, 255, 180)

vs = cv2.VideoCapture(0)
while True:
    no, frame = vs.read()

    clahe = cv2.createCLAHE(5., (4,4))
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l2 = clahe.apply(l)

    lab = cv2.merge((l2, a, b))
    frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    #frame = imutils.resize(frame, width=600)
    blurred = frame#cv2.GaussianBlur(frame, (5,5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    center = None

    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        ((x,y),radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        if radius > 10:
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0))
            cv2.circle(frame, center, 5, (255, 0, 0), -1)

    cv2.imshow("test",frame)
    cv2.imshow("t", mask)
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()


"""
cam = cv2.VideoCapture(0)
while True:
    ret_val, img = cam.read()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, (15, 50, 100), (30, 255, 180))
    im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
    

    center = [0, 0]

    if len(contours) != 0:
        c = max(contours, key = cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        center[0] = int(x+(w/2))
        center[1] = int(y+(h/2))

    cv2.circle(img, (center[0], center[1]), 20, (0, 0, 255), -1)


    cv2.imshow('webcam', mask)
    cv2.imshow('w', img)
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
"""