import cv2
import numpy as np

cam = cv2.VideoCapture(0)
while True:
    ret_val, img = cam.read()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, (10, 175, 80), (30, 255, 255))

    im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
    #print(len(contours))

    minRect = [None]*len(contours)
    for i,c in enumerate(contours):
        minRect[i] = cv2.minAreaRect(c)
        box = cv2.boxPoints(minRect[i])
        box = np.intp(box)
        cv2.drawContours(img, [box], 0, (0, 255,0))

    if len(contours) != 0:
        c = max(contours, key = cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

    #for contour in contours:
    #    (x,y,w,h) = cv2.boundingRect(contour)
    #    cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)

    cv2.imshow('webcam', mask)
    cv2.imshow('w', img)
    if cv2.waitKey(1) == 27:
        break
cv2.destroyAllWindows()