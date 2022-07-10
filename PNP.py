import cv2 as cv
import os
import numpy as np

cam = cv.VideoCapture(0)

mtx = np.float32([[1.14997850e+03, 0.00000000e+00, 6.75181985e+02],
 [0.00000000e+00, 1.14883014e+03, 3.58476818e+02],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

dist = np.float32([[ 1.65045482e-01, -1.09605037e+00,  1.39121516e-03,  6.43303657e-03,
   1.97742182e+00]])


def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((9*6,3), np.float32)
objp[:,:2] = np.mgrid[0:6,0:9].T.reshape(-1,2) * 12 / 21
axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
print(objp)

cv.namedWindow('img', cv.WINDOW_NORMAL)

while True:
    _, img = cam.read()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (6,9),
                                             cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE)
    if ret:
        corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        # Find the rotation and translation vectors.
        ret, rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)
        print(rvecs)
        # project 3D points to image plane
        imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)
        img = draw(img,corners2,imgpts)
        np_rodrigues = np.asarray(rvecs[:, :], np.float64)
        rot_mat = cv.Rodrigues(np_rodrigues)[0]
        tvecs2 = -np.mat(rot_mat).transpose() * np.mat(tvecs)
        cv.putText(img,str(tvecs2), (50,50), cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 2, cv.LINE_AA)
        cv.imshow('img',img)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cv.destroyAllWindows()