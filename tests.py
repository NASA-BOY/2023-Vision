import cv2 as cv
import numpy as np
import functios


lower = np.array([13, 110, 85])
higher = np.array([35, 255, 255])
cones = []
path = "C:\\Users\\itayo\\2023-Vision\\tipped_cones\\cone84.jpg"
cone = cv.imread(path)


hsv = cv.cvtColor(cone, cv.COLOR_BGR2HSV)
h, w = cone.shape[:2]
mask = cv.inRange(hsv, lower, higher)
contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

# Only get the biggest one
cnt = functios.max_contour(contours)


cv.drawContours(cone, cnt, -1, (0, 255, 0), 3)
cv.imshow('hsv', hsv)
cv.imshow('mask', mask)
cv.imshow('cone', cone)

cv.waitKey(0)
cv.destroyAllWindows()
