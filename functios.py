import cv2 as cv

# Contours w/ greatest number of points
# TODO max by area
def biggest_contour(contours):
    max_val = 0
    max_i = None
    for i in range(0, len(contours) - 1):
        if len(contours[i]) > max_val:
            cs = contours[i]
            max_val = len(contours[i])
            max_i = i
    return max_i

def max_contour(contours):
    max_val = 0
    max_cnt = None
    for contour in contours:
        if cv.contourArea(contour) > max_val:
            max_cnt = contour
            max_val = cv.contourArea(contour)

    return max_cnt

