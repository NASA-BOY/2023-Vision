import math
import os

import cv2 as cv
import numpy as np

# Contours w/ greatest number of points
# TODO max by area

def max_contour(contours):
    max_val = 0
    max_cnt = None
    for contour in contours:
        if cv.contourArea(contour) > max_val:
            max_cnt = contour
            max_val = cv.contourArea(contour)

    return max_cnt


def create_cones_contours(path):
    """
    TODO: Add so it works with straight or tipped cone (with a parameter of a folder path)
    :return:
    """

    lower = np.array([15, 100, 65])
    higher = np.array([35, 255, 255])
    cones = []

    img_count = len([entry for entry in os.listdir(path) if os.path.isfile(os.path.join(path, entry))])
    for i in range(1, img_count+1):
        cone_path = path + "\\cone" + str(i) + ".jpg"
        cone = cv.imread(cone_path)

        hsv = cv.cvtColor(cone, cv.COLOR_BGR2HSV)
        h, w = cone.shape[:2]
        mask = cv.inRange(hsv, lower, higher)
        contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        # Only get the biggest one
        cnt = max_contour(contours)

        cones.append(cnt)

    return cones


def cone_shape_match(contour, cones_contours):
    """

    :param contour:
    :param cones_contours:
    :return: The match number of the given contour and straight standing cone
    """

    for cnt in cones_contours:
        ret = cv.matchShapes(contour, cnt, 1, 0.0)
        if ret < 0.1:
            print("matchhh: ", ret)
            return True

    # cv.drawContours(cone1, bc, -1, (0, 255, 0), 3)
    # cv.imshow('hsv', hsv)
    # cv.imshow('mask', mask)

    return False


def get_cone_angle(frame, contour):
    """

    :param frame: The frame to find angle in
    :param contour: The contour to calculate its angle
    :return: The angle between the cone and the camera horizontal axis
    """

    rows, cols = frame.shape[:2]
    [vx, vy, x, y] = cv.fitLine(contour, cv.DIST_L2, 0, 0.01, 0.01)
    lefty = int((-x * vy / vx) + y)
    righty = int(((cols - x) * vy / vx) + y)
    #cv.line(frame, (cols - 1, righty), (0, lefty), (255, 0, 0), 2)

    m = (lefty - righty) / ((cols-1) - 0)
    angle = math.degrees(math.atan(m - 0))
    return int(angle)


def get_cone_state(contour, frame, bad_cones, ok_cones):
    """
    :param contour: The cone contour to analyse its state
    :param frame: The frame of which the cone contour is at
    :param bad_cones: Bad (not base facing or straight) cones images contours
    :param ok_cones: Ok(base facing the camera) cones images contours
    :return:0 - ok cone
            1 - straight standing cone
            2 - tipped right cone
            3 - tipped left cone
            4 - tip facing the camera
    """

    if cone_shape_match(contour, ok_cones):
        return 0, "OK"

    elif cone_shape_match(contour, bad_cones):
        angle = get_cone_angle(frame, contour)

        if 70 < angle <= 90 or -90 <= angle < -70:
            return 1, "straight"

        elif -20 < angle < 0:
            return 2, "right"

        elif 0 <= angle < 20:
            return 3, "left"

    return -1, "none"


def cone_state_half(contour):
    """
    :param contour:
    :return:
    """

    rect = cv.minAreaRect(contour)
    box = cv.boxPoints(rect)
    box = np.int0(box)

    M = cv.moments(box)
    bX = int(M["m10"] / M["m00"])


    right_points = 0
    left_points = 0

    for i in range(len(contour)):
        if contour[i][0][0] > bX:
            right_points += 1
        else:
            left_points += 1


    right_contour = np.zeros((right_points, 1, 2), dtype=np.int32)
    left_contour = np.zeros((left_points, 1, 2), dtype=np.int32)
    rcount = 0
    lcount = 0

    for i in range(len(contour)):
        if contour[i][0][0] > bX:
            right_contour[rcount][0] = contour[i][0]
            rcount += 1
        else:
            left_contour[lcount][0] = contour[i][0]
            lcount += 1

    if left_contour == 0 or right_contour == 0:
        return None, None

    ratio = cv.contourArea(left_contour) / cv.contourArea(right_contour)

    if ratio > 1.2:
        return 2, "right"

    elif ratio < 0.8:
        return 3, "left"

    else:
        return 1, "straight"
