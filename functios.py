import math
from enum import Enum
from pathlib import Path

import cv2 as cv
import numpy as np
import ovl


class ConeSkew(int, Enum):
    no_cone = -1
    base_to_camera = 0
    standing = 1
    base_right = 2
    base_left = 3
    tip_to_camera = 4


def max_contour(contours):
    max_val = 0
    max_cnt = None
    for contour in contours:
        if cv.contourArea(contour) > max_val:
            max_cnt = contour
            max_val = cv.contourArea(contour)

    return max_cnt


def create_cones_contours(path: Path):
    """
    :param path: Path of the cone photos folder
    :return: A list of the given folder path cone contours
    """

    lower = np.array([15, 100, 65])
    higher = np.array([35, 255, 255])
    cones = []

    for image in path.glob("*.jpg"):
        cone = cv.imread(str(image.resolve()))

        hsv = cv.cvtColor(cone, cv.COLOR_BGR2HSV)
        h, w = cone.shape[:2]
        mask = cv.inRange(hsv, lower, higher)
        contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        # Only get the biggest one
        largest_contour = max_contour(contours)
        cones.append(largest_contour)

    return cones


def cone_shape_match(contour, cones_contours, match=0.1):
    """
    :param match: The match compare number
    :param contour: The contour to apply shape match on
    :param cones_contours: List of cones contours list
    :return: The match number of the given contour and straight standing cone
    """
    for cnt in cones_contours:
        ret = cv.matchShapes(contour, cnt, 1, 0.0)
        if ret < match:
            return True

    return False


def get_cone_angle(frame, contour):
    """
    :param frame: The frame to find angle in
    :param contour: The cone contour to calculate its angle
    :return: The angle between the cone and the camera horizontal axis
    """
    rows, cols = frame.shape[:2]
    [vx, vy, x, y] = cv.fitLine(contour, cv.DIST_L2, 0, 0.01, 0.01)
    lefty = int((-x * vy / vx) + y)
    righty = int(((cols - x) * vy / vx) + y)

    m = (lefty - righty) / ((cols - 1) - 0)
    angle = math.degrees(math.atan(m - 0))
    return int(angle)


def get_cone_state(contour, frame, straight_cones, tipped_cones, ok_cones):
    """
    :param contour: The cone contour to analyse its state
    :param frame: The frame of which the cone contour is at
    :param straight_cones: straight cones image contours list
    :param tipped_cones: tipped cones images contours list
    :param ok_cones: OK for intake cones images contours (base facing the camera)
    :return:-1 - none detected
            0 - OK cone
            1 - STRAIGHT standing cone
            2 - cone base RIGHT
            3 - cone base LEFT
            4 - tip facing the camera
            BEWARE: the side of the tipped cone changes with rotated camera as on the robot
    """

    if cone_shape_match(contour, ok_cones, 0.01):
        return ConeSkew.base_to_camera

    angle = get_cone_angle(frame, contour)

    if cone_shape_match(contour, straight_cones):
        if 70 < angle <= 90 or -90 < angle < -70:
            return ConeSkew.standing
        return tipped_cone_side(contour)

    if cone_shape_match(contour, tipped_cones, 0.1):
        return tipped_cone_side(contour)

    return ConeSkew.no_cone


def tipped_cone_side(contour):
    """
    :param contour: The cone contour to analyse its status
    :return: The given cone status
    """
    rect = cv.minAreaRect(contour)
    box = cv.boxPoints(rect)
    box = np.int0(box)
    cX = int(ovl.contour_center(box)[0])

    right_points = 0
    left_points = 0

    for i in range(len(contour)):
        if contour[i][0][0] > cX:
            right_points += 1
        else:
            left_points += 1

    if left_points == 0 or right_points == 0:
        return None, None

    right_contour = np.zeros((right_points, 1, 2), dtype=np.int32)
    left_contour = np.zeros((left_points, 1, 2), dtype=np.int32)
    rcount = 0
    lcount = 0

    for i in range(len(contour)):
        if contour[i][0][0] > cX:
            right_contour[rcount][0] = contour[i][0]
            rcount += 1
        else:
            left_contour[lcount][0] = contour[i][0]
            lcount += 1

    if cv.contourArea(right_contour) == 0:
        return ConeSkew.base_right

    ratio = cv.contourArea(left_contour) / cv.contourArea(right_contour)

    if ratio > 1.1:
        return ConeSkew.base_right

    elif ratio < 0.9:
        return ConeSkew.base_left

    else:
        return ConeSkew.base_to_camera
