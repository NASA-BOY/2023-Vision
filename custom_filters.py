import cv2
import numpy as np
import ovl

import functios


@ovl.predicate_target_filter
def shape_filter(contour, cones):
    """
    :param contour: Contour to check if passes the filter
    :param cones: list of cones contours list
    :return: Whether the contour passes the filter
    """
    for cones_type in cones:
        if functios.cone_shape_match(contour, cones_type):
            return True
    return False

    # return functios.cone_shape_match(contour, tipped_cones) or functios.cone_shape_match(contour, ok_cones) \
    #        or functios.cone_shape_match(contour, straight_cones)

@ovl.image_filter
def close_open(thresh):
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

@ovl.image_filter
def dialation(thresh):
    # apply dilation on src image
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(thresh, kernel, iterations=2)