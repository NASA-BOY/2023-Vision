# -*- coding: utf-8 -*-
import os
import pathlib
from pathlib import Path
from networktables import NetworkTables

import cv2
import ovl

import custom_filters
import functios

# Constants
IMAGE_WIDTH = 320
IMAGE_HEIGHT = 240
target_area = 0
directions = (-2, -2)
game_piece = -1  # 0 for cone || 1 for cube
status = functios.ConeSkew.no_cone
ROOT_DIR: pathlib.PosixPath = pathlib.PosixPath("/home/pi/vision/")
TIPPED_CONES_PATH = ROOT_DIR / "tipped_cones/"
OK_CONES_PATH = ROOT_DIR / "ok_cones/"
STRAIGHT_CONES_PATH = ROOT_DIR / "straight_cones/"
CUBES_IMAGES_PATH = ROOT_DIR / "cubes/"

# Config the robot's network table
# robot = ovl.NetworkTablesConnection("10.19.37.1")
NetworkTables.initialize("10.19.37.2")
# Config the camera size
camera_config = ovl.CameraConfiguration({
    ovl.CameraProperties.IMAGE_WIDTH: IMAGE_WIDTH,
    ovl.CameraProperties.IMAGE_HEIGHT: IMAGE_HEIGHT
})

# Config the camera port, dimensions and exposure
camera = cv2.VideoCapture(-1)
# camera.configure_camera(camera_config)
# camera.get_image()

# Create all the cones images contours lists
tipped_cones = functios.create_cones_contours(TIPPED_CONES_PATH)
ok_cones = functios.create_cones_contours(OK_CONES_PATH)
straight_cones = functios.create_cones_contours(STRAIGHT_CONES_PATH)
cubes_images = functios.create_cubes_contours(CUBES_IMAGES_PATH)

# Config the yellow range for cone detection
yellow = ovl.Color([14, 120, 90], [33, 255, 255])
# ROBOT CAM yellow = ovl.Color([18, 140, 50], [36, 255, 255])

# Config the purple range for cube detection
purple = ovl.Color([110, 70, 45], [160, 255, 255])

# Set the ovl director
director = ovl.Director(directing_function=ovl.xy_normalized_directions, target_selector=1, failed_detection=(-2, -2))

# Cone filters and vision configurations
cone_filters = [ovl.percent_area_filter(minimal_percent=4),
                ovl.area_sort(),
                custom_filters.shape_filter(cones=[tipped_cones, ok_cones, straight_cones])]

# Apply the threshold(Color Mask), director and target filters on the camera feed using ovl vision
detect_cone = ovl.Vision(camera=camera,
                         threshold=yellow,
                         target_filters=cone_filters,
                         image_filters=[ovl.sharpen_image(), ovl.adaptive_brightness()],
                         # Sharpen really helps for some reason
                         morphological_functions=[custom_filters.close_open()],
                         director=director, )

# Cube filters and vision configurations
cube_percent_filter = [ovl.percent_area_filter(minimal_percent=40)]

cube_filters = [ovl.percent_area_filter(minimal_percent=15),
                ovl.area_sort(),
                custom_filters.shape_filter(cones=[cubes_images])]

detect_cube = ovl.Vision(camera=camera,
                         threshold=purple,
                         target_filters=cube_filters,
                         director=director)

detect_cube_percent = ovl.Vision(camera=camera,
                         threshold=purple,
                         target_filters=cube_percent_filter,
                         director=director)


while True:
    # Init the values for default of none (for the case nothing was detected)
    percent_area = 0  # percent area of the camera frame of the cone/cube

    # Detect any cube - no in the intake with a shape and up close with area percentage
    frame = detect_cube.get_image()
    cubes, frame = detect_cube.detect(frame)

    frame_percent = detect_cube_percent.get_image()
    percent_cubes, frame_percent = detect_cube_percent.detect(frame_percent)

    targets = cubes
    print("hey")
    print(len(percent_cubes))

    # Check if any cubes were detected
    if len(cubes) > 0:
        game_piece = 1  # 0 for cone || 1 for cube

    elif len(percent_cubes) > 0:
        game_piece = 1  # 0 for cone || 1 for cube
        targets = percent_cubes
        frame = frame_percent

    # If no cube was detected try to detect cones and determine their state
    else:
        # Get the frame from the camera and find the targets using the vision set above
        frame = detect_cone.get_image()

        targets, frame = detect_cone.detect(frame)

        # Keep only the biggest target and delete the rest (We only want to target the closest cargo)
        targets = targets[:1]
        status = functios.ConeSkew.no_cone
        for contour in targets:
            game_piece = 0  # 0 for cone || 1 for cube
            # Get the cone status - straight, tipped(L/R) and OK for intake
            status = functios.get_cone_state(contour, frame, straight_cones, tipped_cones, ok_cones)

    # Calculate the game piece percent area
    for contour in targets:
        image_size = IMAGE_WIDTH * IMAGE_HEIGHT
        target_area = cv2.contourArea(contour)
        percent_area = target_area / image_size * 100

    # Get the x and y values of the distance of the target from the center of the camera
    directions = detect_cone.get_directions(targets=targets, image=frame)

    x = directions[0]
    y = directions[1]

    # ======NETWORKTABLE=====
    table = NetworkTables.getTable("vision")

    # Send the game piece type detected
    table.putValue("game_piece", game_piece)  # 0 for cone || 1 for cube

    # Send the cone status as described in functions.get_cone_state
    table.putValue("cone_state", status)
    #
    # Send the x and y value of the detected cone
    # The value of x and y are so that you put the value directly to the motor speed (-1 to 1)
    table.putValue("target_x", x)
    table.putValue("target_y", y)
    #
    # Send the percent area of the cone
    table.putValue("target_area", percent_area)
