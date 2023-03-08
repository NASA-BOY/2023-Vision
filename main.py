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
ROOT_DIR: pathlib.PosixPath = pathlib.PosixPath("/home/pi/vision/")
TIPPED_CONES_PATH = ROOT_DIR.joinpath("../../Downloads/kevin_vision/tipped_cones")
print(ROOT_DIR)
OK_CONES_PATH = ROOT_DIR / "ok_cones/"
STRAIGHT_CONES_PATH = ROOT_DIR / "straight_cones/"

# Config the robot's network table
# robot = ovl.NetworkTablesConnection("10.19.37.1")
NetworkTables.initialize("10.19.37.1")
# Config the camera size
camera_config = ovl.CameraConfiguration({
    ovl.CameraProperties.IMAGE_WIDTH: IMAGE_WIDTH,
    ovl.CameraProperties.IMAGE_HEIGHT: IMAGE_HEIGHT
})

# Config the camera port, dimensions and exposure
camera = cv2.VideoCapture(0)
# camera.configure_camera(camera_config)
# camera.get_image()

# Create all the cones images contours lists
print(f"Tipped cones path:", TIPPED_CONES_PATH)
tipped_cones = functios.create_cones_contours(TIPPED_CONES_PATH)
ok_cones = functios.create_cones_contours(OK_CONES_PATH)
straight_cones = functios.create_cones_contours(STRAIGHT_CONES_PATH)

# Config the yellow range for cone detection
yellow = ovl.Color([14, 120, 90], [33, 255, 255])
# ROBOT CAM yellow = ovl.Color([18, 140, 50], [36, 255, 255])

# Config the purple range for cube detection
purple = ovl.Color([120, 90, 65], [150, 255, 255])

# Set the ovl director
director = ovl.Director(directing_function=ovl.xy_normalized_directions, target_selector=1, failed_detection=(-2, -2))

# Cone filters and vision configurations
cone_filters = [ovl.percent_area_filter(minimal_percent=1),
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
cube_filters = [ovl.percent_area_filter(minimal_percent=40)]

detect_cube = ovl.Vision(camera=camera,
                         threshold=purple,
                         target_filters=cube_filters,
                         director=director)

while True:
    frame = detect_cube.get_image()
    # frame = cv2.rotate(frame, cv2.ROTATE_180) NOT NECESSARY

    cubes, _ = detect_cube.detect(frame)
    if len(cubes) > 0:
        pass
        # robot.set(1, "game_piece")  # 0 for cone || 1 for cube

    else:
        # Get the frame from the camera and find the targets using the vision set above
        frame = detect_cone.get_image()

        targets, frame = detect_cone.detect(frame)

        # Keep only the biggest target and delete the rest (We only want to target the closest cargo)
        targets = targets[:1]
        status = (-1, "none")
        for contour in targets:
            # Calculate the cone contour percent area
            image_size = IMAGE_WIDTH * IMAGE_HEIGHT
            target_area = cv2.contourArea(contour)
            percent_area = target_area / image_size * 100  # #TODO: check it

            # Get the cone status - straight, tipped(L/R) and OK for intake
            status = functios.get_cone_state(contour, frame, straight_cones, tipped_cones, ok_cones)

        # Get the x and y values of the distance of the target from the center of the camera
        directions = detect_cone.get_directions(targets=targets, image=frame)

        x = directions[0]
        y = directions[1]
        # print("Table", robot.get_table(robot.table_name))
        # Send 1 as for cone detected
        table = NetworkTables.getTable("vision")  # 0 for cone || 1 for cube

        if (cone_status := status[0]):
            # Send the cone status as described in functions.get_cone_state
            table.putValue("status", status[0])
        print("Current cone status", status[1])
        #
        # # Send the x and y value of the detected cone
        # # The value of x and y are so that you put the value directly to the motor speed (-1 to 1)
        # robot.set(x, "target_x")
        # robot.set(y, "target_y")
        #
        # # Send the percent area of the cone
        # robot.set(percent_area, "target_area")