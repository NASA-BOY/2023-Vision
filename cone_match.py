import os

import cv2
import ovl

import custom_filters
import functios
import numpy as np

# Constants
IMAGE_WIDTH = 320
IMAGE_HEIGHT = 240
target_area = 0
TIPPED_CONES_PATH = os.path.join(os.getcwd(), "tipped_cones")
OK_CONES_PATH = os.path.join(os.getcwd(), "ok_cones")
STRAIGHT_CONES_PATH = os.path.join(os.getcwd(), "straight_cones")

# Config the robot's network table
#robot = ovl.NetworkTablesConnection("10.19.37.1")

# Config the camera size
camera_config = ovl.CameraConfiguration({
    ovl.CameraProperties.IMAGE_WIDTH: IMAGE_WIDTH,
    ovl.CameraProperties.IMAGE_HEIGHT: IMAGE_HEIGHT
})

# Config the camera port, dimensions and exposure
camera = ovl.Camera(0)
camera.configure_camera(camera_config)
#camera.set_exposure(-7)


# Create all the cones images contours lists
tipped_cones = functios.create_cones_contours(TIPPED_CONES_PATH)
ok_cones = functios.create_cones_contours(OK_CONES_PATH)
straight_cones = functios.create_cones_contours(STRAIGHT_CONES_PATH)


# Config the yellow range for cone detection
yellow = ovl.Color([14, 120, 90], [33, 255, 255])
# ROBOT CAM yellow = ovl.Color([18, 140, 50], [36, 255, 255])

# Config the purple range for cube detection
purple = ovl.Color([110, 90, 65], [150, 255, 255])

# Set the ovl director
director = ovl.Director(directing_function=ovl.xy_normalized_directions, target_selector=1, failed_detection=(-2, -2))

# Set the Morphological functions
# morph = [ovl.morphological_functions.erosion(), ovl.morphological_functions.dilation(iterations=1), ovl.morphological_functions.erosion(iterations=1)]


# Cone filters and vision configurations
cone_filters = [ovl.percent_area_filter(minimal_percent=1),
                ovl.area_sort(),
                custom_filters.shape_filter(cones=[tipped_cones, ok_cones, straight_cones])]

# Apply the threshold(Color Mask), director and target filters on the camera feed using ovl vision
detect_cone = ovl.Vision(camera=camera,
                         threshold=yellow,
                         target_filters=cone_filters,
                         image_filters=[ovl.sharpen_image(), ovl.adaptive_brightness()],  # Sharpen really helps for some reason
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
    #frame = cv2.rotate(frame, cv2.ROTATE_180) NOT NECESSARY

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, purple.low, purple.high)
    cv2.imshow("PURP", mask)

    cubes, _ = detect_cube.detect(frame)
    if len(cubes) > 0:
        # robot.send(1, "game_piece")  # 0 for cone || 1 for cube
        print("CUBE")

    else:
        # Get the frame from the camera and find the targets using the vision set above
        frame = detect_cone.get_image()

        targets, frame = detect_cone.detect(frame)

        # Keep only the biggest target and delete the rest (We only want to target the closest cargo)
        targets = targets[:1]

        # ---TEMP Only for testing---
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, yellow.low, yellow.high)
        kernel = np.ones((5, 5), np.uint8)
        # new_mask = cv2.erode(mask, kernel, iterations=1)
        # new_mask = cv2.dilate(new_mask, kernel, iterations=1)
        new_mask = mask
        new_mask = cv2.morphologyEx(new_mask, cv2.MORPH_CLOSE, kernel)
        new_mask = cv2.morphologyEx(new_mask, cv2.MORPH_OPEN, kernel)

        cv2.imshow("new", new_mask)
        cv2.imshow("mask", mask)

        # Display the video with the target marked (This is temp and only for testings, should delete when finished)
        ovl.display_contours(frame, targets, color=(0, 255, 0), display_loop=True)  # Only for testing

        cX = None
        cY = None
        bX = None
        bY = None

        # ----TEMP End----

        for contour in targets:

            # Calculate the cone contour percent area
            image_size = IMAGE_WIDTH * IMAGE_HEIGHT
            target_area = cv2.contourArea(contour)
            percent_area = target_area / image_size * 100  # #TODO: check it
            print(percent_area)  # Only for testing

            # Get the cone status - straight, tipped(L/R) and OK for intake
            status = functios.get_cone_state(contour, frame, straight_cones, tipped_cones, ok_cones)

            # ---TEMP Only for testing---
            cv2.putText(frame, status[1], (100, 20), fontFace=cv2.FONT_HERSHEY_SCRIPT_COMPLEX, fontScale=1, color=(255, 0, 0))  # Only for testing

            # Rectangle
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            M = cv2.moments(box)
            bX = int(M["m10"] / M["m00"])
            bY = int(M["m01"] / M["m00"])

            # compute the center of the contour
            M = cv2.moments(contour)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            print("💥POINTS💥 ", status)
            # ----TEMP End----

            # cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)

            # triangle = cv2.minEnclosingTriangle(contour)
            # int_triangle = triangle.astype(int)

            # cv2.line(frame, int_triangle[0][0], int_triangle[0][1], (255, 0, 0), 3)
            # cv2.line(frame, triangle[1], triangle[2], (255, 0, 0), 3)
            # cv2.line(frame, triangle[0], triangle[2], (255, 0, 0), 3)
            #print(triangle)
            # print(len(triangle))
            #print(triangle[0], triangle[1], triangle[2])

            # cv2.drawContours(frame, triangle, 0, (0, 0, 255), 2)


            # Calculate cone angle
            # angle = functios.get_cone_angle(frame, contour)
            # cv2.putText(frame, str(angle), (200, 100), fontFace=cv2.FONT_HERSHEY_SCRIPT_COMPLEX, fontScale=1, color=(0, 0, 255))

        cv2.circle(frame, (cX, cY), 7, (0, 0, 255), -1)  # Only for testing
        cv2.circle(frame, (bX, bY), 7, (255, 0, 0), -1)  # Only for testing


        # Get the x and y values of the distance of the target from the center of the camera
        directions = detect_cone.get_directions(targets=targets, image=frame)

        x = directions[0]
        y = directions[1]

        # Prints
        print("Target directions", directions)

        # Send 1 as for cone detected
        #robot.send(0, "game_piece") # 0 for cone || 1 for cube

        # Send the cone status as described in functions.get_cone_state
        #robot.send(status[0], "status")

        # Send the x and y value of the detected cone
        # The value of x and y are so that you put the value directly to the motor speed (-1 to 1)
        #robot.send(x, "target_x")
        #robot.send(y, "target_y")

        # Send the percent area of the cone
        #robot.send(target_area, "target_area") #TODO change to area percentage
