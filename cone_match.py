import cv2
import ovl
import functios
import numpy as np

# Constants
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
target_area = 0
BAD_CONES_PATH = "C:\\Users\\itayo\\Charged_Up_2023_Vision\\bad_cones"
OK_CONES_PATH = "C:\\Users\\itayo\\Charged_Up_2023_Vision\\ok_cones"

# Config the robot's network table
#robot = ovl.NetworkTablesConnection("10.19.37.1")

# Config the camera size
camera_config = ovl.CameraConfiguration({
    ovl.CameraProperties.IMAGE_WIDTH: IMAGE_WIDTH,
    ovl.CameraProperties.IMAGE_HEIGHT: IMAGE_HEIGHT,
})

# Config the camera port, dimensions and exposure
camera = ovl.Camera(1)
camera.configure_camera(camera_config)
#camera.set_exposure(-7)

# Create all the cones images contours lists
bad_cones = functios.create_cones_contours(BAD_CONES_PATH)
ok_cones = functios.create_cones_contours(OK_CONES_PATH)


# Config the yellow range for detection
yellow = ovl.Color([16, 120, 100], [36, 255, 255])

# Set the ovl director
director = ovl.Director(directing_function=ovl.xy_normalized_directions, target_selector=1, failed_detection=(-2, -2))


@ovl.predicate_target_filter
def shape_filter(contour):
    return functios.cone_shape_match(contour, bad_cones) or functios.cone_shape_match(contour, ok_cones)


def closing(frame):
    # kernel = np.ones((5, 5), np.uint8)
    # return cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)

    kernelSizes = [(3, 3), (5, 5), (7, 7)]
    # loop over the kernels sizes
    for kernelSize in kernelSizes:
        # construct a rectangular kernel from the current size and then
        # apply an "opening" operation
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
        frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)

    return frame


# Set the desired target filter
target_filters = [ovl.percent_area_filter(minimal_percent=1),
                  ovl.area_sort(),
                  shape_filter()]


# Apply the threshold(Color Mask), director and target filters on the camera feed using ovl vision
detect_cargo = ovl.Vision(threshold=yellow,
                          director=director,
                          target_filters=target_filters,
                          camera=camera)



while True:
    # Get the frame from the camera and find the targets using the vision set above
    frame = detect_cargo.get_image()

    targets, _ = detect_cargo.detect(frame)

    # Keep only the biggest target and delete the rest (We only want to target the closest cargo)
    targets = targets[:1]

    # Display the video with the target marked (This is temp and only for testings, should delete when finished)
    ovl.display_contours(frame, targets, color=(0, 255, 0), display_loop=True)

    # Calculate the contours area
    for contour in targets:
        target_area = cv2.contourArea(contour)
        print(target_area)

        status = functios.get_cone_state(contour, frame, bad_cones, ok_cones)

        cv2.putText(frame, status[1], (100, 100), fontFace=cv2.FONT_HERSHEY_SCRIPT_COMPLEX, fontScale=1,color=(0, 0, 255))

        # ret = functios.cone_shape_match(contour, bad_cones)
        # print("match: ", ret)
        # if ret:
        #     print("❤")

        # # Rectangle
        # rect = cv2.minAreaRect(contour)
        # box = cv2.boxPoints(rect)
        # box = np.int0(box)
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
        angle = functios.get_cone_angle(frame, contour)
        cv2.putText(frame, str(angle), (200, 100), fontFace=cv2.FONT_HERSHEY_SCRIPT_COMPLEX, fontScale=1, color=(0, 0, 255))

    # Get the x and y values of the distance of the target from the center of the camera
    directions = detect_cargo.get_directions(targets=targets, image=frame)

    x = directions[0]
    y = directions[1]

    # Prints
    print("Target directions", directions)

    # Send the x and y value and target area from above to the networktable under vision table
    # The value of x and y are so that you put the value directly to the motor speed (-1 to 1)
    #robot.send(x, "target_x")
    #robot.send(y, "target_y")
    #robot.send(target_area, "target_area")
