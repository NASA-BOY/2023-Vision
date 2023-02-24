import cv2
import ovl
import functios
import numpy as np

# Constants
IMAGE_WIDTH = 320
IMAGE_HEIGHT = 240
target_area = 0
TIPPED_CONES_PATH = "C:\\Users\\itayo\\Charged_Up_2023_Vision\\tipped_cones"
OK_CONES_PATH = "C:\\Users\\itayo\\Charged_Up_2023_Vision\\ok_cones"
STRAIGHT_CONES_PATH = "C:\\Users\\itayo\\Charged_Up_2023_Vision\\straight_cones"

# Config the robot's network table
#robot = ovl.NetworkTablesConnection("10.19.37.1")

# Config the camera size
camera_config = ovl.CameraConfiguration({
    ovl.CameraProperties.IMAGE_WIDTH: IMAGE_WIDTH,
    ovl.CameraProperties.IMAGE_HEIGHT: IMAGE_HEIGHT
})

# Config the camera port, dimensions and exposure
camera = ovl.Camera(1)
camera.configure_camera(camera_config)
#camera.set_exposure(-7)
# camera.set(cv2.CAP_PROP_SHARPNESS, 0.2)

# Create all the cones images contours lists
tipped_cones = functios.create_cones_contours(TIPPED_CONES_PATH)
ok_cones = functios.create_cones_contours(OK_CONES_PATH)
straight_cones = functios.create_cones_contours(STRAIGHT_CONES_PATH)


# Config the yellow range for cone detection
yellow = ovl.Color([11, 100, 70], [33, 255, 255])
# ROBOT CAM yellow = ovl.Color([18, 140, 50], [36, 255, 255])


# Set the ovl director
director = ovl.Director(directing_function=ovl.xy_normalized_directions, target_selector=1, failed_detection=(-2, -2))

# Set the Morphological functions
# morph = [ovl.morphological_functions.erosion(), ovl.morphological_functions.dilation(iterations=1), ovl.morphological_functions.erosion(iterations=1)]


@ovl.predicate_target_filter
def shape_filter(contour):
    return functios.cone_shape_match(contour, tipped_cones) or functios.cone_shape_match(contour, ok_cones) \
           or functios.cone_shape_match(contour, straight_cones)

@ovl.image_filter
def close_open(thresh):
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

def dialation(thresh):
    # apply dilation on src image
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(thresh, kernel, iterations=2)


# Set the desired target filter
target_filters = [ovl.percent_area_filter(minimal_percent=1),
                  ovl.area_sort(),
                  shape_filter()]


# Apply the threshold(Color Mask), director and target filters on the camera feed using ovl vision
detect_cone = ovl.Vision(camera=camera,
                         threshold=yellow,
                         target_filters=target_filters,
                         # image_filters=[ovl.gaussian_blur()],
                         morphological_functions=[close_open()],
                         director=director, )

while True:
    # Get the frame from the camera and find the targets using the vision set above
    frame = detect_cone.get_image()

    #frame = cv2.rotate(frame, cv2.ROTATE_180)

    targets, frame = detect_cone.detect(frame)

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

    # Keep only the biggest target and delete the rest (We only want to target the closest cargo)
    targets = targets[:1]

    # Display the video with the target marked (This is temp and only for testings, should delete when finished)
    ovl.display_contours(frame, targets, color=(0, 255, 0), display_loop=True)

    cX = None
    cY = None
    bX = None
    bY = None

    # Calculate the contours area
    for contour in targets:

        image_size = IMAGE_WIDTH * IMAGE_HEIGHT
        target_area = cv2.contourArea(contour)
        percent_area = target_area / image_size * 100  # #TODO: check it

        print(target_area)

        # status = functios.get_cone_state(contour, frame, bad_cones, ok_cones)
        status = functios.get_cone_state(contour, frame, straight_cones, tipped_cones, ok_cones)

        cv2.putText(frame, status[1], (100, 20), fontFace=cv2.FONT_HERSHEY_SCRIPT_COMPLEX, fontScale=1, color=(255, 0, 0))

        # Rectangle
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        M = cv2.moments(box)
        bX = int(M["m10"] / M["m00"])
        bY = int(M["m01"] / M["m00"])

        print("💥POINTS💥 ", status)

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

        # compute the center of the contour
        M = cv2.moments(contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])


        # Calculate cone angle
        # angle = functios.get_cone_angle(frame, contour)
        # cv2.putText(frame, str(angle), (200, 100), fontFace=cv2.FONT_HERSHEY_SCRIPT_COMPLEX, fontScale=1, color=(0, 0, 255))

    cv2.circle(frame, (cX, cY), 7, (0, 0, 255), -1)
    cv2.circle(frame, (bX, bY), 7, (255, 0, 0), -1)

    # Get the x and y values of the distance of the target from the center of the camera
    directions = detect_cone.get_directions(targets=targets, image=frame)

    x = directions[0]
    y = directions[1]

    # Prints
    print("Target directions", directions)

    # Send the x and y value and target area from above to the networktable under vision table
    # The value of x and y are so that you put the value directly to the motor speed (-1 to 1)
    #robot.send(x, "target_x")
    #robot.send(y, "target_y")
    #robot.send(target_area, "target_area") #TODO change to area percentage
