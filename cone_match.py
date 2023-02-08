import cv2
import ovl
import functios

# Constants
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
target_area = 0

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
straight_cones = functios.create_cones_contours()


# Config the yellow range for detection
yellow = ovl.Color([19, 90, 90], [35, 255, 255])

# Set the ovl director
director = ovl.Director(directing_function=ovl.xy_normalized_directions, target_selector=1, failed_detection=(-2, -2))


@ovl.predicate_target_filter
def shape_filter(contour):
    return functios.cone_shape_match(contour, straight_cones)


# Set the desired target filter
target_filters = [ovl.percent_area_filter(minimal_percent=1),
                  ovl.area_sort(),
                  shape_filter()]


# Apply the threshold(Color Mask), director and target filters on the camera feed using ovl vision
detect_cargo = ovl.Vision(threshold=yellow,
                          director=director,
                          target_filters=target_filters,
                          camera=camera)

# cone1 = cv.imread("C:/Users/itayo/Charged_Up_2023_Vision/cones/cone2.jpg")
# cv.imshow('OG', cone1)
# print(cone1)
#
# lower = np.array([18, 100, 40])
# higher = np.array([35, 255, 255])
#
# hsv = cv.cvtColor(cone1, cv.COLOR_BGR2HSV)
# h, w = cone1.shape[:2]
# mask = cv.inRange(hsv, lower, higher);
#
# contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
#
# # Only draw the biggest one
# # bc = functios.biggest_contour(contours)
#
# bc = functios.max_contour(contours)
# cv.drawContours(cone1, bc, -1, (0, 255, 0), 3)
#
# cv.imshow('hsv', hsv)
# cv.imshow('mask', mask)

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
        ret = functios.cone_shape_match(contour, straight_cones)
        print("match: ", ret)
        if ret:
            print("❤")

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