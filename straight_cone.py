import cv2
import ovl

# Constants
IMAGE_WIDTH = 320
IMAGE_HEIGHT = 240
target_area = 0

# Config the robot's network table
#robot = ovl.NetworkTablesConnection("10.19.37.1")

# Config the camera size
camera_config = ovl.CameraConfiguration({
    ovl.CameraProperties.IMAGE_WIDTH: IMAGE_WIDTH,
    ovl.CameraProperties.IMAGE_HEIGHT: IMAGE_HEIGHT,
})

# Config the camera port, dimensions and exposure
camera = ovl.Camera(0)
camera.configure_camera(camera_config)
camera.set_exposure(-4)


# Config the yellow range for detection
yellow = ovl.Color([18, 100, 40], [35, 255, 255])

# Set the ovl director
director = ovl.Director(directing_function=ovl.xy_normalized_directions, target_selector=1, failed_detection=(-2, -2))

# Set the desired target filter
target_filters = [ovl.percent_area_filter(minimal_percent=1),
                  ovl.triangle_filter(approximation_coefficient=0.12),
                  ovl.area_sort()]

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