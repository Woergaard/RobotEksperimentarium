import time

import numpy as np
import cv2

import robot

LANDMARKS = [1,2,3,4] # SET ARUCO NUMBERS

POWER = 70

TURN_T = 8.3 # 1 degree
DRIVE_T = 22 # 1 centimeter

RIGHT_WHEEL_OFFSET = 5

CLOCKWISE_OFFSET = 0.8

FOCAL_LENGTH = 1691

CAMERA_MATRIX = np.array(
    [[FOCAL_LENGTH,0,512],[0,FOCAL_LENGTH,360],[0,0,1]],
    dtype=np.float32
)
DIST_COEF = np.array([0,0,0,0,0], dtype=np.float32)

def find_aruco(image):
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    aruco_params = cv2.aruco.DetectorParameters_create()
    corners, ids, _ = cv2.aruco.detectMarkers(
        image,
        aruco_dict,
        parameters=aruco_params
    )

    if corners is None:
        return []

    return {ids[i][0]: box[0] for i, box in enumerate(corners)}


def drunk_drive(drive_time, arlo, careful_range = 600):
    start = time.time()

    end = start + drive_time
    turning = None
    while time.time() < end:
        forward_dist = arlo.read_front_ping_sensor()
        right_dist = arlo.read_right_ping_sensor()
        left_dist = arlo.read_left_ping_sensor()
        if forward_dist > careful_range and all(x > 250 for x in [right_dist, left_dist]):
            turning = None
            arlo.go_diff(POWER, POWER + RIGHT_WHEEL_OFFSET, 1, 1)
        else:
            if turning == "R" or (forward_dist > careful_range and right_dist > 250 and turning is None):
                arlo.go_diff(POWER, POWER, 1, 0)
                turning = "R"
            else:
                arlo.go_diff(POWER, POWER + RIGHT_WHEEL_OFFSET, 0, 1)
                turning = "L"

        time.sleep(0.01)

def careful_forward(drive_time, arlo, careful_range = 600):
    start = time.time()
    if drive_time > 0.8:
        drive_time = 0.8
        hit_something = True
    else:
        hit_something = False

    end = start + drive_time
    turning = None
    while time.time() < end:
        forward_dist = arlo.read_front_ping_sensor()
        right_dist = arlo.read_right_ping_sensor()
        left_dist = arlo.read_left_ping_sensor()
        if forward_dist > careful_range and all(x > 250 for x in [right_dist, left_dist]):
            turning = None
            arlo.go_diff(POWER, POWER + RIGHT_WHEEL_OFFSET, 1, 1)
        else:
            hit_something = True
            if turning == "R" or (forward_dist > careful_range and right_dist > 250 and turning is None):
                arlo.go_diff(POWER, POWER, 1, 0)
                turning = "R"
            else:
                arlo.go_diff(POWER, POWER + RIGHT_WHEEL_OFFSET, 0, 1)
                turning = "L"

        time.sleep(0.01)

    return hit_something

def main():
    landmark_order = LANDMARKS + [
        LANDMARKS[0]
    ]

    noah = robot.Robot()

    while landmark_order != []:
        landmark = landmark_order.pop(0)
        print(f"looking for landmark {landmark}")

        while True:
            while True:
                for _ in range(24):
                    arucos = find_aruco(noah.take_photo())
                    if landmark in arucos:
                        break
                    noah.go_diff(POWER, POWER, 0, 1)
                    time.sleep((20 * TURN_T)/1000)
                    noah.stop()

                if landmark in arucos:
                    break

                drunk_drive(3, noah)


            position = cv2.aruco.estimatePoseSingleMarkers(
                np.array([arucos[landmark]]), 14.5, CAMERA_MATRIX, DIST_COEF
            )[1][0][0]

            angle = np.rad2deg(np.arctan(position[0]/position[2])) - 5
            drive_distance = np.sqrt(position[0]**2 + position[2]**2) - 100

            if angle < 0:
                noah.go_diff(POWER, POWER, 0, 1)
                time.sleep((abs(angle) * TURN_T)/1000)
                noah.stop()
            else:
                noah.go_diff(POWER, POWER, 1, 0)
                time.sleep((abs(angle) * TURN_T * CLOCKWISE_OFFSET)/1000)
                noah.stop()

            if not careful_forward((drive_distance * DRIVE_T)/1000, noah, 400):
                arucos = find_aruco(noah.take_photo())
                if landmark in arucos:
                    position = cv2.aruco.estimatePoseSingleMarkers(
                        np.array([arucos[landmark]]), 14.5, CAMERA_MATRIX, DIST_COEF
                    )[1][0][0]
                    drive_distance = np.sqrt(position[0]**2 + position[2]**2) - 25
                    noah.go_diff(POWER, POWER, 1, 1)
                    time.sleep((drive_distance * DRIVE_T)/1000)
                    noah.stop()

                    break


if __name__ == "__main__":
    start = time.time()
    main()
    print(f"Drove for {time.time() - start} seconds")