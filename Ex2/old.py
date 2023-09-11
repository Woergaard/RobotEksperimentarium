# Move forward
#arlo.go_diff(_utils.leftWheelFactor*50, _utils.rightWheelFactor*50, 1, 1)

def sensorReadings():
        # request to read Front sonar ping sensor
        print("Front sensor = ", arlo.read_front_ping_sensor())
        sleep(0.041)

        # request to read Back sonar ping sensor
        print("Back sensor = ", arlo.read_back_ping_sensor())
        sleep(0.041)

        # request to read Right sonar ping sensorutils

        print("Right sensor = ", arlo.read_right_ping_sensor())
        sleep(0.041)

        # request to read Left sonar ping sensor
        print("Left sensor = ", arlo.read_left_ping_sensor())
        sleep(0.041)
