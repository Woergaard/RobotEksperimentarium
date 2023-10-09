
###################################
######### ÆLDRE VERSIONER #########
###################################

### MANUELLE SENSORISKE KØRSLER ###
def drive_old():
    pingFront, pingLeft, pingRight, pingBack = sensor()
    arlo.go_diff(leftWheelFactor*50, rightWheelFactor*50, 1, 1)
    dirLst = ['right', 'left']

    while (pingFront > 350 and pingLeft > 250 and pingRight > 250):
        pingFront, pingLeft, pingRight, pingBack = sensor()
        print(pingFront, pingLeft, pingRight, pingBack)
        sleep(0.041)
    
    # three sensors detected
    if (pingFront < 350 and pingLeft < 250 and pingRight < 250):
        sharp_turn('left', 180.0)  

    # two sensors detected
    elif (pingFront < 350 and pingLeft < 250):
        sharp_turn('right', 90.0)
    elif (pingFront < 350 and pingRight < 250):
        sharp_turn('left', 90.0)
    elif (pingLeft < 250 and pingRight < 250):
        sharp_turn('left', 180.0)

    # one sensors deteced
    elif(pingFront <= 350):
        randomDirection = random.choice(dirLst)
        sharp_turn(randomDirection, 90.0)
    elif (pingLeft < 250): 
        sharp_turn('right', 45.0)
    elif (pingRight < 250):
        sharp_turn('left', 45.0)
    
        
    drive_old()



### GAMLE VERSIONER ###

def pose_estimation(img, arucoDict): 
    '''
    Funktionen genkender QR-koder via kameraet og returnerer en liste, sorteret efter, hvilke, der er nærmest,
    der indeholder placering, id, retning og vinkel på landmarksene.
    Argumenter:
        img:  billedet fra kameraet
        arucoDict:   en dictonary af QR koder.
    '''

    aruco_corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(img, arucoDict)
    w, h = 1280, 720
    focal_length = 1744.36 
    camera_matrix = np.array([[focal_length, 0, w/2], [0, focal_length, h/2], [0, 0, 1]])
    arucoMarkerLength = 145.0
    distortion = 0

    lst = [] # liste af landmarks
    
    # Giver distancen til boxen
    _, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(aruco_corners, arucoMarkerLength, camera_matrix, distortion)
    
    # Udregner vinkel og retning til landmarket
    for i in range(len(ids)):
        world_angle = np.arccos(np.dot(tvecs[i]/np.linalg.norm(tvecs[i]), np.array([0, 0, 1])))
        print(world_angle)
        if np.dot(tvecs[i], np.array([1, 0, 0])) < 0:
            direction = 'left'
        else:
            direction = 'right'
        print(direction)

        lst.append([linalg.norm(tvecs[i]), world_angle, direction, ids[i][0], tvecs[i]])
    
    lst.sort() # Sorterer listen efter distance
    
    namelst = ['distance', 'vinkel', 'retning', 'id', 'tvec']    

    # printer alt, den har genkendt
    for element in lst:
        for i in range(len(element)):
            print(namelst[i] + '=' + str(element[i]) + '\n') 

    return lst

def camera():
    # Open a camera device for capturing
    imageSize = (1280, 720)
    FPS = 60
    cam = picamera2.Picamera2()
    frame_duration_limit = int(1/FPS * 1000000) # Microseconds
    # Change configuration to set resolution, framerate
    picam2_config = cam.create_video_configuration({"size": imageSize, "format": 'RGB888'},
                                                                controls={"FrameDurationLimits": (frame_duration_limit, frame_duration_limit)},
                                                                queue=False)
    cam.configure(picam2_config) # Not really necessary
    cam.start(show_preview=False)

    print(cam.camera_configuration()) # Print the camera configuration in use

    time.sleep(1)  # wait for camera to setup

    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    # Open a window
    WIN_RF = "Example 1"
    cv2.namedWindow(WIN_RF)
    cv2.moveWindow(WIN_RF, 100, 100)
    
    while cv2.waitKey(4) == -1: # Wait for a key pressed event
        image = cam.capture_array("main")
        
        # Show frames
        cv2.imshow(WIN_RF, image)
        #landmark_drive('left', image, arucoDict)
        if turn_and_watch('left', image) == True: 
            arlo.stop()
            landmarks_lst = pose_estimation(image, arucoDict)
            drive_to_landmarks(landmarks_lst)
            #arlo.stop()
            time.sleep(15)
