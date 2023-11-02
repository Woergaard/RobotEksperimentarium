# Her kommer libraries 
import numpy as np
import robot
import _utils
import time
import random
import cv2
from time import sleep
from numpy import linalg


try:
    import picamera2
    print("Camera.py: Using picamera2 module")
except ImportError:
    print("Camera.py: picamera2 module not available")
    exit(-1)

print("OpenCV version = " + cv2.__version__)

# Her kommer specifikation af numeriske variable
arlo = robot.Robot()

rightWheelFactor = 1.0
leftWheelFactor = 1.06225
standardSpeed = 50.0

frontLimit = 400.0
sideLimit = 400.0

frontLimitCoastal = 700.0
sideLimitCoastal = 700.0

landmarkIDs = [1, 2, 3, 4, 1]
lostLimit = 1000.0

# Her kommer funktioner
# DONE 1. (turn_and_watch) Robotten roterer til den finder det rigtige landmark 
# 2. (lost) Hvis den ikke finder det rigtige landmark kør mod et landmark med id > 4
# 3. (drive_when_lost) Når den er på midten kør med sensor måling så afstanden altid er > 1000 mm
# DONE 4. (sensor measurement) Robotten tjekker afstand og vinkel til landmark 
# DONE 5. (drive_carefully_to_landmark) Robotten kører ca. 3/4 dele af afstanden til landmark og slår over til sensor måling men køre ligefrem. 
# DONE 6. (approach) Robotten når det rigtige landmark og stopper.
# DONE 7. Tilbage til 1. 


def costaldrive(goalID, image, arucoDict, frontLimitCoastal, sideLimitCoastal):
    '''
    Robotten kører langs kysten og leder efter et landmark med et bestemt id.
    Robotten holder sikkerhedsafstand baseret på ping væk fra landmarks med 
    id > 4. 
    '''

    if goalID == 3:
        # Robotten kører og holder øje med, om den støder ind i noget
        pingFront, pingLeft, pingRight, pingBack = _utils.sensor()

        # Robotten drejer og kører, indtil den har fundet et landmark, OG der er frit.
        landmarkFound = False
        while not landmarkFound:
            landmarkFound = turn_and_watch('left', image, [1], arucoDict)

        while pingFront > frontLimitCoastal and pingLeft > sideLimitCoastal-650 and pingRight > sideLimitCoastal:
            pingFront, pingLeft, pingRight, pingBack = _utils.sensor()
            arlo.go_diff(leftWheelFactor*standardSpeed, rightWheelFactor*standardSpeed, 1, 1)

        _utils.sharp_turn('left', 90)

        # tag højde for at der er tomt
        while pingFront > frontLimitCoastal and pingLeft > sideLimitCoastal and pingRight > sideLimitCoastal-650:
            pingFront, pingLeft, pingRight, pingBack = _utils.sensor()
            arlo.go_diff(leftWheelFactor*standardSpeed, rightWheelFactor*standardSpeed, 1, 1)
            

    if goalID == 1:
        # Robotten kører og holder øje med, om den støder ind i noget
        pingFront, pingLeft, pingRight, pingBack = _utils.sensor()

        # Robotten drejer og kører, indtil den har fundet et landmark, OG der er frit.
        landmarkFound = False
        while not landmarkFound:
            landmarkFound = turn_and_watch('left', image, [3], arucoDict)

        while pingFront > frontLimitCoastal and pingLeft > sideLimitCoastal and pingRight > sideLimitCoastal:
            pingFront, pingLeft, pingRight, pingBack = _utils.sensor()
            arlo.go_diff(leftWheelFactor*standardSpeed, rightWheelFactor*standardSpeed, 1, 1)

        _utils.sharp_turn('right', 90)

        while pingFront > frontLimitCoastal and pingLeft > sideLimitCoastal and pingRight > sideLimitCoastal:
            pingFront, pingLeft, pingRight, pingBack = _utils.sensor()
            arlo.go_diff(leftWheelFactor*standardSpeed, rightWheelFactor*standardSpeed, 1, 1)


    # Robotten drejer og kører, indtil den har fundet et landmark, OG der er frit.
    landmarkFound = False
    while not landmarkFound:
        landmarkFound = turn_and_watch('left', image, [goalID], arucoDict)
    
def detect_landmarks(img, arucoDict):
    aruco_corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(img, arucoDict)
    w, h = 1280, 720
    focal_length = 1744.36 
    camera_matrix = np.array([[focal_length, 0, w/2], [0, focal_length, h/2], [0, 0, 1]])
    arucoMarkerLength = 145.0
    distortion = 0
    
    # Giver distancen til boxen
    _, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(aruco_corners, arucoMarkerLength, camera_matrix, distortion)

    seenLandmarks = []
    # Udregner vinkel og retning til landmarket
    if ids is not None:
        for i in range(len(ids)):
            world_angle = np.arccos(np.dot(tvecs[i]/np.linalg.norm(tvecs[i]), np.array([0, 0, 1])))
            if np.dot(tvecs[i], np.array([1, 0, 0])) < 0:
                direction = 'left'
            else:
                direction = 'right'

            seenLandmarks.append(_utils.Landmark(linalg.norm(tvecs[i]), world_angle, direction, ids[i][0], tvecs[i][0]))
            #print('dist: ' + str(linalg.norm(tvecs[i])) + ', vinkel: ' + str(world_angle) + direction + ', id: ' + str(ids[i][0]) + ', tvec: ' + str(tvecs[i][0]))
        
        seenLandmarks.sort(key=lambda x: x.distance, reverse=False) # sorterer listen, så det nærmeste landmark er først.
    else:
        print("No marker detected")
    

    return seenLandmarks, ids, aruco_corners
    
def turn_and_watch(direction, img, landmarkIDs, arucoDict):
    '''
    Robotten drejer om egen akse, indtil den har fundet et landmark, OG der er frit.
    Argumenter:
        direction:  enten 'right' eller 'left'
    '''
    if direction == "right":
        r, l = 1, 0
    elif direction == "left":
        r, l = 0, 1
    else:
        print("Indtast enten 'left' eller 'right' som retning")
        return

    arlo.go_diff(leftWheelFactor*standardSpeed*0.6, rightWheelFactor*standardSpeed*0.6, r, l)

    seenLandmarks, ids, aruco_corners = detect_landmarks(img, arucoDict)
    
    landmark_spotted = False
    if ids is not None:
        if not landmarkIDs:
            for id in ids:
                print('hej obstacle')
                if id > 4:
                    landmark_spotted = True
        else: 
            for id in landmarkIDs:
                print('hej landmark')
                if id in ids:
                    landmark_spotted = True
        
    # If at least one marker is detected
    if len(aruco_corners) > 0 and landmark_spotted:
        for i in range(len(ids)):
            print('Landmark ' + str(ids[i]) + ' detekteret via turn_and_watch.')
        return True, seenLandmarks
    else: 
        arlo.go_diff(leftWheelFactor*standardSpeed, rightWheelFactor*standardSpeed, 0, 1)
        turnSeconds = _utils.degreeToSeconds(20)
        _utils.wait(turnSeconds)
        arlo.stop()
        _utils.wait(2.0)
        print('Intet landmark fundet.')

        return False, seenLandmarks 


def wait_and_sense(seconds):
    '''
    Funktionen venter i den tid, som robotten skal udføre en bestemt action.
    Samme tanke som Sleep, men kan måske sørge for den kan holde sig opdateret på sine omgivelser.
    Argumenter:
        start:    starttidspunkts
        seconds:  sekunder, der skal ventes
    '''
    start = time.perf_counter()
    isDriving = True
    pingFront, pingLeft, pingRight, pingBack = _utils.sensor()

    while (isDriving and pingFront > 350 and pingLeft > 250 and pingRight > 250):
        pingFront, pingLeft, pingRight, pingBack = _utils.sensor()
        sleep(0.041)
        if (time.perf_counter() - start > seconds):
            isDriving = False

def drive_carefully(direction, meters):
    '''
    Funktionen kører robotten et antal meter.
    Argumenter:
        direction:  enten 'forwards', 'backwards', 'left' eller 'right'
        meters:   meter, der skal køres
    '''
    if direction == 'forwards':
        arlo.go_diff(leftWheelFactor*standardSpeed, rightWheelFactor*standardSpeed, 1, 1)
    elif direction == 'backwards':
        arlo.go_diff(leftWheelFactor*standardSpeed, rightWheelFactor*standardSpeed, 0, 0)

    driveSeconds = _utils.metersToSeconds(meters)
    wait_and_sense(driveSeconds)

def approach(maxdist):
    print('Nærmer sig landmarket.')

    arlo.go_diff(leftWheelFactor*standardSpeed*0.6, rightWheelFactor*standardSpeed*0.6, 1, 1)
    pingFront, pingLeft, pingRight, pingBack = _utils.sensor()
    isDriving = True
    start = time.perf_counter()
    seconds = _utils.metersToSeconds(maxdist/1000) * 0.6

    while isDriving and (pingFront > 350 and pingLeft > 350 and pingRight > 350):
        pingFront, pingLeft, pingRight, pingBack = _utils.sensor()
        sleep(0.041)
        #if (time.perf_counter() - start > seconds):
        #    isDriving = False

    arlo.stop()

def drive_carefully_to_landmark(landmark, frontLimit, sideLimit): #Robotten kører ca. 3/4 dele af afstanden til landmark og slår over til sensor måling men køre ligefrem.
    direction = landmark.retning
    degrees = landmark.vinkel
    print('Drejer ' + str(degrees) + 'grader i retning ' + str(direction)) 

    # Robotten drejer
    _utils.sharp_turn(direction, degrees)
    
    # Robotten tjekker, om der er frit
    pingFront, pingLeft, pingRight, pingBack = _utils.sensor()
    if pingFront < frontLimit and pingLeft < sideLimit and pingRight < sideLimit:
        return False, 0

    distance = landmark.distance

    # Robotten kører, mens den holder øje, om den støder ind i noget
    if landmark.id:
        safetyDist = distance - 1000.0
    else:
        safetyDist = distance - 700.0

    dist = min([distance*(3.0/4.0), safetyDist])
    
    drive_carefully('forwards', dist/1000)

    # Tjekker, om der er blevet afbrudt grundet spærret vej
    pingFront, pingLeft, pingRight, pingBack = _utils.sensor()
    if (pingFront < frontLimit and pingLeft < sideLimit and pingRight < sideLimit):
        return False, 0

    maxdist = distance*(1.0/4.0)
    return True, maxdist

def camera_setup():
    '''
    Funktionen åbner kameraet og udfører en kommando.
    Argumenter:
        command:  kommando
        show:   bool, der angiver, om kameraets output skal vises
    '''
    # Open a camera device for capturing
    imageSize = (1280, 720)
    FPS = 60
    cam = picamera2.Picamera2()
    frame_duration_limit = int(1/FPS * 1000000) # Microseconds
    # Change configuration to set resolution, framerate
    picam2_config = cam.create_video_configuration({"size": imageSize, "format": 'RGB888'}, controls={"FrameDurationLimits": (frame_duration_limit, frame_duration_limit)}, queue=False)
    cam.configure(picam2_config) # Not really necessary
    cam.start(show_preview=False)

    #print('hej2')
    #print(cam.camera_configuration()) # Print the camera configuration in use

    time.sleep(1)  # _utils.wait for camera to setup

    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)

    return cam, arucoDict

def use_camera(cam, arucoDict, command, params, show):
    # Open a window'''
    if show:
        print('Kameraet vises.')
        WIN_RF = "Example 1"
        cv2.namedWindow(WIN_RF)
        cv2.moveWindow(WIN_RF, 100, 100)
    
    while cv2.waitKey(4) == -1: # _utils.wait for a key pressed event
        image = cam.capture_array("main")
        
        # Show frames
        if show:
            cv2.imshow(WIN_RF, image)

        elif command == 'turn_and_watch':
            return turn_and_watch('left', image, params[0], arucoDict)
        elif command == 'costaldrive':
            costaldrive(params[0], image, arucoDict, params[1], params[2])
            return

# Her kommer main programmet
def main(landmarkIDs, frontLimit, sideLimit, show):

    # vi åbner kameraet
    cam, arucoDict = camera_setup()

    for goalID in landmarkIDs:
        #if goalID == 2 or goalID == 4: 
            print('Søger efter landmark ' + str(goalID))
            landmarkFound = False
            iters = 0

            while not landmarkFound:
                landmarkSeen = False

                # vi drejer og kører, indtil vi har fundet det landmark, vi søger efter
                while not landmarkSeen:
                    if iters < 18:
                        print('Drejer og leder efter landmark ' + str(goalID))
                        landmarkSeen, seenLandmarks = use_camera(cam, arucoDict, 'turn_and_watch', [[goalID]], show)
                        
                        if landmarkSeen:
                            landmarkIndex = 0
                            for i in range(len(seenLandmarks)):
                                if seenLandmarks[i].id == goalID:
                                    landmarkIndex = i
                        
                            # Robotten kører og apporacher landmarket
                            landmarkFound, maxdist = drive_carefully_to_landmark(seenLandmarks[landmarkIndex], frontLimit, sideLimit)
                            
                            print('Sikrer os, at vi er nær landmarket.')
                            
                            approach(maxdist)

                        iters += 1
                    elif iters < 36:
                        print('Søger efter et landmark i midten med id > 4.')
                        landmarkSeen, seenLandmarks = use_camera(cam, arucoDict, 'turn_and_watch', [[]], show)

                        if landmarkSeen:
                            # finder landmark i liste
                            landmarkIndex = 0
                            for i in range(len(seenLandmarks)):
                                if seenLandmarks[i].id > 4:
                                    landmarkIndex = i
                                    
                            print('Kører mod ' + str(seenLandmarks[landmarkIndex].id))
                            landmarkFound, maxdist = drive_carefully_to_landmark(seenLandmarks[landmarkIndex], frontLimit, sideLimit)
                            approach(maxdist)

                            print('Kører langs kysten og leder efter ' + str(goalID))
                            use_camera(cam, arucoDict, 'costaldrive', [goalID, frontLimitCoastal, sideLimitCoastal], show)

                        iters += 1

                        #drive_free_carefully(2.0, frontLimit, sideLimit)
                        #(lost) Hvis den ikke finder det rigtige landmark kør mod et landmark med id > 4
                        #(drive_when_lost) Når den er på midten kør med sensor måling så afstanden altid er > 1000 mm
                    else: 
                        iters = 0

                arlo.stop()
                

            #if landmarkFound:
            #    print('Landmark ' + str(goalID) + ' er fundet! Tillykke!')
            #    arlo.stop()
               
#        else: 
#            print('Søger efter landmark ' + str(goalID))
#            landmarkFound = False
#            iters = 0

#            while not landmarkFound:
#                landmarkSeen = False

                # vi drejer og kører, indtil vi har fundet det landmark, vi søger efter
#                while not landmarkSeen:
#                    if iters < 18:
#                        print('Drejer og leder efter landmark ' + str(goalID))
#                        landmarkSeen, seenLandmarks = use_camera(cam, arucoDict, 'turn_and_watch', [[goalID]], show)
#                        iters += 1
#                    elif iters < 36:
#                        print('Søger efter et landmark i midten med id > 4.')
#                        landmarkSeen, seenLandmarks = use_camera(cam, arucoDict, 'turn_and_watch', [[]], show)

                        # finder landmark i liste
#                        landmarkIndex = 0
#                        for i in range(len(seenLandmarks)):
#                            if seenLandmarks[i].id > 4:
#                                landmarkIndex = i
                                
#                        print('Kører mod ' + str(seenLandmarks[landmarkIndex].id))
#                        _, _ = drive_carefully_to_landmark(seenLandmarks[landmarkIndex], frontLimit, sideLimit)

#                        print('Kører langs kysten og leder efter ' + str(goalID))
#                        use_camera(cam, arucoDict, 'costaldrive', [goalID, frontLimit, sideLimit], show)

#                        iters += 1
                        #drive_free_carefully(2.0, frontLimit, sideLimit)
                        #(lost) Hvis den ikke finder det rigtige landmark kør mod et landmark med id > 4
                        #(drive_when_lost) Når den er på midten kør med sensor måling så afstanden altid er > 1000 mm
#                    else:    
#                        iters = 0

#                arlo.stop()
                
#                landmarkIndex = 0
#                for i in range(len(seenLandmarks)):
#                    if seenLandmarks[i].id == goalID:
#                        landmarkIndex = i
            
                # Robotten kører og apporacher landmarket
#                landmarkFound, maxdist = drive_carefully_to_landmark(seenLandmarks[landmarkIndex], frontLimitCoastal, sideLimitCoastal)
                
#                print('Sikrer os, at vi er nær landmarket.')
                
                
#                approach(maxdist)

#            if landmarkFound:
#                print('Landmark ' + str(goalID) + ' er fundet! Tillykke!')
#                arlo.stop()
                        
    arlo.stop()
    print('Rute færdiggjort!')

main(landmarkIDs, frontLimit, sideLimit, False)












