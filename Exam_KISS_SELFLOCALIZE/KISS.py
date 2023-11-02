# Her kommer libraries 
import numpy as np
import robot
import _utils
import time
import random
import cv2
from time import sleep
from numpy import linalg
import particle
import sys
import camera
import math


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

landmarks_dict = {
    1: (0.0, 0.0),
    2: (0.0, 3000.0),
    3: (4000.0, 0.0),
    4: (4000.0, 3000.0)
}
CRED, CGREEN, CBLUE, CCYAN, CYELLOW, CMAGENTA, CWHITE, CBLACK = _utils.bgr_colors()
landmark_colors = [CRED, CGREEN, CBLUE, CYELLOW]

# Her kommer funktioner
# DONE 1. (turn_and_watch) Robotten roterer til den finder det rigtige landmark 
# 2. (lost) Hvis den ikke finder det rigtige landmark kør mod et landmark med id > 4
# 3. (drive_when_lost) Når den er på midten kør med sensor måling så afstanden altid er > 1000 mm
# DONE 4. (sensor measurement) Robotten tjekker afstand og vinkel til landmark 
# DONE 5. (drive_carefully_to_landmark) Robotten kører ca. 3/4 dele af afstanden til landmark og slår over til sensor måling men køre ligefrem. 
# DONE 6. (approach) Robotten når det rigtige landmark og stopper.
# DONE 7. Tilbage til 1. 


onRobot = True

def isRunningOnArlo():
    """Return True if we are running on Arlo, otherwise False.
       You can use this flag to switch the code from running on you laptop to Arlo - you need to do the programming here!
    """
    return onRobot
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
  
def selflocalize(cam, showGUI, arucoDict, maxiters, landmarkIDs, landmarks_dict, landmark_colors, prior_position, particles, distance, angle):
    #return (750.0,0.0)
    est_pose = _utils.Node(prior_position.x/10, prior_position.z/10, None)
    est_pose.setTheta(prior_position.theta)
    print('self_localize søger efter landmarks.')
    
    try:
        # Initialize particles
        num_particles = 1000
        
        if not particles:
            particles = _utils.initialize_particles(num_particles)
        else:
            est_pose
            for p in particles:
                print('Partiklerne flyttes ')
                p.setX(distance + math.sin(angle) + particle.getX())
                p.setY(distance + math.sin(angle) + particle.getY())
                p.setTheta(particle.getTheta() + angle)
        
        est_pose = particle.estimate_pose(particles) # The estimate of the robots current pose

        for iters in range(maxiters):
            print(iters)
            # get the image
            img = cam.capture_array("main")

            # Detect landmarks
            seenLandmarks, ids, aruco_corners = detect_landmarks(img, arucoDict)

            dists = []
            objectIDs = []
            angles = []
            for landmark in seenLandmarks:
                dists.append(landmark.distance)
                objectIDs.append(landmark.id)
                angles.append(landmark.vinkel)
  
            sigma_theta = 0.57
            sigma_d = 5.0
            particle.add_uncertainty(particles, sigma_d, sigma_theta)

            if not isinstance(objectIDs, type(None)):
                print('selflocalize opdager landmarks: ' +  objectIDs)
                landmarks_lst = _utils.make_list_of_landmarks(objectIDs, dists, angles, landmarks_dict)
                
                # omregner til milimeter
                for landmark in landmarks_lst:
                    landmark.x /= 10
                    landmark.z /= 10
                    landmark.tvec[0] /= 10
                    landmark.tvec[1] /= 10

                _utils.update_weights(sigma_d, sigma_theta, landmarks_lst, particles)    
                    
                _utils.normalize_weights(particles)

                intervals = _utils.make_intervals(particles)

                particles = _utils.generate_new_particles(num_particles, particles, intervals)

                # Draw detected objects
                #cam.draw_aruco_objects(img)
            else:
                # No observation - reset weights to uniform distribution
                for p in particles:
                    p.setWeight(1.0/num_particles)
            #print('hej1.3')
            est_pose = particle.estimate_pose(particles)
            #print(est_pose)

            #if showGUI:
                #print('hej2')
                # Draw map
                #_utils.draw_world(est_pose, particles, world, landmarks_dict, landmarkIDs, landmark_colors)
                
                # Show frame
                #cv2.imshow(WIN_RF1, img)

                # Show world
                #cv2.imshow(WIN_World, world)

                #time.sleep(10)

    finally:
    
        # Make sure to clean up even if an exception occurred
        
        # Close all windows
        #cv2.destroyAllWindows()

        # Clean-up capture thread
        #cv2.destroyAllWindows()
        #cam.terminateCaptureThread()

        est_pose = _utils.Node(est_pose.x * 10.0, est_pose.y * 10.0, None) # VI GLEMMER AT EKSPORTERE VINKLEN!!!
        est_pose.setTheta(est_pose.theta)

        return est_pose, particles

def selflocalize_360_degrees(cam, show, arucoDict, params):  
    arlo_position, particles = selflocalize(cam, show, arucoDict, params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7])
    # drejer 360 grader om sig selv og selflocalizer
    for _ in range(18):
        arlo.go_diff(leftWheelFactor*standardSpeed, rightWheelFactor*standardSpeed, 0, 1)
        turnSeconds = _utils.degreeToSeconds(20)
        _utils.wait(turnSeconds)
        arlo.stop()
        _utils.wait(2.0)
        arlo_position, particles = selflocalize(cam, show, arucoDict, params[0], params[1], params[2], params[3], arlo_position, particles, 0.0, 20.0)
    return arlo_position, particles
        
def landmark_reached(reached_node, temp_goal):
    if _utils.dist(reached_node, temp_goal) < 350.0:
        return True
    else:
        return False

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

        while pingFront > frontLimitCoastal and pingLeft > sideLimitCoastal and pingRight > sideLimitCoastal:
            pingFront, pingLeft, pingRight, pingBack = _utils.sensor()
            arlo.go_diff(leftWheelFactor*standardSpeed, rightWheelFactor*standardSpeed, 1, 1)

        _utils.sharp_turn('left', 90)

        # tag højde for at der er tomt
        while pingFront > frontLimitCoastal and pingLeft > sideLimitCoastal and pingRight > sideLimitCoastal:
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
    
    print('Rotten kører ' +  str(dist/1000) + ' meter mod landmark ' + str(landmark.id))
    drive_carefully('forwards', dist/1000)

    # Tjekker, om der er blevet afbrudt grundet spærret vej
    pingFront, pingLeft, pingRight, pingBack = _utils.sensor()
    if (pingFront < frontLimit and pingLeft < sideLimit and pingRight < sideLimit):
        return False, 0

    #maxdist = distance*(1.0/4.0)
    return True, distance, degrees

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
        
        if command == 'selflocalize':
            arlo_position, particles = selflocalize(cam, show, arucoDict, params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7])
            return arlo_position, particles
        
        if command == 'selflocalize_360':
            arlo_position, particles = selflocalize_360_degrees(cam, show, arucoDict, params)
            return arlo_position, particles

        elif command == 'turn_and_watch':
            return turn_and_watch('left', image, params[0], arucoDict)
        
        elif command == 'costaldrive':
            costaldrive(params[0], image, arucoDict, params[1], params[2])
            return

def find_goal_in_seenLandmarks(seenLandmarks, goalID, type):
    landmarkIndex = 0
    for i in range(len(seenLandmarks)):
        if type == 'landmark' and seenLandmarks[i].id == goalID:
            landmarkIndex = i
        elif type == 'obstacle' and seenLandmarks[i].id > 4:
                landmarkIndex = i
    return seenLandmarks[landmarkIndex]

# Her kommer main programmet
def main(landmarkIDs, frontLimit, sideLimit, show):

    # vi åbner kameraet
    cam, arucoDict = camera_setup()

    # itererer efter landmarks i rækkefølge
    for goalID in landmarkIDs:
        print('Søger efter landmark ' + str(goalID))
        landmarkFound = False
        iters = 0
        particles = []
        arlo_position = _utils.Node(-500, -500, None)
        arlo_position, particles = use_camera(cam, arucoDict, 'selflocalize_360', [10, landmarkIDs, landmarks_dict, landmark_colors, arlo_position, particles, 0.0, 0.0], show)

        # søger efter landmarket, indtil det er fundet og berørt
        while not landmarkFound:
            landmarkSeen = False

            # vi drejer og kører, indtil vi har fundet det landmark, vi søger efter
            while not landmarkSeen:
                # vi leder efter et landmark 10 gange
                if iters < 18:
                    print('Drejer og leder efter landmark ' + str(goalID))
                    landmarkSeen, seenLandmarks = use_camera(cam, arucoDict, 'turn_and_watch', [[goalID]], show)
                    iters += 1
                # hvis vi ikke kan finde et landmark, håndter vi det således: 
                else:
                    print('Søger efter et landmark i midten med id > 4.')
                    landmarkSeen, seenLandmarks = use_camera(cam, arucoDict, 'turn_and_watch', [[]], show)

                    # finder landmark i liste
                    obstacleLandmark = find_goal_in_seenLandmarks(seenLandmarks, goalID, 'obstacle')
                            
                    print('Kører mod ' + str(obstacleLandmark.id))
                    _, _, _ = drive_carefully_to_landmark(obstacleLandmark, frontLimit, sideLimit)

                    print('Kører langs kysten og leder efter ' + str(goalID))
                    use_camera(cam, arucoDict, 'costaldrive', [goalID, frontLimitCoastal, sideLimitCoastal], show)
                    particles = []
                    distance = 0

                    #drive_free_carefully(2.0, frontLimit, sideLimit)
                    #(lost) Hvis den ikke finder det rigtige landmark kør mod et landmark med id > 4
                    #(drive_when_lost) Når den er på midten kør med sensor måling så afstanden altid er > 1000 mm
                    iters = 0

            arlo.stop()
            
            goalLandmark = find_goal_in_seenLandmarks(seenLandmarks, goalID, 'landmark')
        
            # Robotten kører hen mod landmarket
            landmarkFound, distance, angle = drive_carefully_to_landmark(goalLandmark, frontLimit, sideLimit)
            
            # Robotten approacher landmarket
            print('Approacher landmark.')
            approach(distance)

            
            print('Sikrer os, at vi er nær landmarket ved selflocalization.')
            print('Begynder selflokalisering.')
            arlo_position, particles = use_camera(cam, arucoDict, 'selflocalize_360', [10, landmarkIDs, landmarks_dict, landmark_colors, arlo_position, particles, distance, angle], show)
            print('Arlo befinder sig på position: ', math.floor(arlo_position.x), math.floor(arlo_position.z), math.floor(arlo_position.theta))
            goalNode = _utils.Node(goalLandmark.x, goalLandmark.z, None)
            landmarkFound = landmark_reached(arlo_position, goalNode)
                        
    arlo.stop()
    print('Rute færdiggjort!')

main(landmarkIDs, frontLimit, sideLimit, False)












