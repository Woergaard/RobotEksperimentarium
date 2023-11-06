import cv2
#import camera
import numpy as np
import time
from timeit import default_timer as timer
import sys
import _utils
import robot
import particle
from time import sleep
import random
import camera

from time import sleep
import robot
import time
import random
import cv2 # OpenCV library
import time
from pprint import *
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
import math
import particle
from timeit import default_timer as timer
import random


try:
    import picamera2
    print("Camera.py: Using picamera2 module")
except ImportError:
    print("Camera.py: picamera2 module not available")
    exit(-1)

print("OpenCV version = " + cv2.__version__)


### DEFINEREDE PARAMETRE ###
arlo = robot.Robot()

rightWheelFactor = 1.0
leftWheelFactor = 1.06225
standardSpeed = 50.0

def selflocalize(cam, showGUI, maxiters, landmarkIDs, landmarks_dict, landmark_colors, prior_position, particles, path, stepLength):
    #return (750.0,0.0)
    est_pose = _utils.Node(prior_position.x/10, prior_position.z/10, None)
    est_pose.setTheta(prior_position.theta)
    print('self_localize søger efter landmarks.')
    
    try:
        # Initialize particles
        num_particles = 1000
        
        if not particles:
            print('Initialiserer nye partikler')
            particles = _utils.initialize_particles(num_particles)

        if path:
            print('Flytter partiklerne.')
            prevnode = est_pose
            for node in path:
                direction, degrees = _utils.find_turn_angle(prevnode, node)
                print(degrees)
                delta_x = abs(prevnode.x - node.x/10)
                delta_z = abs(prevnode.z - node.z/10)
                print(delta_x)
                print(delta_z)
                for p in particles:
                    print('Partikel flyttes: ', delta_x, delta_z, degrees)
                    particle.move_particle(p, delta_x, delta_z, degrees)
        else:
            print('Partikler flyttes ikke, da der ikke er foretaget ændringer i position.')
        
        print('Estimerer position')
        est_pose = particle.estimate_pose(particles) # The estimate of the robots current pose

        for iters in range(3):
            print(iters)
            # Fetch next frame
            img = cam.get_next_frame()

            # Detect objects
            print('Forsøger at detektere via kameraet.')
            objectIDs, dists, angles = cam.detect_aruco_objects(img)

            sigma_theta = 0.57
            sigma_d = 5.0
            print('Tilføjer støj.')
            particle.add_uncertainty(particles, sigma_d, sigma_theta)

            if not isinstance(objectIDs, type(None)):
                print('selflocalize opdager landmarks: ' +  objectIDs)
                landmarks_lst = _utils.make_list_of_landmarks(objectIDs, dists, angles, landmarks_dict)
                
                # omregner fra milimeter til centimeter
                for landmark in landmarks_lst:
                    landmark.x /= 10
                    print(landmark.x)
                    landmark.z /= 10
                    print(landmark.z)
                    landmark.tvec = (landmark.tvec[0]/10, landmark.tvec[1]/10)
                    print(landmark.tvec)

                print('Opdaterer vægte')
                _utils.update_weights(sigma_d, sigma_theta, landmarks_lst, particles)    
                
                print('Normaliserer vægte')
                _utils.normalize_weights(particles)

                intervals = _utils.make_intervals(particles)

                print('Genererer nye partikler')
                particles = _utils.generate_new_particles(num_particles, particles, intervals)

                '''
                # Draw detected objects
                cam.draw_aruco_objects(img)
                '''
            else:
                # No observation - reset weights to uniform distribution
                print('selflocalization detekterede ikke nogle landmarks.')
                for p in particles:
                    p.setWeight(1.0/num_particles)
 
            est_pose = particle.estimate_pose(particles)

    finally:
        # Clean-up capture thread
        cv2.destroyAllWindows()
        cam.terminateCaptureThread()

        est_pose = _utils.Node(est_pose.x * 10.0, est_pose.y * 10.0, None) # VI GLEMMER AT EKSPORTERE VINKLEN!!!
        est_pose.setTheta(est_pose.theta)

        return est_pose, particles


def selflocalize_360_degrees(cam, show, params):  
    arlo_position, particles = selflocalize(cam, show, params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7])
    # drejer 360 grader om sig selv og selflocalizer
    for _ in range(18):
        arlo.go_diff(leftWheelFactor*standardSpeed, rightWheelFactor*standardSpeed, 0, 1)
        turnSeconds = _utils.degreeToSeconds(20)
        _utils.wait(turnSeconds)
        arlo.stop()
        _utils.wait(2.0)
        arlo_position, particles = selflocalize(cam, show, params[0], params[1], params[2], params[3], arlo_position, particles, 0.0, 20.0)
        print('Arlo befinder sig på position: ', math.floor(arlo_position.x), math.floor(arlo_position.z), math.floor(arlo_position.theta))
    return arlo_position, particles
   
def turn_and_watch(direction, img, landmarkIDs):
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
    
    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    aruco_corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(img, arucoDict)
    
    landmark_spotted = False
    for id in landmarkIDs:
        if ids is not None:
            if id in ids:
                landmark_spotted = True

    # If at least one marker is detected
    if len(aruco_corners) > 0 and landmark_spotted:
        for i in range(len(ids)):
            corner = aruco_corners[i]
            # The corners are ordered as top-left, top-right, bottom-right, bottom-left
            top_left = corner[0][0]
            top_right = corner[0][1]
            bottom_right = corner[0][2]
            bottom_left = corner[0][3]

            print('Landmark ' + str(ids[i]) + 'detekteret via turn_and_watch.')
           # print(top_left, top_right, bottom_right, bottom_left)

        return True
    else: 
        arlo.go_diff(leftWheelFactor*standardSpeed, rightWheelFactor*standardSpeed, 0, 1)
        turnSeconds = _utils.degreeToSeconds(20)
        _utils.wait(turnSeconds)
        arlo.stop()
        _utils.wait(2.0)
        print('Intet landmark fundet.')

        return False
    
def RRT(goal, mapsizex, mapsizez, maxiter, seen_landmarks, rootNode, stepLength, bias):
    '''
    Funktionen returnerer et RRT med Arlo som rod.
    Argumenter:
        goal:  vores målposition som node
        mapsizex: størrelsen på banen i x-aksen
        mapsizez: størrelsen på banen i z-aksen
        maxiter:  det maksimale antal iterationer
        seen_landmarks: liste af seen_landmarks
        rootNode:  Arlos position
        stepLength:   distancen mellem den returnerede node og nearest_node
        bias:   en værdi mellem 0 og 100, der jo højere den er, angiver, at vi vurderer, at banen er meget simpel.
    '''
    print('Udregner RRT med goal på position ' + str(goal.x) + ', ' + str(goal.z))
    G = _utils.Graf([rootNode], [])
    iters = 0
    #print('hej4')
    while iters < maxiter:
        iters += 1
        random_number = random.randint(0, 100)
        if random_number < bias:
            steering_node = _utils.Node(goal.x, goal.z, None)
        else:
            steering_node = _utils.Node(random.randrange(-500, mapsizex), random.randrange(-500, mapsizez), None)
        
        box_radius = 400.0

        nearest_node = _utils.find_nearest_node(steering_node, G)
        new_node = _utils.steer(nearest_node, steering_node, stepLength)
        halfway_node = _utils.steer(nearest_node, steering_node, stepLength/2)

        #print(new_node.x)
        #print(new_node.z)
        #print(iters) 
        if _utils.is_spot_free(new_node, seen_landmarks, box_radius) and _utils.is_spot_free(halfway_node, seen_landmarks, box_radius):
            #print("new node:", new_node.x, new_node.z, "steering node:", steering_node.x, steering_node.z)
            G.nodes.append(new_node)
            G.edges.append((nearest_node, new_node))
            #print('Node med position ' + str(math.floor(new_node.x)) + ', ' + str(math.floor(new_node.z)) + ' tilføjes til RRT')

            goal_radius = 200.0
            if not _utils.is_spot_free(new_node, [goal], goal_radius):
                goalNode = _utils.Node(goal.x, goal.z, new_node)
                G.nodes.append(goalNode)
                G.edges.append((new_node, goal))
                return G, new_node
            
            # INDSÆT TJEK FOR, OM DEN NYE NODE HAR FRI PASSAGE TIL GOAL
    
    return G, new_node

def make_RRT_path(img, arucoDict, draw, arlo_position, goal, rally_landmarks):
    '''
    Funktionen danner et RRT og kører efter det.
    Argumenter:
        img:  billedet fra kameraet
        arucoDict:   en dictonary af QR koder.
        draw: bool, der angiver, om funktionen skal tegne et kort
        goal: et Landmark, der er vores midlertidige mål
    '''

    seen_landmarks = _utils.landmark_detection(img, arucoDict)
    for seen_landmark in seen_landmarks:
        seen_landmark.x = seen_landmark.x + arlo_position.x
        seen_landmark.z = seen_landmark.z + arlo_position.z
        seen_landmark.tvec = (seen_landmark.x, seen_landmark.z)
        print('Landmarks, der kan ses: ' + str(seen_landmark.id) + ' på position ' + str(math.floor(seen_landmark.x)) + ', ' + str(math.floor(seen_landmark.z)))

    for landmark in rally_landmarks:
        seen_landmarks.append(landmark)
        print('Landmarks, vi ved er der: ' + str(landmark.id) + ' på position ' + str(math.floor(landmark.x)) + ', ' + str(math.floor(landmark.z)))

    rootNode = arlo_position # Arlos position
    maxiter = 200
    bias = 50

    localMap = _utils.Map(4500.0, 3500.0)

    #print(goal.x)
    #print(goal.z)

    G, new_node = RRT(goal, localMap.xlim, localMap.zlim, maxiter, seen_landmarks, rootNode, stepLength, bias)

    #print('G - noder:')
    #print(G.nodes)
    
    if draw:
        localMap.draw_tree(G)
        localMap.draw_landmarks(seen_landmarks)
        localMap.draw_goal(goal)

        path = []
        goalNode = G.nodes[-1]
        # print(len(G.nodes))
        # positions_list = []
        # for i in range(len(G.nodes)):
        #     positions_list.append((G.nodes[i].x, G.nodes[i].z))
        # print(positions_list)
        path.append(goalNode.parent)
        #print(type(path[-1]))
        #print(path)
        #localMap.draw_path(path)
        print('Driving towards', goalNode.x, goalNode.z)

        notRoot = True

        if not goalNode.x == rootNode.x and not goalNode.z == rootNode.z:
            while notRoot:
                if path[-1].parent != None:
                    path.append(path[-1].parent)
                else:
                    notRoot = False

            path.pop()
            path.reverse()
            #print("PATH: ", path)

            localMap.draw_path(path)
            localMap.show_map(arlo_position)
    
        return path
    
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

def drive_one_move(frontLimit, sideLimit):
    pingFront, pingLeft, pingRight, pingBack = _utils.sensor()

    if pingFront < 100:
        _utils.sharp_turn('left', 90.0)
    else:
        arlo.go_diff(leftWheelFactor*standardSpeed, rightWheelFactor*standardSpeed, 1, 1)
    
    dirLst = ['right', 'left']

    while (pingFront > frontLimit and pingLeft > sideLimit and pingRight > sideLimit):
        pingFront, pingLeft, pingRight, pingBack = _utils.sensor()
        #print(pingFront, pingLeft, pingRight, pingBack)
        sleep(0.041)
    
    # three sensors detected
    if (pingFront < frontLimit and pingLeft < sideLimit and pingRight < sideLimit):
        print('Front, left og right sensor blokeret.')
        _utils.turn_and_sense('left')  

    # two sensors detected
    elif (pingFront < frontLimit and pingLeft < sideLimit):
        print('Front og left sensor blokeret.')
        _utils.turn_and_sense('right')
    elif (pingFront < frontLimit and pingRight < sideLimit):
        print('Front og right sensor blokeret.')
        _utils.turn_and_sense('left')
    elif (pingLeft < sideLimit and pingRight < sideLimit):
        print('Left og right sensor blokeret.')
        _utils.turn_and_sense('left')

    # one sensors deteced
    elif(pingFront <= frontLimit):
        print('Front sensor blokeret.')
        randomDirection = random.choice(dirLst)
        _utils.turn_and_sense(randomDirection)
    elif (pingLeft < sideLimit): 
        print('Left sensor blokeret.')
        _utils.turn_and_sense('right')
    elif (pingRight < sideLimit):
        print('Right sensor blokeret.')
        _utils.turn_and_sense('left')
        
def drive_free_carefully(seconds, frontLimit, sideLimit):
    '''
    Funktionen kører robotten frem, indtil der er en forhindring. Så drejer den, indtil at der er frit.
    '''
    start = time.perf_counter()
    isDriving = True
    #pingFront, pingLeft, pingRight, pingBack = _utils.sensor()

    while isDriving:
        drive_one_move(frontLimit, sideLimit)
        if (time.perf_counter() - start > seconds):
            isDriving = False

def landmark_reached(reached_node, temp_goal):
    if _utils.dist(reached_node, temp_goal) < 350.0:
        return True
    else:
        return False

def drive_path_and_sense(path, temp_goal, num_steps, stepLength, arlo_position):
    prevnode = arlo_position
    for i in range(num_steps):
        node = path[i]
        direction, degrees = _utils.find_turn_angle(prevnode, node)
        print('Drejer ' + str(degrees) + 'grader i retning ' + str(direction)) 

        # Robotten drejer
        _utils.sharp_turn(direction, degrees)
        
        # Robotten tjekker, om der er frit
        pingFront, pingLeft, pingRight, pingBack = _utils.sensor()
        if pingFront < 500 and pingLeft < 400 and pingRight < 400:
            return False
    
        # Robotten kører, mens den holder øje, om den støder ind i noget
        drive_carefully('forwards', stepLength/1000)

        # Tjekker, om der er blevet afbrudt grundet spærret vej
        pingFront, pingLeft, pingRight, pingBack = _utils.sensor()
        if (pingFront < 350 and pingLeft < sideLimit and pingRight < sideLimit):
            return False

        # Robotten drejer tilbage
        _utils.sharp_turn(_utils.inverse_direction(direction), degrees)
        
        if landmark_reached(node, temp_goal):
            return True

        prevnode = node
    return False

onRobot = True

def isRunningOnArlo():
    """Return True if we are running on Arlo, otherwise False.
      You can use this flag to switch the code from running on you laptop to Arlo - you need to do the programming here!
    """
    return onRobot


def camera_setup():
    '''
    Funktionen åbner kameraet og udfører en kommando.
    Argumenter:
        command:  kommando
        show:   bool, der angiver, om kameraets output skal vises
    '''

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
    '''
    time.sleep(1)  # _utils.wait for camera to setup

    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)

    print("Opening and initializing camera for self_localization")
    if isRunningOnArlo():
        # XXX: You need to change this path to point to where your robot.py file is located
        sys.path.append("../")

    try:
        #import Ex5.src.handout.python.robot as robot
        onRobot = True
    except ImportError:
        print("selflocalize.py: robot module not present - forcing not running on Arlo!")
        onRobot = False
    
    print("Opening and initializing camera")
    if camera.isRunningOnArlo():
        cam = camera.Camera(0, 'arlo', useCaptureThread = True)
    else:
        cam = camera.Camera(0, 'macbookpro', useCaptureThread = True)


    return cam, arucoDict

def use_camera(cam, arucoDict, command, params, showcamera, show):
    # Open a window'''
    if showcamera:
        print('Kameraet vises.')
        WIN_RF = "Example 1"
        cv2.namedWindow(WIN_RF)
        cv2.moveWindow(WIN_RF, 100, 100)
    
    while cv2.waitKey(4) == -1: # _utils.wait for a key pressed event
        #image = cam.capture_array("main")
        image = cam.get_next_frame()
        
        # Show frames
        if showcamera:
            cv2.imshow(WIN_RF, image)

        if command == 'selflocalize':
            arlo_position, particles = selflocalize(cam, show, params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7])
            return arlo_position, particles

        elif command == '360_selflocalize':
            arlo_position, particles = selflocalize_360_degrees(cam, show, params)
            return arlo_position, particles

        elif command == 'turn_and_watch':
            return turn_and_watch('left', image, params[0])

        elif command == 'RRT':
            #arlo_position = selflocalize(cam, show, params[0], params[1], params[2], params[3], params[4])
            path = make_RRT_path(image, arucoDict, True, params[0], params[1], params[2])
            return path 

CRED, CGREEN, CBLUE, CCYAN, CYELLOW, CMAGENTA, CWHITE, CBLACK = _utils.bgr_colors()
'''
landmarkIDs = [1, 2, 3, 4, 1]
landmarks = {
    1: (0.0, 0.0),
    2: (0.0, 300.0),
    3: (400.0, 0.0),
    4: (400.0, 300.0),
}
landmark_colors = [CRED, CGREEN,  CCYAN, CYELLOW] # Colors used when drawing the landmarks
'''

landmarkIDs = [1, 2, 3, 4]
landmarks_dict = {
    1: (0.0, 0.0),
    2: (0.0, 3000.0),
    3: (4000.0, 0.0),
    4: (4000.0, 3000.0)
}
landmark_colors = [CRED, CGREEN, CBLUE, CYELLOW] # Colors used when drawing the landmarks

num_steps = 3

stepLength = 1000.0 # milimeter

def robo_rally(landmarkIDs, landmarks_dict, landmark_colors, showcamera, show, frontLimit, sideLimit):
    globalMap = _utils.Map(3000, 4000) # kortets størrelse
    for landmarkID in landmarkIDs:
        globalMap.landmarks.append(_utils.Landmark(None, None, None, landmarkID, landmarks_dict[landmarkID])) # liste af alle spottede landmarks
    rally_landmarks = globalMap.landmarks # liste af landmarks i rækkefølgen efter rallyet

    cam, arucoDict = camera_setup()

    arlo_position = _utils.Node(500.0, 0.0, None)

    particles = []
    path = []

    for temp_goal in rally_landmarks:
        print('Søger efter landmark ' + str(temp_goal.id))
        landmarkfound = False

        temp_goal_Node = _utils.Node(temp_goal.x, temp_goal.z, None)

        while not landmarkfound:

            lost = True
            iterations = 0

            

            while lost:
                if iterations < 18:
                    found = use_camera(cam, arucoDict, 'turn_and_watch', [landmarkIDs], showcamera, show)
                    if found:
                        lost = False
                    else:
                        print('Drejer og forsøger at finde landmarks.')
                    iterations += 1
                else:
                    print('Påbegynder frikørsel for at få øje på et landmark')
                    drive_free_carefully(2.0, frontLimit, sideLimit)
                    iterations = 0
                    particles = []
            
            arlo.stop()
            print('Begynder selflokalisering.')
            arlo_position, particles = use_camera(cam, arucoDict, '360_selflocalize', [3, landmarkIDs, landmarks_dict, landmark_colors, arlo_position, particles, [], stepLength], showcamera, show)
            
            #arlo_position, particles = use_camera(cam, arucoDict, 'selflocalize', [3, landmarkIDs, landmarks_dict, landmark_colors, arlo_position, particles, path, stepLength], showcamera, show)
            
            #print(math.floor(arlo_position.x), math.floor(arlo_position.z), math.floor(arlo_position.theta))
            #arlo_node = _utils.Node(arlo_position.x, arlo_position.z, None)
            landmarkfound = landmark_reached(arlo_position, temp_goal_Node)

            print('Arlo befinder sig på position ', math.floor(arlo_position.x), math.floor(arlo_position.z), math.floor(arlo_position.theta))

            
            if not landmarkfound:
                print('Påbegynder RRT-sti.')
                path = use_camera(cam, arucoDict, 'RRT', [arlo_position, temp_goal_Node, rally_landmarks], showcamera, show) #laver en path med RRT, skal også have arlo position
                if len(path) > 1:
                    print('Kører ' + str(num_steps) + ' trin af vores RRT sti.')
                    landmarkfound = drive_path_and_sense(path, temp_goal_Node, num_steps, stepLength, arlo_position) # kører num_steps antal trin af RRT path, stopper, hvis sensorerne opfanger noget.
                    path = path[num_steps]
                else:
                    print('RRT-træ betod kun af Arlos position, æv :(')

            arlo_position, particles = use_camera(cam, arucoDict, '360_selflocalize', [3, landmarkIDs, landmarks_dict, landmark_colors, arlo_position, particles, path, stepLength], showcamera, show)
            #arlo_position, particles = use_camera(cam, arucoDict, 'selflocalize', [10, landmarkIDs, landmarks_dict, landmark_colors, arlo_position, particles, path, stepLength], showcamera, show)
            landmarkfound = landmark_reached(arlo_position, temp_goal_Node)
                    
        if landmarkfound:
            print('Landmark ' + str(temp_goal.id) + ' er fundet! Tillykke!')

    arlo.stop()
    print('Rute færdiggjort!')
    
frontLimit = 500.0
sideLimit = 400.0

robo_rally(landmarkIDs, landmarks_dict, landmark_colors, False, False, frontLimit, sideLimit)