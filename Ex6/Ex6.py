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

def selflocalize(cam, showGUI, maxiters):
    try:
        # Initialize particles
        num_particles = 1000
        particles = _utils.initialize_particles(num_particles)
        est_pose = particle.estimate_pose(particles) # The estimate of the robots current pose

        # Allocate space for world map
        world = np.zeros((500,500,3), dtype=np.uint8)

        # Draw map
        _utils.draw_world(est_pose, particles, world, landmarks, landmarkIDs, landmark_colors)

        Xlst = []
        iters = 0
        while iters > maxiters:
            # Fetch next frame
            colour = cam.get_next_frame()
            
            # Detect objects
            objectIDs, dists, angles = cam.detect_aruco_objects(colour)
            
            
            Xlst.append(particles) # således at Xlst[iter] er lig de nuværende particles
            
            sigma_theta = 0.2 #0.57
            sigma_d = 2.0 #5.0
            particle.add_uncertainty(particles, sigma_d, sigma_theta)

            if not isinstance(objectIDs, type(None)):
                landmarks_lst = _utils.make_list_of_landmarks(objectIDs, dists, angles, landmarks)
                
                _utils.update_weights(sigma_d, sigma_theta, landmarks_lst, particles)
                    
                _utils.normalize_weights(particles)

                intervals = _utils.make_intervals(particles)

                particles = _utils.generate_new_particles(num_particles, particles, intervals)

                # Draw detected objects
                cam.draw_aruco_objects(colour)
            else:
                # No observation - reset weights to uniform distribution
                for p in particles:
                    p.setWeight(1.0/num_particles)
        
            est_pose = particle.estimate_pose(particles) # The estimate of the robots current pose

            #def path_planning(img, arucoDict, draw, drive_after_plan): 
            #    
            #    _utils.RRT((0.0, 1500.0), 4000.0, 4000.0, 100, landmarks, est_pose, 300.0, 50)
            #
            #    _utils.run_RRT(img, arucoDict, draw, drive_after_plan)

            if showGUI:
                # Draw map
                _utils.draw_world(est_pose, particles, world, landmarks, landmarkIDs, landmark_colors)
            
            iters += 1

    finally: 
        # Make sure to clean up even if an exception occurred
        
        # Close all windows
        cv2.destroyAllWindows()

        # Clean-up capture thread
        cam.terminateCaptureThread()

        return est_pose, landmarks_lst

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
            print(top_left, top_right, bottom_right, bottom_left)

        return True
    else: 
        arlo.go_diff(leftWheelFactor*standardSpeed, rightWheelFactor*standardSpeed, 0, 1)
        turnSeconds = _utils.degreeToSeconds(20)
        _utils.wait(turnSeconds)
        arlo.stop()
        _utils.wait(2.0)
        print('No landmark detected')

        return False
    
def RRT(goal, mapsizex, mapsizez, maxiter, landmarks, rootNode, stepLength, bias):
    '''
    Funktionen returnerer et RRT med Arlo som rod.
    Argumenter:
        goal:  vores målposition som node
        mapsizex: størrelsen på banen i x-aksen
        mapsizez: størrelsen på banen i z-aksen
        maxiter:  det maksimale antal iterationer
        landmarks: liste af landmarks
        rootNode:  Arlos position
        stepLength:   distancen mellem den returnerede node og nearest_node
        bias:   en værdi mellem 0 og 100, der jo højere den er, angiver, at vi vurderer, at banen er meget simpel.
    '''
    G = _utils.Graf([rootNode], [])
    iters = 0
    while iters < maxiter:
        iters += 1
        random_number = random.randint(0, 100)
        if random_number < bias:
            steering_node = _utils.Node(goal.x, goal.z, None)
        else:
            steering_node = _utils.Node(random.randrange(-mapsizex, mapsizex), random.randrange(0, mapsizez), None)
        
        box_radius = 400.0

        nearest_node = _utils.find_nearest_node(steering_node, G)
        new_node = _utils.steer(nearest_node, steering_node, stepLength)
        halfway_node = _utils.steer(nearest_node, steering_node, stepLength/2)
        
        if _utils.is_spot_free(new_node, landmarks, box_radius) and _utils.is_spot_free(halfway_node, landmarks, box_radius):
            #print("new node:", new_node.x, new_node.z, "steering node:", steering_node.x, steering_node.z)
            G.nodes.append(new_node)
            G.edges.append((nearest_node, new_node))

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
        print('Landmarks, der kan ses: ', seen_landmark.id)

    for landmark in rally_landmarks:
        seen_landmarks.append(landmark)

    rootNode = _utils._utils.Node(arlo_position[0], arlo_position[1], None) # Arlos position
    maxiter = 1500
    bias = 50

    localMap = _utils.Map(2000, 4500)

    G, new_node = RRT(goal, localMap.xlim, localMap.zlim, maxiter, seen_landmarks, rootNode, stepLength, bias)

    if draw:
        localMap.draw_tree(G)
        localMap.draw_landmarks(seen_landmarks)
        localMap.draw_goal(goal)

        path = []
        goalNode = G.nodes[-1]
        path.append(goalNode.parent)
        localMap.draw_path(path)
        print('driving towards', goalNode)

        notRoot = True

        while notRoot:
            if path[-1].parent != None:
                path.append(path[-1].parent)
            else:
                notRoot = False

        path.pop()
        path.reverse()
        print("PATH: ", path)

        localMap.draw_path(path)
        localMap.show_map()
    
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
    _utils.wait_and_sense(driveSeconds)


def drive_one_move():
    pingFront, pingLeft, pingRight, pingBack = _utils.sensor()

    if pingFront < 100:
        _utils.sharp_turn('left', 90.0)
    else:
        arlo.go_diff(leftWheelFactor*standardSpeed, rightWheelFactor*standardSpeed, 1, 1)
    
    dirLst = ['right', 'left']

    while (pingFront > 350 and pingLeft > 250 and pingRight > 250):
        pingFront, pingLeft, pingRight, pingBack = _utils.sensor()
        #print(pingFront, pingLeft, pingRight, pingBack)
        sleep(0.041)
    
    # three sensors detected
    if (pingFront < 350 and pingLeft < 250 and pingRight < 250):
        _utils.turn_and_sense('left')  

    # two sensors detected
    elif (pingFront < 350 and pingLeft < 250):
        _utils.turn_and_sense('right')
    elif (pingFront < 350 and pingRight < 250):
        _utils.turn_and_sense('left')
    elif (pingLeft < 250 and pingRight < 250):
        _utils.turn_and_sense('left')

    # one sensors deteced
    elif(pingFront <= 350):
        randomDirection = random.choice(dirLst)
        _utils.turn_and_sense(randomDirection)
    elif (pingLeft < 250): 
        _utils.turn_and_sense('right')
    elif (pingRight < 250):
        _utils.turn_and_sense('left')
        
def drive_free_carefully(seconds):
    '''
    Funktionen kører robotten frem, indtil der er en forhindring. Så drejer den, indtil at der er frit.
    '''
    start = time.perf_counter()
    isDriving = True
    pingFront, pingLeft, pingRight, pingBack = _utils.sensor()

    while isDriving:
        drive_one_move()
        if (time.perf_counter() - start > seconds):
            isDriving = False

def landmark_reached(reached_node, temp_goal):
    if _utils.dist(reached_node, temp_goal) < 350.0:
        return True
    else:
        return False

def drive_path_and_sense(path, temp_goal, num_steps, stepLength):
    prevnode = _utils._utils.Node(0, 0, None)
    for i in range(num_steps):
        node = path[i]
        direction, degrees = _utils.find_turn_angle(prevnode, node)
        print(direction, degrees) 

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
        if (pingFront < 350 and pingLeft < 250 and pingRight < 250):
            return False

        # Robotten drejer tilbage
        _utils.sharp_turn(_utils.inverse_direction(direction), degrees)
        
        if landmark_reached(node, temp_goal):
            return True

        prevnode = node
    return False

def use_camera(command, params, show):
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
    picam2_config = cam.create_video_configuration({"size": imageSize, "format": 'RGB888'},
                                                                controls={"FrameDurationLimits": (frame_duration_limit, frame_duration_limit)},
                                                                queue=False)
    cam.configure(picam2_config) # Not really necessary
    cam.start(show_preview=False)

    print(cam.camera_configuration()) # Print the camera configuration in use

    time.sleep(1)  # _utils.wait for camera to setup

    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    
    # Open a window
    if show:
        WIN_RF = "Example 1"
        cv2.namedWindow(WIN_RF)
        cv2.moveWindow(WIN_RF, 100, 100)
    
    while cv2.waitKey(4) == -1: # _utils.wait for a key pressed event
        image = cam.capture_array("main")
        
        # Show frames
        if show:
            cv2.imshow(WIN_RF, image)

        if command == 'selflocalize':
            arlo_position = selflocalize(cam, show, params[0])
            return arlo_position

        elif command == 'turn_and_watch':
            return turn_and_watch('left', image, params[0])

        elif command == 'RRT':
            arlo_position = selflocalize(cam, show, params[0])
            path = make_RRT_path(image, arucoDict, True, arlo_position, params[1], params[2])
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

landmarkIDs = [1, 2]
landmarks = {
    1: (150.0, 200.0),
    2: (0.0, 200.0)
}
landmark_colors = [CRED, CGREEN] # Colors used when drawing the landmarks


num_steps = 3

stepLength = 300.0 # milimeter

def robo_rally(landmarkIDs):
    globalMap = _utils.Map(3000, 4000) # kortets størrelse
    for landmarkID in landmarkIDs:
        globalMap.landmarks.append(_utils.Landmark(None, None, None, landmarkID, landmarks[landmarkID])) # liste af alle spottede landmarks
    rally_landmarks = globalMap.landmarks # liste af landmarks i rækkefølgen efter rallyet

    for temp_goal in rally_landmarks:
        print('Søger efter landmark ' + str(temp_goal.id))
        landmarkfound = False

        while not landmarkfound:

            if use_camera('turn_and_watch', [landmarkIDs], True):
                
                print('Begynder selflokalisering.')
                arlo_position = use_camera('selflocalize', [200], True)
                arlo_node = _utils._utils.Node(arlo_position[0], arlo_position[1], None)
                landmarkfound = landmark_reached(arlo_node, temp_goal)

                print('Arlo befinder sig på position ' + str(arlo_position))

                if not landmarkfound:
                    print('Påbegynder RRT-sti.')
                    path = use_camera('RRT', [200, temp_goal, rally_landmarks], True) #laver en path med RRT, skal også have arlo position
                    print('Kører ' + str(num_steps) + 'af vores RRT sti.')
                    landmarkfound = _utils.drive_path_and_sense(path, temp_goal, num_steps, stepLength) # kører num_steps antal trin af RRT path, stopper, hvis sensorerne opfanger noget.
        
            else:
                print('Påbegynder frikørsel for at få øje på et landmark')
                drive_free_carefully(2.0)

        if landmarkfound:
            print('Landmark ' + str(temp_goal.id) + ' er fundet! Tillykke!')

    arlo.stop()
    print('Rute færdiggjort!')

robo_rally(landmarkIDs)
            