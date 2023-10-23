import cv2
import camera
import numpy as np
import time
from timeit import default_timer as timer
import sys
import _utils
import robot
import particle

arlo = robot.Robot()

def drive(sensorer):
'''
    Tjekker om den kan fortsætte denne på gældende vej
        - ping sensorerne om den er i god nok distance fra obstacles eller landmarks
'''
    
def drive_carefull(): 
'''
    Tjekker om den har ramt et obstacle eller ej.
        - Hvis ja, skal den ping sensorer og finde hvilken vej er den bedste mulige vej
        - Hvis nej, skal den fortsætte som den gjorde
'''



           
        


## hav noget med et kort, hvor tingene kan tegnes ind



### SOFIE ###




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


def turn_and_watch(direction, img):
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
    aruco_corners, _, _ = cv2.aruco.detectMarkers(img, arucoDict)
    # If at least one marker is detected
    if len(aruco_corners) > 0:
        for corner in aruco_corners:
            # The corners are ordered as top-left, top-right, bottom-right, bottom-left
            top_left = corner[0][0]
            top_right = corner[0][1]
            bottom_right = corner[0][2]
            bottom_left = corner[0][3]

            print('Landmark detected')
            print(top_left, top_right, bottom_right, bottom_left)

        return True
    else: 
        arlo.go_diff(leftWheelFactor*standardSpeed, rightWheelFactor*standardSpeed, 0, 1)
        turnSeconds = degreeToSeconds(20)
        wait(turnSeconds)
        arlo.stop()
        wait(2.0)

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
    G = Graf([rootNode], [])
    iters = 0
    while iters < maxiter:
        iters += 1
        random_number = random.randint(0, 100)
        if random_number < bias:
            steering_node = Node(goal.x, goal.z, None)
        else:
            steering_node = Node(random.randrange(-mapsizex, mapsizex), random.randrange(0, mapsizez), None)
        
        box_radius = 400.0

        nearest_node = find_nearest_node(steering_node, G)
        new_node = steer(nearest_node, steering_node, stepLength)
        halfway_node = steer(nearest_node, steering_node, stepLength/2)
        
        if is_spot_free(new_node, landmarks, box_radius) and is_spot_free(halfway_node, landmarks, box_radius):
            #print("new node:", new_node.x, new_node.z, "steering node:", steering_node.x, steering_node.z)
            G.nodes.append(new_node)
            G.edges.append((nearest_node, new_node))

            goal_radius = 200.0
            if not is_spot_free(new_node, [goal], goal_radius):
                goalNode = Node(goal.x, goal.z, new_node)
                G.nodes.append(goalNode)
                G.edges.append((new_node, goal))
                return G, new_node
            
            # INDSÆT TJEK FOR, OM DEN NYE NODE HAR FRI PASSAGE TIL GOAL
    
    return G, new_node

def make_RRT_path(img, arucoDict, draw, arlo_position, goal):
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

    rootNode = _utils.Node(arlo_position[0], arlo_position[1], None) # Arlos position
    stepLength = 300.0 # milimeter
    maxiter = 1500
    bias = 50

    localMap = _utils.Map(2000, 4500)

    G, new_node = RRT(goal, localMap.xlim, localMap.zlim, maxiter, seen_landmarks, rootNode, stepLength, bias)

    if draw:
        localMap.draw_tree(G)
        localMap.draw_landmarks(landmarks)
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


def drive_path_and_sense(path, stepLength):
        prevnode = _utils.Node(0, 0, None)
        for node in path:
            direction, degrees = _utils.find_turn_angle(prevnode, node)
            print(direction, degrees) 
            _utils.sharp_turn(direction, degrees)
            drive('forwards', stepLength/1000)
            _utils.sharp_turn(_utils.inverse_direction(direction), degrees)
            prevnode = node
        
        print('Arrived at goal!')



def camera(command, params, show):
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

    time.sleep(1)  # wait for camera to setup

    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    
    # Open a window
    if show:
        WIN_RF = "Example 1"
        cv2.namedWindow(WIN_RF)
        cv2.moveWindow(WIN_RF, 100, 100)
    
    while cv2.waitKey(4) == -1: # Wait for a key pressed event
        image = cam.capture_array("main")
        
        # Show frames
        if show:
            cv2.imshow(WIN_RF, image)

        if command == 'selflocalize':
            arlo_position = selflocalize(cam, show, params[0])
            return arlo_position

        elif command == 'drive_to_landmark':
            if turn_and_watch('left', image) == True: 
                arlo.stop()
                landmarks = landmark_detection(image, arucoDict)
                drive_to_landmarks(landmarks)
                time.sleep(15)
        
        elif command == 'turn_and_watch':
            turn_and_watch('right', image)

        elif command == 'RRT':
            arlo_position = selflocalize(cam, show, params[0])
            




def check_if_in_range(): 

    #Distancen til fra arlo til landmark
    landmark_dist = 

    #Den range som vi minimum skal være fra landmark 
    check_range = 300
 
    
    if landmark_dist > check_range:
        return True

    else: 
        return False 

          



CRED, CGREEN, CBLUE, CCYAN, CYELLOW, CMAGENTA, CWHITE, CBLACK = _utils.bgr_colors()

landmarkIDs = [1, 2, 3, 4, 1]
landmarks = {
    1: (0.0, 0.0),
    2: (0.0, 300.0),
    3: (400.0, 0.0),
    4: (400.0, 300.0),
}
landmark_colors = [CRED, CGREEN] # Colors used when drawing the landmarks

num_of_steps = 2

def robo_rally(landmarkIDs):
    
    globalMap = _utils.Map(3000, 4000) # kortets størrelse

    for landmarkID in landmarkIDs:
        globalMap.landmarks.append(_utils.Landmark(None, None, None, landmarkID, landmarks[landmarkID])) # liste af alle spottede landmarks
    
    rally_landmarks = globalMap.landmarks # liste af landmarks i rækkefølgen efter rallyet

    lost = True
    for goal in rally_landmarks:
        print('Søger efter landmark ' + str(goal.id))
        landmarkfound = False

        while not landmarkfound:

            arlo_position = camera('selflocalize', [200], True)

            if arlo_position != int:
                lost = True

            if lost:
                _utils.turn_and_watch()
                arlo_position = camera('selflocalize', [200], True)
                lost = False
            
            
            path = camera('RRT', [200, goal], True) # skriv

            path_clear = True
            
            while path_clear:
                for i in range(num_of_steps):
                    path_clear = _utils.drive_to_landmark_and_sense() # stopper hvis den støder på noget
                    
                    landmarkfound = check_if_in_range(goal.id)
        
        if landmarkfound:
            print('Landmark ' + str(goal.id) + ' er fundet! Tillykke!')

    
    # gør noget, hvis vi ikke kan se landmarket, men vi skal approache
                    

                




'''
FOR ID IN ROUTEID:

    WHILE NOT LANDMARKFOUND:

        HVIS LOST:
            TURN AND WATCH
            SELFLOCALIZE

        LAV RRT PLAN

        FOR I IN RANGE STEP_LÆNGDE:
            DRIVE AND SENSE # 1. den drejer en vinkel 2. den kører en distance IMENS HOLDER DEN ØJE MED SINE SONARER
            SELFLOCALIZE

            LANDMARKFOUND = CHECK IF LANDMARK IS FOUND BASED ON DISTANCE (ID)

        WHEN LANDMARK IS LOST OUT OF SIGHT BUT WE THINK WE ARE CLOSE:
            APPROACH AND SENSE - kør tæt på, mens den sørger for, at den ikke kører ind ind i.

    IF LANDMARKFOUND CONTINUE

'''

### KAMERA ###  
          
def turn_and_sense(direction):
    '''
    Funktionen drejer robotten om egen akse, indtil der er frit.
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

    arlo.go_diff(leftWheelFactor*standardSpeed, rightWheelFactor*standardSpeed, r, l)
    pingFront, pingLeft, pingRight, pingBack = sensor()

    while (pingFront < 500 and pingLeft < 400 and pingRight < 400):
        pingFront, pingLeft, pingRight, pingBack = sensor()

def drive_and_sense():
    '''
    Funktionen kører robotten frem, indtil der er en forhindring. Så drejer den, indtil at der er frit.
    '''
    pingFront, pingLeft, pingRight, pingBack = sensor()

    if pingFront < 100:
        sharp_turn('left', 90.0)
    else:
        arlo.go_diff(leftWheelFactor*standardSpeed, rightWheelFactor*standardSpeed, 1, 1)
    
    dirLst = ['right', 'left']

    while (pingFront > 350 and pingLeft > 250 and pingRight > 250):
        pingFront, pingLeft, pingRight, pingBack = sensor()
        print(pingFront, pingLeft, pingRight, pingBack)
        sleep(0.041)
    
    # three sensors detected
    if (pingFront < 350 and pingLeft < 250 and pingRight < 250):
        turn_and_sense('left')  

    # two sensors detected
    elif (pingFront < 350 and pingLeft < 250):
        turn_and_sense('right')
    elif (pingFront < 350 and pingRight < 250):
        turn_and_sense('left')
    elif (pingLeft < 250 and pingRight < 250):
        turn_and_sense('left')

    # one sensors deteced
    elif(pingFront <= 350):
        randomDirection = random.choice(dirLst)
        turn_and_sense(randomDirection)
    elif (pingLeft < 250): 
        turn_and_sense('right')
    elif (pingRight < 250):
        turn_and_sense('left')
    
        
    drive_and_sense()