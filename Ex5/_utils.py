### IMPORTERING ###
from time import sleep
import robot
import time
import random
import cv2 # OpenCV library
import time
from pprint import *
import numpy as np
from numpy import linalg

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

### PARAMETERUDREGNING ###

def degreeToSeconds(degrees):
    '''
    Funktionen omregner grader, der skal drejes, til sekunder, der skal ventes.
    Argumenter:
        degrees:    grader, der skal drejes
    '''
    ninetyDegreeTurnSeconds = 0.95 
    return (ninetyDegreeTurnSeconds/90) * degrees

def metersToSeconds(meters):
    '''
    Funktionen omregner meter, der skal køres, til sekunder, der skal ventes.
    Argumenter:
        meters:    meter, der skal køres
    '''
    oneMeterSeconds = 2.8
    return oneMeterSeconds * meters

### BEVÆGELSE ###

def wait(seconds):
    '''
    Funktionen venter i den tid, som robotten skal udføre en bestemt action.
    Samme tanke som Sleep, men kan måske sørge for den kan holde sig opdateret på sine omgivelser.
    Argumenter:
        start:    starttidspunkts
        seconds:  sekunder, der skal ventes
    '''
    start = time.perf_counter()
    isDriving = True
    while (isDriving):
        if (time.perf_counter() - start > seconds):
            isDriving = False

def drive(direction, meters):
    '''
    Funktionen kører robotten et antal meter.
    Argumenter:
        start:    starttidspunkt
        direction:  enten 'forwards', 'backwards', 'left' eller 'right'
        meters:   meter, der skal køres
    '''
    if direction == 'forwards':
        arlo.go_diff(leftWheelFactor*standardSpeed, rightWheelFactor*standardSpeed, 1, 1)
    elif direction == 'backwards':
        arlo.go_diff(leftWheelFactor*standardSpeed, rightWheelFactor*standardSpeed, 0, 0)

    driveSeconds = metersToSeconds(meters)
    wait(driveSeconds)

def sharp_turn(direction, degrees):
    '''
    Funktionen kører robotten et antal meter.
    Argumenter:
        start:    starttidspunkt
        direction:  enten 'right' eller 'left'
        degrees:   grader, der skal køres
    '''
    if direction == "right":
        arlo.go_diff(leftWheelFactor*standardSpeed, rightWheelFactor*standardSpeed, 1, 0)
    elif direction == "left":
        arlo.go_diff(leftWheelFactor*standardSpeed, rightWheelFactor*standardSpeed, 0, 1)
    
    turnSeconds = degreeToSeconds(degrees)
    wait(turnSeconds)

def soft_turn(direction, degrees):
    '''
    Funktionen kører robotten et antal meter.
    Argumenter:
        start:    starttidspunkt
        direction:  enten 'right' eller 'left'
        degrees:   grader, der skal køres
    '''
    if direction == "right":
        arlo.go_diff(leftWheelFactor*100, rightWheelFactor*40, 1, 1)
    elif direction == "left":
        arlo.go_diff(leftWheelFactor*40, rightWheelFactor*100, 1, 1)

    turnSeconds = degreeToSeconds(degrees)
    wait(turnSeconds)
            
### SAMMENSAT BEVÆGELSE ###

def move_in_square(meters):
    for _ in range(4):
        drive(1.0)

        sharp_turn('right', 90)
    

def move_around_own_axis(i): 
    for _ in range(i):
        sharp_turn('right', 360)
                
        sharp_turn('left', 360)


def move_in_figure_eight(i):
    soft_turn('right', 360)

    soft_turn('left', 360)

    for _ in range(i):   
            soft_turn('right', 360)
            
            soft_turn('left', 360)



### SENSORER ###

def sensor():
    front = arlo.read_front_ping_sensor()
    right = arlo.read_right_ping_sensor()
    left = arlo.read_left_ping_sensor()
    back = arlo.read_back_ping_sensor()
    
    return front, left, right, back

def sensor_print(): 
    '''
    Funktionen printer sensorenes målinger.
    '''
    while True:
        pingFront, pingBack, pingRight, pingLeft = sensor()
        print('front', pingFront, 'left', pingLeft, 'right', pingRight, 'back', pingBack)

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

### BRUG AF KAMERA ###

def turn_and_watch(direction, img):
    '''
    Funktionen drejer robotten om egen akse, indtil den har fundet et landmark, OG der er frit.
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

def drive_to_landmarks(landmarks_lst):
    landmark = landmarks_lst[0]

    dist = landmark[0]
    angle = landmark[1]
    direction = landmark[2]
    id = landmark[3]
    tvec = landmark[4]

    print('Driving towards landmark ' + str(id) + ', distance = ' + str(dist))

    print('tvec = ' + str(tvec))

    sharp_turn(direction, angle)

    drive('forwards', (dist-150)/1000)

    print('Arrived at landmark!')
    arlo.stop()
    return
    

def pose_estimation(img, arucoDict): 
    """Funktionen returnerer en liste af placeringer i kameraets koordinatsystem samt id på de givne QR-koder"""
    """Denne funktion skal finde positionen af et landmark i forhold til robotten og udregne vinklen og afstanden til landmarket."""

    # 1. Udregne afstanden til QR code - DONE 
    # 2. Udregn vinkel til QR code 
    # 3. Dreje robotten, så den er vinkelret på QR code
    # 4. Køre fremad, indtil robotten er tæt nok på QR code

    aruco_corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(img, arucoDict)
    w, h = 1280, 720
    focal_length = 1744.36 
    camera_matrix = np.array([[focal_length, 0, w/2], [0, focal_length, h/2], [0, 0, 1]])
    arucoMarkerLength = 145.0
    distortion = 0
    
    # Giver distancen til boxen
    _, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(aruco_corners, arucoMarkerLength, camera_matrix, distortion)
    
    lst = []
    
    for i in range(len(ids)):
        world_angle = np.arccos(np.dot(tvecs[i]/np.linalg.norm(tvecs[i]), np.array([0, 0, 1])))
        print(world_angle)
        if np.dot(tvecs[i], np.array([1, 0, 0])) < 0:
            direction = 'left'
        else:
            direction = 'right'
        print(direction)

        lst.append([linalg.norm(tvecs[i]), world_angle, direction, ids[i][0], tvecs[i]])
    
    lst.sort()
    
    namelst = ['distance', 'vinkel', 'retning', 'id', 'tvec']    

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


### RRT ###
import cv2 # Import the OpenCV library
import time
#from pprint import *
import numpy as np
import robot
from numpy import linalg
from time import sleep
import random
import _utils
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import math

arlo = robot.Robot()

try:
    import picamera2
    print("Camera.py: Using picamera2 module")
except ImportError:
    print("Camera.py: picamera2 module not available")
    exit(-1)

print("OpenCV version = " + cv2.__version__)




#radius_arlo = 22.5
#radius_QR = 17.5 


### KLASSER ###

class Landmark:
    def __init__(self, distance, vinkel, retning, id, tvec):
        self.distance = distance
        self.vinkel = vinkel
        self.retning = retning
        self.id = id
        self.tvec = tvec
        self.x = tvec[0]
        self.z = tvec[2]
        
class Node:
    def __init__(self, x, z, parent):
        self.x = x
        self.z = z
        self.path = []
        self.parent = parent

class Graf:
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges

class Map:
    def __init__(self, xlim, zlim):
        self.landmarks = []
        self.nodes = []
        self.edges = []
        self.xlim = xlim
        self.zlim = zlim

    def draw_landmarks(self, landmarks):
        self.landmarks += landmarks

        for landmark in self.landmarks:
            plt.plot(landmark.x, landmark.z, 'ro', markersize = 17)
            plt.annotate(str(landmark.id), xy=(landmark.x, landmark.z))
    
    def draw_tree(self, G):
        #for node in G.nodes:
        #    plt.plot(node.x, node.z, 'go')
        nodes = G.nodes
        edges = G.edges
        
        #print('Kanter: ' + str(G.edges))
        #print('Noder: ' + str(G.nodes))
        for edge in edges:

            #print('Nearest node: ', edge[0].x, edge[0].z, 'New node: ', edge[1].x, edge[1].z)

            plt.plot([edge[0].x, edge[1].x] , [edge[0].z, edge[1].z], 'yo-')
        
        #for node in nodes: 
        #    plt.plot(node.x, node.z, 'go')
    
    def draw_goal(self, goal):
        plt.plot(goal.x, goal.z, 'co', markersize = 17)
        plt.annotate('Mål', xy=(goal.x, goal.z))

    def draw_path(self, path):
        for node in path:
            plt.plot(node.x, node.z, 'mo')

    def show_map(self):
        plt.plot(0,0, 'bo')
        plt.annotate('ArloCinque', xy=(0, 0))
        
        plt.ylim(0, self.zlim)
        plt.xlim(-self.xlim, self.xlim)
        plt.xlabel('x')
        plt.ylabel('z')
        plt.show()
        #plt.savefig('rute.png')


### HJÆLPEFUNKTIONER, RRT ###
def dist(node1, node2):
    return np.sqrt((node1.x-node2.x)**2+(node1.z-node2.z)**2) 

def find_turn_angle(position, node):
    myradians = math.atan2(node.z-position.z, node.x-position.x)
    mydegrees = math.degrees(myradians)

    if mydegrees > 90:
        mydegrees = mydegrees - 90
        direction = 'left'
    else:
        mydegrees = 90 - mydegrees
        direction = 'right'

    return direction, mydegrees

def inverse_direction(direction):
    if direction == 'left':
        return 'right'
    elif direction == 'right':
        return 'left'

def landmark_detection(img, arucoDict): 
    """Funktionen returnerer en liste af placeringer i kameraets koordinatsystem samt id på de givne QR-koder"""
    """Denne funktion skal finde positionen af et landmark i forhold til robotten og udregne vinklen og afstanden til landmarket."""

    aruco_corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(img, arucoDict)
    w, h = 1280, 720
    focal_length = 1744.36 
    camera_matrix = np.array([[focal_length, 0, w/2], [0, focal_length, h/2], [0, 0, 1]])
    arucoMarkerLength = 145.0
    distortion = 0
    
    # Giver distancen til boxen
    _, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(aruco_corners, arucoMarkerLength, camera_matrix, distortion)
    
    lst = []
    
    if ids == []: 
        print("No marker detected")
        return lst
    else:
        for i in range(len(ids)):
            world_angle = np.arccos(np.dot(tvecs[i]/np.linalg.norm(tvecs[i]), np.array([0, 0, 1])))
            if np.dot(tvecs[i], np.array([1, 0, 0])) < 0:
                direction = 'left'
            else:
                direction = 'right'

            lst.append(Landmark(linalg.norm(tvecs[i]), world_angle, direction, ids[i][0], tvecs[i][0]))
            print('dist: ' + str(linalg.norm(tvecs[i])) + ', vinkel: ' + str(world_angle) + direction + ', id: ' + str(ids[i][0]) + ', tvec: ' + str(tvecs[i][0]))
        
        lst.sort(key=lambda x: x.distance, reverse=False)

    return lst

def is_spot_free(spot, landmarks, radius):

    for landmark in landmarks:        
        if dist(spot, landmark) < radius:
            print('Occupied by ' + str(landmark.id))
            return False
    #print('Spot free!')
    return True
    

def find_nearest_node(x_new, G):
    distances = []
    for node in G.nodes:
        distances.append(abs(dist(x_new, node)))
    nearest_i = np.argmin(distances)
    print('nearest i:', nearest_i)
    return G.nodes[nearest_i]



def steer(nearest_node, steering_node, stepLength):
    # s er antal skridt, så vi itere over denne hver gang et skridt er taget for at tjekke om der er frit. 
    steering_vec = np.array([steering_node.x, steering_node.z])
    nearest_vec = np.array([nearest_node.x, nearest_node.z])

    v = steering_vec - nearest_vec
    e = v / linalg.norm(v)
    q = nearest_vec + stepLength * e
    
    new_node = Node(q[0], q[1], nearest_node)
    return new_node

### RRT ###
#  Path planning med Rapidly-exploring random trees
def RRT(goal, mapsizex, mapsizez, maxiter, landmarks, rootNode, stepLength, bias):
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


### MAIN ###
def run_RRT(img, arucoDict, draw, drive):
    landmarks = landmark_detection(img, arucoDict)
    for landmark in landmarks:
        print('landmarks fundet: ', landmark.id)

    goal = Landmark(None, None, None, 'mål', [0, 0, 3000])  # lav om evt. landmarks[-1]

    rootNode = Node(0, 0, None)
    stepLength = 300.0 # milimeter
    maxiter = 1500
    bias = 50

    ourMap = Map(1500, 4000)

    G, new_node = RRT(goal, ourMap.xlim, ourMap.zlim, maxiter, landmarks, rootNode, stepLength, bias)

    if draw:
        ourMap.draw_tree(G)
        ourMap.draw_landmarks(landmarks)
        ourMap.draw_goal(goal)

        path = []
        goalNode = G.nodes[-1]
        path.append(goalNode.parent)
        ourMap.draw_path(path)
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

        ourMap.draw_path(path)
        ourMap.show_map()

    if drive:
        prevnode = Node(0, 0, None)
        
        #print('path', path)
        for node in path:
            direction, degrees = find_turn_angle(prevnode, node)
            print(direction, degrees) 
            _utils.sharp_turn(direction, degrees)
            #print('vinkelskift slut')
            #arlo.stop()
            _utils.drive('forwards', stepLength/1000)
            #print('kørsel slut')
            _utils.sharp_turn(inverse_direction(direction), degrees)
            prevnode = node
        
        print('Arrived at goal!')
    

def camera(command):
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
    #WIN_RF = "Example 1"
    #cv2.namedWindow(WIN_RF)
    #cv2.moveWindow(WIN_RF, 100, 100)
    
    while cv2.waitKey(4) == -1: # Wait for a key pressed event
        image = cam.capture_array("main")
        
        # Show frames
        #cv2.imshow(WIN_RF, image)

        if command == 'build_map':
            landmarks = landmark_detection(image, arucoDict)

            ourMap = Map(4000, 4000)
            ourMap.draw_landmarks(landmarks)

            is_spot_free(574, 846, landmarks, 400.0)
            sleep(2)

        if command == 'RRT':
            run_RRT(image, arucoDict, True, True)

camera('RRT')

