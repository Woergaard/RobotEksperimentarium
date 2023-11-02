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

### KLASSER ###

class Landmark:
    '''
    Denne klasse indeholder landmarks.
    '''
    def __init__(self, distance, vinkel, retning, id, tvec):
        self.distance = distance
        self.vinkel = vinkel
        self.retning = retning
        self.id = id
        self.tvec = tvec
        self.x = tvec[0]
        self.z = tvec[1]
        
class Node:
    '''
    Denne klasse indeholder noder i et RRT.
    '''
    def __init__(self, x, z, parent):
        self.x = x
        self.z = z
        self.y = z
        self.path = []
        self.parent = parent
        self.theta = 0

    def setTheta(self, theta):
        self.theta = theta

class Graf:
    '''
    Denne klasse indeholder grafen/træet for en RRT.
    '''
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges

class Map:
    '''
    Denne klasse tegner et kort over robottens lokale observationer med f. eks. RRT.
    '''
    def __init__(self, xlim, zlim):
        self.landmarks = []
        self.nodes = []
        self.edges = []
        self.xlim = xlim
        self.zlim = zlim

    def draw_landmarks(self, landmarks):
        '''
        Funktionen plotter landmarks.
        '''
        self.landmarks += landmarks

        for landmark in self.landmarks:
            plt.plot(landmark.x, landmark.z, 'ro', markersize = 17)
            plt.annotate(str(landmark.id), xy=(landmark.x, landmark.z))
    
    def draw_tree(self, G):
        '''
        Funktionen plotter en graf G.
        '''
        edges = G.edges

        for edge in edges:
            plt.plot([edge[0].x, edge[1].x] , [edge[0].z, edge[1].z], 'yo-')
        
    def draw_goal(self, goal):
        '''
        Funktionen indtegner Arlos målposition.
        '''
        plt.plot(goal.x, goal.z, 'co', markersize = 17)
        plt.annotate('Mål', xy=(goal.x, goal.z))

    def draw_path(self, path):
        '''
        Funktionen plotter Arlos sti gennem et RRT.
        '''
        i = 0
        for node in path:
            #print(i)
            #print(node.x)
           # print(node.z)
           # print(node.parent)
            plt.plot(node.x, node.z, 'mo')
            i += 1

    def show_map(self, arlo_position):
        '''
        Funktionen viser kortet.
        '''
        plt.plot(arlo_position.x,arlo_position.z, 'bo')
        plt.annotate('ArloCinque', xy=(arlo_position.x, arlo_position.z))
        
        plt.ylim(-500, self.zlim)
        plt.xlim(-500, self.xlim)
        plt.xlabel('x')
        plt.ylabel('z')
        plt.show()
        #plt.savefig('rute.png')


### BRUG AF KAMERA ###

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

def drive_to_landmarks(landmarks):
    '''
    Robotten kører til det første landmark på en liste af landmarks.
    Argumenter:
        landmarks:   liste af landmarks
    '''    
    landmark = landmarks[0]

    print('Driving towards landmark ' + str(landmark.id) + ', distance = ' + str(landmark.distance))
    print('tvec = ' + str(landmark.tvec))

    sharp_turn(landmark.direction, landmark.angle)
    drive('forwards', (landmark.distance-150)/1000)

    print('Arrived at landmark!')
    arlo.stop()
    return
    
### HJÆLPEFUNKTIONER, RRT ###
def dist(node1, node2):
    '''
    Funktionen retunerer distancen mellem to noder (punkter i et koordinatsystem).
    Argumenter:
        node1:  første node
        node2:   anden node
    '''
    return np.sqrt((node1.x-node2.x)**2+(node1.z-node2.z)**2) 

def find_turn_angle(position, node):
    '''
    Funktionen returnerer retning og vinkel fra en node til en ønsket position (node).
    Argumenter:
        position:  node, der ønskes at dreje mod
        node:   noden, der drejes fra
    '''
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
    '''
    Funktionen returnerer den modsatte retning.
    '''
    if direction == 'left':
        return 'right'
    elif direction == 'right':
        return 'left'

def landmark_detection(img, arucoDict): 
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
    
    # Giver distancen til boxen
    _, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(aruco_corners, arucoMarkerLength, camera_matrix, distortion)
    
    
    lst = []
    
    # Udregner vinkel og retning til landmarket
    if ids == []: 
        print("No marker detected")
        return lst
    elif ids is not None:
        for i in range(len(ids)):
            world_angle = np.arccos(np.dot(tvecs[i]/np.linalg.norm(tvecs[i]), np.array([0, 0, 1])))
            if np.dot(tvecs[i], np.array([1, 0, 0])) < 0:
                direction = 'left'
            else:
                direction = 'right'

            lst.append(Landmark(linalg.norm(tvecs[i]), world_angle, direction, ids[i][0], tvecs[i][0]))
            #print('dist: ' + str(linalg.norm(tvecs[i])) + ', vinkel: ' + str(world_angle) + direction + ', id: ' + str(ids[i][0]) + ', tvec: ' + str(tvecs[i][0]))
        
        lst.sort(key=lambda x: x.distance, reverse=False) # sorterer listen, så det nærmeste landmark er først.

    return lst

def is_spot_free(spot, landmarks, radius):
    '''
    Funktionen retunerer en bool, der angiver, om positionen er ledig.
    Argumenter:
        spot:  position, der skal tjekkes, om er fri
        landmarks:  liste af detekterede landmarks eller forhindringer
        radius: radius af landmarksene.
    '''
    for landmark in landmarks:        
        if dist(spot, landmark) < radius:
            #print('Feltet ' + str(math.floor(spot.x)) + ', ' + str(math.floor(spot.z)) + ' er besat af ' + str(landmark.id))
            return False
    return True
    
def find_nearest_node(x_new, G):
    '''
    Funktionen retunerer den nærmeste node i et træ, G,  til en x_new.
    Argumenter:
        x_new:  punktet, som den returnerede node skal være tættest på
        G:   grafen/RRTet med noder
    '''
    distances = []
    for node in G.nodes:
        distances.append(abs(dist(x_new, node)))
    nearest_i = np.argmin(distances)
    #print('nearest i:', nearest_i)
    return G.nodes[nearest_i]

def steer(nearest_node, steering_node, stepLength):
    '''
    Funktionen returnerer en node, der er stepLength væk fra nearest_node i retning af steering_node.
    Argumenter:
        nearest_node:  den nærmeste node
        steering_node:  den node, der angiver retningen
        stepLength:   distancen mellem den returnerede node og nearest_node
    '''
    # s er antal skridt, så vi itere over denne hver gang et skridt er taget for at tjekke om der er frit. 
    steering_vec = np.array([steering_node.x, steering_node.z])
    nearest_vec = np.array([nearest_node.x, nearest_node.z])

    v = steering_vec - nearest_vec
    e = v / linalg.norm(v)
    q = nearest_vec + stepLength * e
    
    new_node = Node(q[0], q[1], nearest_node)
    return new_node

### RRT (path planning med Rapidly-exploring random trees) ###

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

def run_RRT(img, arucoDict, draw, drive_after_plan):
    '''
    Funktionen danner et RRT og kører efter det.
    Argumenter:
        img:  billedet fra kameraet
        arucoDict:   en dictonary af QR koder.
        draw: bool, der angiver, om funktionen skal tegne et kort
        drive_after_plan: bool, der angiver, om robotten skal køre efter planen
    '''
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
        ourMap.show_map((0,0))

    if drive_after_plan:
        prevnode = Node(0, 0, None)
        for node in path:
            direction, degrees = find_turn_angle(prevnode, node)
            print(direction, degrees) 
            sharp_turn(direction, degrees)
            drive('forwards', stepLength/1000)
            sharp_turn(inverse_direction(direction), degrees)
            prevnode = node
        
        print('Arrived at goal!')


### KAMERA ###  
def camera(command, show):
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

        if command == 'build_map':
            landmarks = landmark_detection(image, arucoDict)

            ourMap = Map(4000, 4000)
            ourMap.draw_landmarks(landmarks)

            is_spot_free(574, 846, landmarks, 400.0)
            sleep(2)

        elif command == 'RRT':
            run_RRT(image, arucoDict, True, True)

        elif command == 'drive_to_landmark':
            if turn_and_watch('left', image) == True: 
                arlo.stop()
                landmarks = landmark_detection(image, arucoDict)
                drive_to_landmarks(landmarks)
                time.sleep(15)
        
        elif command == 'turn_and_watch':
            turn_and_watch('right', image)


### SIR RESAMPLING ###

import numpy as np
from scipy.stats import norm

def p(x, gaussians):
    '''
    Funktionen definerer en posefordeling.
    Argumenter:
        x:  antal datapunkter
    '''
    return sum(w * norm.pdf(x, mu, sigma) for w, mu, sigma in gaussians)

def q_uniform(x):
    '''
    Funktionen definerer en uniform proposalfordeling.
    Argumenter:
        x:  antal datapunkter
    '''
    return np.random.uniform(0, 15, size=int(x))

def q_gauss(x, drawSample : bool):
    '''
    Funktionen definerer en normaltfordelt proposalfordeling.
    Argumenter:
        x:  antal datapunkter
    '''
    if drawSample: 
        return np.random.normal(5, 4, size=int(x))
    else: 
        return norm.pdf(x, 5, 4)

def sir(k, distibrution, gaussian):
    '''
    Funktionen udfører SIR-algoritmen.
    Argumenter:
        k:  antal datapunkter / x fra før 
        distribution:  fordelingen, der skal resamples fra
    '''
    # Compute weights
    if distibrution == 'uniform':
        # Generate initial samples
        samples = q_uniform(k)
        weights = p(samples) / (1/15)#q_uniform(k) #This is wrong: You should divide by the probability density of the sample. For this uniform distribution it is 1/15.
    elif distibrution == 'gauss': 
        # Generate initial samples
        samples = q_gauss(k, True)
        weights = p(samples, gaussian) / q_gauss(samples, False) #This is again wrong: Here you should divide by the probability density of the samples 
                                                       #evaluated in the density function for the  Gaussian a.k.a Normal distribution.
    # Normalize weights
    weights /= sum(weights) 

    # Resample according to weights
    resamples = np.random.choice(samples, size=k, p=weights)

    return resamples


### EGET KODE TIL SELF-LOCALIZE ###
def distance_for_particle(particle_i, landmark):
    d_i = dist(particle_i, landmark)
    return d_i

def distance_weights(d_M, d_i, sigma_d):
    '''
    Funktionen returnerer sandsynligheden for at obserrvere d_M givet d_i.
    '''
    nævner1 = 2 * np.pi * sigma_d**2
    tæller2 = (d_M - d_i)**2
    nævner2 = 2 * sigma_d**2

    return (1 / nævner1) * math.exp(-(tæller2 / nævner2))

def orientation_distribution(phi_M, sigma_theta, particle, landmark):
    theta_i, lx, ly, x_i, y_i = particle.theta, landmark.x, landmark.z, particle.x, particle.y
    d_i = np.sqrt((lx-x_i)**2+(ly-y_i)**2)
    e_l = np.array([lx-x_i, ly-y_i]).T/d_i 
    e_theta = np.array([np.cos(theta_i), np.sin(theta_i)]).T
    hat_e_theta = np.array([-np.sin(theta_i), np.cos(theta_i)]).T 
    phi_i = np.sign(np.dot(e_l,hat_e_theta))*np.arccos(np.dot(e_l, e_theta))

    p = (1/np.sqrt(2*np.pi*sigma_theta**2)) * np.exp(-((phi_M-phi_i)**2)/(2*sigma_theta**2))

    return p

def update_weights(sigma_d, sigma_theta, landmarks_lst, particles):     
    for i in range(len(particles)):
        landmark_weight = 1.0
        
        for j in range(len(landmarks_lst)):
            landmark = landmarks_lst[j]
            d_M = landmark.distance # den målte distance til landmarket
            phi_M = landmark.vinkel 
            d_j = distance_for_particle(particles[i], landmark)
            dist_weight_j = distance_weights(d_M, d_j, sigma_d)

            orientation_weight_j = orientation_distribution(phi_M, sigma_theta, particles[i], landmark)

            landmark_weight = landmark_weight * dist_weight_j * orientation_weight_j

        particles[i].setWeight(landmark_weight)

def make_intervals(particles):
    intervals = []
    lower = 0.0
    for par in particles:
        weight =  par.getWeight()
        upper = lower + weight
        intervals.append((lower, upper))
        lower = upper

    return intervals

def normalize_weights(particles):
    sum_weights = 0.0
    for par in particles:
        weight =  par.getWeight()
        sum_weights += weight
    
    weights_after = 0.0
    for par in particles:
        weight =  par.getWeight()
        par.setWeight((weight / sum_weights))
        weights_after += weight / sum_weights

def find_interval(værdi, intervals):
    for i in range(0, len(intervals)):
        if intervals[i][0] <= værdi < intervals[i][1]:
            return i
    return -1

def generate_new_particles(num_particles, particles, intervals):
    new_particles = []
    for i in range(num_particles):
        drawnNumber = random.uniform(0, 1)
        drawnIndex = find_interval(drawnNumber, intervals)
        drawnSample = particles[drawnIndex]
        newParticle = particle.Particle(x = drawnSample.x, y = drawnSample.y, theta = drawnSample.theta, weight = drawnSample.weight)
        new_particles.append(newParticle)
    
    return new_particles

def make_list_of_landmarks(objectIDs, dists, angles, landmarks):
    landmarks_lst = [] # liste af landmarks
    # List detected objects
    for i in range(len(objectIDs)):
        print("Object ID = ", objectIDs[i], ", Distance = ", dists[i], ", angle = ", angles[i])
        if objectIDs[i] in objectIDs[i+1:]:
            print('Der ses to sider af landmark ', objectIDs[i])
            same_id_indexes = [index for index, id in enumerate(objectIDs) if id == objectIDs[i]]
            index_and_dist = [(dists[index], index) for index in same_id_indexes] 
            index_and_dist.sort(key=lambda x: x[0])
            closest_index = index_and_dist[0][1]
            for index in same_id_indexes:
                objectIDs[i] = 404
        else: 
            closest_index = i

        if objectIDs[i] != 404:
            print('ID:')
            print(closest_index)
            tvectuple = landmarks[objectIDs[closest_index]]
            new_landmark = Landmark(dists[closest_index], angles[closest_index], None, objectIDs[closest_index], (0,0,0))
            new_landmark.x = tvectuple[0]
            new_landmark.z = tvectuple[1]

            landmarks_lst.append(new_landmark)

        landmarks_lst.sort(key=lambda x: x.distance, reverse=False) # sorterer efter, hvor tætte objekterne er på os

    return landmarks_lst

### UDLEVERET KODE TIL SELF-LOCALIZE ###

def bgr_colors():
    # Some color constants in BGR format
    CRED = (0, 0, 255)
    CGREEN = (0, 255, 0)
    CBLUE = (255, 0, 0)
    CCYAN = (255, 255, 0)
    CYELLOW = (0, 255, 255)
    CMAGENTA = (255, 0, 255)
    CWHITE = (255, 255, 255)
    CBLACK = (0, 0, 0)
    return CRED, CGREEN, CBLUE, CCYAN, CYELLOW, CMAGENTA, CWHITE, CBLACK

CRED, CGREEN, CBLUE, CCYAN, CYELLOW, CMAGENTA, CWHITE, CBLACK = bgr_colors()

def jet(x):
    """Colour map for drawing particles. This function determines the colour of 
    a particle from its weight."""
    r = (x >= 3.0/8.0 and x < 5.0/8.0) * (4.0 * x - 3.0/2.0) + (x >= 5.0/8.0 and x < 7.0/8.0) + (x >= 7.0/8.0) * (-4.0 * x + 9.0/2.0)
    g = (x >= 1.0/8.0 and x < 3.0/8.0) * (4.0 * x - 1.0/2.0) + (x >= 3.0/8.0 and x < 5.0/8.0) + (x >= 5.0/8.0 and x < 7.0/8.0) * (-4.0 * x + 7.0/2.0)
    b = (x < 1.0/8.0) * (4.0 * x + 1.0/2.0) + (x >= 1.0/8.0 and x < 3.0/8.0) + (x >= 3.0/8.0 and x < 5.0/8.0) * (-4.0 * x + 5.0/2.0)

    return (255.0*r, 255.0*g, 255.0*b)

def draw_world(est_pose, particles, world, landmarks, landmarkIDs, landmark_colors):
    """Visualization.
    This functions draws robots position in the world coordinate system."""

    # Fix the origin of the coordinate system
    offsetX = 100
    offsetY = 250

    # Constant needed for transforming from world coordinates to screen coordinates (flip the y-axis)
    ymax = world.shape[0]

    world[:] = CWHITE # Clear background to white

    # Find largest weight
    max_weight = 0
    for particle in particles:
        max_weight = max(max_weight, particle.getWeight())

    # Draw particles
    for particle in particles:
        x = int(particle.getX() + offsetX)
        y = ymax - (int(particle.getY() + offsetY))
        colour = jet(particle.getWeight() / max_weight)
        cv2.circle(world, (x,y), 2, colour, 2)
        b = (int(particle.getX() + 15.0*np.cos(particle.getTheta()))+offsetX, 
        ymax - (int(particle.getY() + 15.0*np.sin(particle.getTheta()))+offsetY))
        cv2.line(world, (x,y), b, colour, 2)

    # Draw landmarks
    for i in range(len(landmarkIDs)):
        ID = landmarkIDs[i]
        lm = (int(landmarks[ID][0] + offsetX), int(ymax - (landmarks[ID][1] + offsetY)))
        cv2.circle(world, lm, 5, landmark_colors[i], 2)

    # Draw estimated robot pose
    a = (int(est_pose.getX())+offsetX, ymax-(int(est_pose.getY())+offsetY))
    b = (int(est_pose.getX() + 15.0*np.cos(est_pose.getTheta()))+offsetX, 
    ymax-(int(est_pose.getY() + 15.0*np.sin(est_pose.getTheta()))+offsetY))
    cv2.circle(world, a, 5, CMAGENTA, 2)
    cv2.line(world, a, b, CMAGENTA, 2)

def initialize_particles(num_particles):
    particles = []
    for i in range(num_particles):
        # Random starting points. 
        p = particle.Particle(600.0*np.random.ranf() - 100.0, 600.0*np.random.ranf() - 250.0, np.mod(2.0*np.pi*np.random.ranf(), 2.0*np.pi), 1.0/num_particles)
        particles.append(p)

    return particles