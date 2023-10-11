import cv2
import particle
import camera
import numpy as np
import time
from timeit import default_timer as timer
import sys
import _utils
import math


# Flags
showGUI = True  # Whether or not to open GUI windows
onRobot = True # Whether or not we are running on the Arlo robot


def isRunningOnArlo():
    """Return True if we are running on Arlo, otherwise False.
      You can use this flag to switch the code from running on you laptop to Arlo - you need to do the programming here!
    """
    return onRobot


if isRunningOnArlo():
    # XXX: You need to change this path to point to where your robot.py file is located
    sys.path.append("../")


try:
    import robot
    onRobot = True
except ImportError:
    print("selflocalize.py: robot module not present - forcing not running on Arlo!")
    onRobot = False


# Some color constants in BGR format
CRED = (0, 0, 255)
CGREEN = (0, 255, 0)
CBLUE = (255, 0, 0)
CCYAN = (255, 255, 0)
CYELLOW = (0, 255, 255)
CMAGENTA = (255, 0, 255)
CWHITE = (255, 255, 255)
CBLACK = (0, 0, 0)

# Landmarks.
# The robot knows the position of 2 landmarks. Their coordinates are in the unit centimeters [cm].
landmarkIDs = [1, 4]
landmarks = {
    1: (0.0, 0.0),  # Coordinates for landmark 1
    4: (300.0, 0.0)  # Coordinates for landmark 2
}
landmark_colors = [CRED, CGREEN] # Colors used when drawing the landmarks


def jet(x):
    """Colour map for drawing particles. This function determines the colour of 
    a particle from its weight."""
    r = (x >= 3.0/8.0 and x < 5.0/8.0) * (4.0 * x - 3.0/2.0) + (x >= 5.0/8.0 and x < 7.0/8.0) + (x >= 7.0/8.0) * (-4.0 * x + 9.0/2.0)
    g = (x >= 1.0/8.0 and x < 3.0/8.0) * (4.0 * x - 1.0/2.0) + (x >= 3.0/8.0 and x < 5.0/8.0) + (x >= 5.0/8.0 and x < 7.0/8.0) * (-4.0 * x + 7.0/2.0)
    b = (x < 1.0/8.0) * (4.0 * x + 1.0/2.0) + (x >= 1.0/8.0 and x < 3.0/8.0) + (x >= 3.0/8.0 and x < 5.0/8.0) * (-4.0 * x + 5.0/2.0)

    return (255.0*r, 255.0*g, 255.0*b)

def draw_world(est_pose, particles, world):
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


# Main program #
try:
    if showGUI:
        # Open windows
        WIN_RF1 = "Robot view"
        cv2.namedWindow(WIN_RF1)
        cv2.moveWindow(WIN_RF1, 50, 50)

        WIN_World = "World view"
        cv2.namedWindow(WIN_World)
        cv2.moveWindow(WIN_World, 500, 50)


    # Initialize particles
    num_particles = 1000
    particles = initialize_particles(num_particles)

    est_pose = particle.estimate_pose(particles) # The estimate of the robots current pose

    # Driving parameters
    velocity = 0.0 # cm/sec
    angular_velocity = 0.0 # radians/sec

    # Initialize the robot (XXX: You do this)
    if onRobot: 
        arlo = robot.Robot()

    # Allocate space for world map
    world = np.zeros((500,500,3), dtype=np.uint8)

    # Draw map
    draw_world(est_pose, particles, world)

    print("Opening and initializing camera")
    if camera.isRunningOnArlo():
        cam = camera.Camera(0, 'arlo', useCaptureThread = True)
    else:
        cam = camera.Camera(0, 'macbookpro', useCaptureThread = True)

    Xlst = []
    iters = 0
    while True:
        
        # Move the robot according to user input (only for testing)
        action = cv2.waitKey(10)
        if action == ord('q'): # Quit
            break
    
        if not isRunningOnArlo():
            if action == ord('w'): # Forward
                velocity += 4.0
            elif action == ord('x'): # Backwards
                velocity -= 4.0
            elif action == ord('s'): # Stop
                velocity = 0.0
                angular_velocity = 0.0
            elif action == ord('a'): # Left
                angular_velocity += 0.2
            elif action == ord('d'): # Right
                angular_velocity -= 0.2

        # Use motor controls to update particles
        # XXX: Make the robot drive
        # XXX: You do this
        if onRobot: 
            _utils.drive('forwards', 2)


        # Fetch next frame
        colour = cam.get_next_frame()
        
        # Detect objects
        objectIDs, dists, angles = cam.detect_aruco_objects(colour)
        landmarks_lst = [] # liste af landmarks

        Xlst.append(particles) # således at Xlst[iter] er lig de nuværende particles

        if not isinstance(objectIDs, type(None)):
            # List detected objects
            for i in range(len(objectIDs)):
                print("Object ID = ", objectIDs[i], ", Distance = ", dists[i], ", angle = ", angles[i])
                tvectuple = landmarks[objectIDs[i]]
                new_landmark = _utils.Landmark(dists[i], angles[i], None, objectIDs[i], (0,0,0))
                new_landmark.x = tvectuple[0]
                new_landmark.z = tvectuple[0]

                landmarks_lst.append(new_landmark)
            
            landmarks_lst.sort(key=lambda x: x.distance, reverse=False) # sorterer efter, hvor tætte objekterne er på os


                # XXX: Do something for each detected object - remember, the same ID may appear several times
                
                # The handout code can also detect different landmarks in an image. To handle, this we suggest that you
                # consider each landmark as an independent observation in the observation model. What this means is
                # that the observation model should be extended with a product between the contributions from each
                # landmark,

                #lx og ly er landmarkets position 
                # sigma d er afstandsfejlen for kameraret + en lille størrelse 
                

    
            def distance_for_particle(particle_i, landmark):
                d_i = _utils.dist(particle_i, landmark)
                return d_i
        
            def distance_weights(d_M, d_i, sigma_d):
                '''
                Funktionen returnerer sandsynligheden for at obserrvere d_M givet d_i.
                '''
                nævner1 = 2 * np.pi * sigma_d**2
                tæller2 = (d_M - d_i)**2
                nævner2 = 2 * sigma_d**2

                return (1 / nævner1) * math.exp(-(tæller2 / nævner2))
            
            # Compute particle weights
            # XXX: You do this
            # Beregn ligning (2) og (3) i opgave teksten
            # kendte variable: lx, ly x(i), y(i), theta(i), d_M, phi_M

            # beregn: e_l, e_theta, d_i, \hat{e}_‡heta, p_theta, p_d
            
            def orientation_distribution(phi_M, sigma_theta, particle, landmark):
                theta_i, lx, ly, x_i, y_i = particle.theta, landmark.x, landmark.z, particle.x, particle.y
                d_i = np.sqrt((lx-x_i)**2+(ly-y_i)**2)
                e_l = np.array([lx-x_i, ly-y_i]).T/d_i 
                e_theta = np.array([np.cos(theta_i), np.sin(theta_i)]).T
                hat_e_theta = np.array([-np.sin(theta_i), np.cos(theta_i)]).T 
                phi_i = np.sign(np.dot(e_l,hat_e_theta))*np.arccos(np.dot(e_l, e_theta))

                p = (1/np.sqrt(2*np.pi*sigma_theta**2)) * np.exp(-((phi_M-phi_i)**2)/(2*sigma_theta**2))
                
                return p
            

            def update_weights(d_M, sigma_d, phi_M, sigma_theta, landmark, particles):    
                for i in range(len(particles)):
                    d_i = distance_for_particle(particles[i], landmark)
                    dist_weight_i = distance_weights(d_M, d_i, sigma_d)

                    orientation_weight_i = orientation_distribution(phi_M, sigma_theta, particles[i], landmark)
                
                    particles[i].setWeight(dist_weight_i * orientation_weight_i)
                
            landmark = landmarks_lst[0]
            d_M = landmark.distance # den målte distance til det nærmeste landmark
            phi_M = landmark.vinkel
            sigma_theta = 0.5
            sigma_d = 0.9

            update_weights(d_M, sigma_d, phi_M, sigma_theta, landmark, particles)


            # Resampling
            # XXX: You do this
            # Resample (x,y) from eq. (2) and \theta from (3)
            # Ikke kopiere pointeren til objektet medn kopierer det faktisk objekt (eller lave det som en objekt)

            # Asgers forsøg
#            particle.add_uncertainty(particles, sigma_d, sigma_theta)
            #
            

            ### Sofie¨
#            new_particles = []
#            for i in range(num_particles):
#                new_particles.append()
#            particles = 
#            gaussian = [(0.3, 2.0, 1.0), (0.4, 5.0, 2.0), (0.3, 9.0, 1.0)]

#            _utils.sir(1000, 'gaussian', gaussian)

            

            # sofies noter
            ### tilføj forskellig mængde støj afhængig af, om u fx. robotten drejer, kører ligeud eller holder stille.
            ### tilføj 2 vases: 1. den ser to sider af det samme landmark. 2. den ser mere end ét landmarks.


            # Draw detected objects
            cam.draw_aruco_objects(colour)
        else:
            # No observation - reset weights to uniform distribution
            for p in particles:
                p.setWeight(1.0/num_particles)

    
        est_pose = particle.estimate_pose(particles) # The estimate of the robots current pose

        if showGUI:
            # Draw map
            draw_world(est_pose, particles, world)

            # Show frame
            cv2.imshow(WIN_RF1, colour)

            # Show world
            cv2.imshow(WIN_World, world)

    iters += 1
  
finally: 
    # Make sure to clean up even if an exception occurred
    
    # Close all windows
    #cv2.destroyAllWindows()

    # Clean-up capture thread
    cam.terminateCaptureThread()

