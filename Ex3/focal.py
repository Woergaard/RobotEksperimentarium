import numpy as np
import cv2 # Import the OpenCV library

def height_of_QR_in_image(img):
    '''
    Funktionen returnerer højden af en box på et billede i pixels.n
    Argumenter:
        image:  det givne billede
    '''
    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    
    aruco_corners, _, _ = cv2.aruco.detectMarkers(img, arucoDict) 
    print(aruco_corners)


    # If at least one marker is detected
    if len(aruco_corners) > 0:
        for corner in aruco_corners:
            # The corners are ordered as top-left, top-right, bottom-right, bottom-left
            top_left = corner[0][0]
            bottom_left = corner[0][3]
            
            # Compute the height in pixels
            height = bottom_left[1] - top_left[1]
            print("Height of the ArUco marker:", height, "pixels")



def focal_length(xlst, X, Zlst):
    '''
    Funktionen returnerer højden af en box på et billede i pixels.
    Argumenter:
        x = height of QR code in image
        X = height of QR code in image irl
        Z = dist
    '''
    flst = []
    for i in range(len(xlst)):
        flst.append(xlst[i] * (Zlst[i]/X)) 

    fmean = np.mean(flst)
    fstd = np.std(flst)
    return fmean, fstd 


# x = height of QR code in image
# X = height of QR code in image irl
# Z = dist
# f = focal length

X = 145.0 # height of QR code IRL
Zlst = [600.0, 700.0, 800.0, 900.0, 1000.0, 1100.0, 1200.0, 1300.0, 1400.0, 1500.0, 1600.0, 1700.0, 1800.0, 1900.0, 2000.0] #length of distances
xlst = [428.0, 365.0, 314.0, 281.0, 252.0, 230.0, 212.0, 195.0, 181.0, 168.0, 157.0, 149.0, 140.0, 132.0, 126.0] # height of QR code in image

print(focal_length(xlst, X, Zlst))

