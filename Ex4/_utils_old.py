
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