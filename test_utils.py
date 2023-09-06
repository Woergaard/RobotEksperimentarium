import _utils
import robot 

### DEFINEREDE PARAMETRE ###
arlo = robot.Robot()

rightWheelFactor = 1.0
leftWheelFactor = 1.06225
stanardSpeed = 50.0

### KØRSEL ####
_utils.drive('forwards', 1.5)

_utils.drive('backwards', 0.75)

_utils.sharp_turn('right', 30)

_utils.sharp_turn('left', 90)

_utils.move_in_figure_eight(1)

_utils.move_around_own_axis(1)

arlo.stop()
