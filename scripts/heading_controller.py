#!/usr/bin/env python3

import numpy
import rclpy

from asl_tb3_lib.control import BaseHeadingController
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_msgs.msg import TurtleBotControl
from asl_tb3_msgs.msg import TurtleBotState

class HeadingController(BaseHeadingController):
    def __init__(self):
        super().__init__()
        self.kp=2.0
    
    def compute_control_with_goal(self,state:TurtleBotState,goal:TurtleBotState):
        message = TurtleBotControl()
        err = wrap_angle(abs(goal.theta-state.theta))
        om = self.kp*err
        message.omega = om
        return message

if __name__=="__main__":
    rclpy.init()
    controller = HeadingController()
    rclpy.spin(controller)
    rclpy.shutdown()