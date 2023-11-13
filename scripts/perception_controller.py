#!/usr/bin/env python3

import numpy
import rclpy

from asl_tb3_lib.control import BaseHeadingController
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_msgs.msg import TurtleBotControl
from asl_tb3_msgs.msg import TurtleBotState
from std_msgs.msg import Bool

class PerceptionController(BaseHeadingController):
    def __init__(self):
        super().__init__("perception_controller")
        self.declare_parameter("active",True)
        self.image_detected = False
        self.create_subscription(Bool,'/detector_bool',self.sub_callback, 10)

    @property
    def active(self):
        return self.get_parameter("active").value
    
    def compute_control_with_goal(self,state:TurtleBotState,goal:TurtleBotState):
        message = TurtleBotControl()
        if not self.image_detected :
            om = 0.2
        else :
            om = 0.0
        message.omega = om
        return message
    
    def sub_callback(self,message):
        if message.data :
            self.image_detected = True
        else :
            self.image_detected = False


if __name__=="__main__":
    rclpy.init()
    controller = PerceptionController()
    rclpy.spin(controller)
    rclpy.shutdown()