#!/usr/bin/env python3

import numpy as np
import rclpy
from scipy.signal import convolve2d
import typing as T
from geometry_msgs.msg import Twist

from rclpy.node import Node
from asl_tb3_lib.control import BaseController, BaseHeadingController
from asl_tb3_msgs.msg import TurtleBotState, TurtleBotControl
from nav_msgs.msg import OccupancyGrid,Path
from asl_tb3_lib.grids import snap_to_grid, StochOccupancyGrid2D
from std_msgs.msg import Bool

class FrontierExplorer(Node):
    def __init__(self):
        super().__init__("frontier_explorer")
        self.nav_success_sub = self.create_subscription(Bool,'/nav_success',self.pub_nav, 10)
        self.state_sub  =self.create_subscription(TurtleBotState,'/state',self.sub_callback_state,10)
        self.map_sub = self.create_subscription(OccupancyGrid,'/map',self.sub_callback_map,10)
        self.cmd_nav = self.create_publisher(TurtleBotState,"/cmd_nav",10)
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.timer = self.create_timer(5,self.start_nav)
        self.create_subscription(Bool,'/detector_bool',self.stop_callback, 10)


    #def sub_callback_nav(self, msg): #figure out when to send the new navigation command
    #     self.pub_nav()
    
    def sub_callback_state(self,msg:TurtleBotState):
        self.state = TurtleBotState(x=msg.x, y=msg.y, theta=msg.theta)

    def sub_callback_map(self,msg:OccupancyGrid):
        self.occupancy = StochOccupancyGrid2D(
            resolution=msg.info.resolution,
            size_xy=np.array([msg.info.width, msg.info.height]),
            origin_xy=np.array([msg.info.origin.position.x, msg.info.origin.position.y]),
            window_size=9,
            probs=msg.data,
        )


    def start_nav(self):
        msg = TurtleBotState()
        goal_state = self.explore(self.occupancy)
        msg.x,msg.y,msg.theta = goal_state.x,goal_state.y,goal_state.theta

        # if self.count  == 0:
        self.cmd_nav.publish(msg)
        self.timer.cancel()

    def pub_nav(self, msg:Bool):
        if msg.data:
            self.get_logger().info(f"reached goal pose")

            message = TurtleBotState()
            goal_state = self.explore(self.occupancy)
            message.x,message.y,message.theta = goal_state.x,goal_state.y,goal_state.theta
            self.cmd_nav.publish(message)
        else:
            self.get_logger().info(f"replanning fails")

            message = TurtleBotState()
            goal_state = self.explore(self.occupancy)
            message.x,message.y,message.theta = goal_state.x,goal_state.y,goal_state.theta
            self.cmd_nav.publish(message)
    
            
    def stop_callback(self,message):
        # if stop sign is detected, send 0 command to stop robot
        if message.data:
            print("stop sign detected")
            self.cmd_nav.publish(TurtleBotState(x=self.state.x,y=self.state.y,theta=0.0))



    def explore(self,occupancy):
        current_state = np.array([self.state.x,self.state.y])
        window_size = 13 
        unknown_mask = (occupancy.probs == -1)
        occupied_mask = (occupancy.probs >= 0.5)
        unoccupied_mask = (occupancy.probs <= 0.5) & (occupancy.probs > -1)
        window = np.ones((window_size, window_size))/(window_size * window_size)

        unknown_result = convolve2d(unknown_mask, window, mode = 'same')
        occupied_result = convolve2d(occupied_mask, window, mode = 'same')
        unoccupied_result = convolve2d(unoccupied_mask, window, mode = 'same')

        frontier_states_list = []

        for i in range(occupancy.probs.shape[0]):
            for j in range(occupancy.probs.shape[1]):

                if (unknown_result[i][j] >= 0.2) and (occupied_result[i][j] == 0) and\
                    (unoccupied_result[i][j] >= 0.3) and occupancy.is_free(np.array([j,i])):

                    state = occupancy.grid2state(np.array([j,i]))
                    frontier_states_list.append(state)


        frontier_states = np.array(frontier_states_list)
        dist = [np.linalg.norm(state-current_state) for state in frontier_states]
        if len(dist)==0:
            print("empty list")
            return TurtleBotState(x=self.state.x,y=self.state.y,theta=0.0)

        min_state = np.array(frontier_states[np.argmin(dist)])
        self.get_logger().info(f'Frontier state: {min_state[0]}, {min_state[1]}')
        return TurtleBotState(x=min_state[0],y=min_state[1],theta=0.0)


if __name__=="__main__":
    rclpy.init()
    controller = FrontierExplorer()
    rclpy.spin(controller)
    rclpy.shutdown()