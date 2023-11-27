#!/usr/bin/env python3

import numpy as np
import rclpy
from scipy.signal import convolve2d
import typing as T

from rclpy.node import Node
from asl_tb3_lib.control import BaseHeadingController
from asl_tb3_msgs.msg import TurtleBotState, TurtleBotControl
from nav_msgs.msg import OccupancyGrid,Path
from asl_tb3_lib.grids import snap_to_grid, StochOccupancyGrid2D
from std_msgs.msg import Bool

class FrontierExplorer(Node):
    def __init__(self):
        super().__init__("frontier_explorer")
        #self.create_subscription(Bool,'/nav_success',10)
        self.create_subscription(TurtleBotState,'/state',self.sub_callback_state,10)
        self.create_subscription(OccupancyGrid,'/map',self.sub_callback_map,10)
        self.cmd_nav = self.create_publisher(TurtleBotState,"/cmd_nav",10)
        timer = self.create_timer(10,self.pub_nav)


    
    #def sub_callback_nav(self, msg): #figure out when to send the new navigation command
    #     self.pub_nav()
    
    def sub_callback_state(self,message):
        self.state = TurtleBotState(x=message.x,y=message.y,theta=message.theta)

    def sub_callback_map(self,msg:OccupancyGrid):
        self.occupancy = StochOccupancyGrid2D(
            resolution=msg.info.resolution,
            size_xy=np.array([msg.info.width, msg.info.height]),
            origin_xy=np.array([msg.info.origin.position.x, msg.info.origin.position.y]),
            window_size=9,
            probs=msg.data,
        )


    def pub_nav(self):
        msg = TurtleBotState()
        goal_state = self.explore(self.occupancy)
        msg.x,msg.y,msg.theta = goal_state.x,goal_state.y,goal_state.theta
        self.cmd_nav.publish(msg)



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
        min_state = frontier_states[np.argmin(dist)]
        self.get_logger().info(f'Frontier state: {min_state[0]}, {min_state[1]}')
        return TurtleBotState(x=min_state[0],y=min_state[1],theta=0.0)


if __name__=="__main__":
    rclpy.init()
    controller = FrontierExplorer()
    rclpy.spin(controller)
    rclpy.shutdown()

'''
   def initialize_occupancy(self):
        map_grid_size = np.array((100,100)) # ????
        resolution = 0.1
        sensing_radius = 3.0
        occupancy_probs,occupancy_gt = -np.ones((map_grid_size[1],map_grid_size[0])),-np.ones((map_grid_size[1],map_grid_size[0]))
        current_state =self.state
        observed_bounds = [(current_state-sensing_radius)/resolution,(current_state+sensing_radius)/resolution]
        occupancy = StochOccupancyGrid2D(resolution=resolution,
                                         size_xy=map_grid_size,
                                         origin_xy=np.zeros((2,)),
                                         window_size=7,
                                         probs=occupancy_probs,
                                         thresh=0.5)
        for x_idx in range(int(observed_bounds[0][0]), int(observed_bounds[1][0]), 1): 
            for y_idx in range(int(observed_bounds[0][1]), int(observed_bounds[1][1]), 1): 
                if np.linalg.norm(np.array([x_idx*resolution, y_idx*resolution]) - current_state) < sensing_radius: 
                    if occupancy_gt[y_idx, x_idx] < 0.0: 
                        occupancy_probs[y_idx, x_idx] = 0.0 
                    else: 
                        occupancy_probs[y_idx, x_idx] = 0.8'''