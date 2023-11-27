#!/usr/bin/env python3

import numpy as np
import rclpy
from scipy.signal import convolve2d

from rclpy import Node
from asl_tb3_lib.control import BaseHeadingController
from asl_tb3_msgs.msg import TurtleBotState, TurtleBotControl
from nav_msgs.msg import OccupancyGrid,Path
from asl_tb3_lib.grids import snap_to_grid, StochOccupancyGrid2D
from std_msgs.msg import Bool

class FrontierExplorer(Node):
    def __init__(self):
        super().__init__("frontier_explorer")
        self.declare_parameter("occupancy",self.initialize_occupancy()) #None ?
    
        self.create_subscription(Bool,'/nav_success',self.sub_callback_nav,10)
        self.create_subscription(TurtleBotState,'/state',self.sub_callback_state,10)
        self.create_subscription(OccupancyGrid,'/map',self.sub_callback_map,10)

    @property
    def occupancy(self):
        return self.get_parameter("occupancy").value
    
    def sub_callback_nav(self,message): #figure out when to send the new navigation command
        if message.data :
            #robot reached the commanded nav pose
            pass
        else :
            #planning or replanning failed
            pass
    
    def sub_callback_state(self,message):
        self.state = message.data

    def sub_callback_map(self,message):
        self.occupancy = message.data

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
                        occupancy_probs[y_idx, x_idx] = 0.8

    def explore(self,occupancy:OccupancyGrid): 
        current_state = self.state
        window_size = 13   
        unknown_mask = occupancy.probs < 0
        occupied_mask = occupancy.probs >= 0.5
        unoccupied_mask = (occupancy.probs > -0.1) & (occupancy.probs < 0.5)
        kernel = np.ones((window_size,window_size))
        unknown_convolved = convolve2d(unknown_mask,kernel).astype(int)
        occupied_convolved = convolve2d(occupied_mask,kernel).astype(int)
        unoccupied_convolved = convolve2d(unoccupied_mask,kernel).astype(int)
        frontier_states = np.flip(occupancy.grid2state(np.argwhere((occupied_convolved==0) & (unknown_convolved>=0.2*window_size*window_size) & (unoccupied_convolved>=0.3*window_size*window_size))),axis=1)
        distances = [np.sqrt((current_state[0]-state[0])**2+(current_state[1]-state[1])**2) for state in frontier_states]
        return frontier_states


if __name__=="__main__":
    rclpy.init()
    controller = FrontierExplorer()
    rclpy.spin(controller)
    rclpy.shutdown()