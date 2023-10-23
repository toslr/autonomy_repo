#!/usr/bin/env python3

import typing as T
import numpy as np
import matplotlib.pyplot as plt
from asl_tb3_lib.grids import StochOccupancyGrid2D
import rclpy
import scipy

from asl_tb3_lib.navigation import BaseNavigator, TrajectoryPlan
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_lib.tf_utils import quaternion_to_yaw
from asl_tb3_msgs.msg import TurtleBotControl
from asl_tb3_msgs.msg import TurtleBotState
from asl_tb3_lib.control import BaseHeadingController

class Navigator(BaseNavigator):
    def __init__(self,kpx: float, kpy: float, kdx: float, kdy:float) -> None:
        super().__init__()
        self.kp = 2.0
        self.V_PREV_THRESH = 0.0001
        self.kpx = kpx
        self.kpy = kpy
        self.kdx = kdx
        self.kdy = kdy
    
    def reset(self):
        self.v_desired = 0.15
        self.V_prev = 0.0
        self.t_prev = 0.0
    
    def compute_heading_control(self,state: TurtleBotState,goal: TurtleBotState):
        message = TurtleBotControl()
        err = wrap_angle(goal.theta-state.theta)
        om = self.kp*err
        message.omega = om
        return message

    def compute_trajectory_tracking_control(self, state: TurtleBotState, plan: TrajectoryPlan, t: float) -> TurtleBotControl:
        dt = t-self.t_prev
        x_d = scipy.interpolate.splev(t,plan.path_x_spline,0)
        xd_d = scipy.interpolate.splev(t,plan.path_x_spline,1)
        xdd_d = scipy.interpolate.splev(t,plan.path_x_spline,2)
        y_d = scipy.interpolate.splev(t,plan.path_y_spline,0)
        yd_d = scipy.interpolate.splev(t,plan.path_y_spline,1)
        ydd_d = scipy.interpolate.splev(t,plan.path_y_spline,2)
        
        u_1 = xdd_d + self.kpx*(x_d-state.x) + self.kdx*(xd_d-self.V_prev*np.cos(state.theta))
        u_2 = ydd_d + self.kpy*(y_d-state.y) + self.kdy*(yd_d-self.V_prev*np.sin(state.theta))

        if self.V_prev < self.V_PREV_THRESH :
            self.V_prev = self.V_PREV_THRESH
    
        V = self.V_prev + dt*(np.cos(state.theta)*u_1+np.sin(state.theta)*u_2)
        om = 1/self.V_prev*(-np.sin(state.theta)*u_1+np.cos(state.theta)*u_2)

        if V<self.V_PREV_THRESH:
            V=self.V_PREV_THRESH

        return TurtleBotControl(v=V, omega=om)
    
    def compute_trajectory_plan(self, state: TurtleBotState, goal: TurtleBotState, occupancy: StochOccupancyGrid2D, resolution: float, horizon: float) -> TrajectoryPlan | None:
        astar = AStar((state.x-horizon,state.y-horizon),(state.x+horizon,state.y+horizon), np.asarray((state.x,state.y)), np.asarray((goal.x,goal.y)), occupancy, resolution)
        if not astar.solve() or len(astar.path)<4:
            return None
        self.reset()
        path = np.asarray(astar.path)
        time_list=[0]
        for i in range (1,path.shape[0]):
            time_list.append(time_list[-1]+astar.distance(path[i],path[i-1])/self.v_desired)
        ts =np.asarray(time_list)
        path_x_spline = scipy.interpolate.splrep(ts,path[:,0],k=3,s=0.05)
        path_y_spline = scipy.interpolate.splrep(ts,path[:,1],k=3,s=0.05)
        return TrajectoryPlan(path=path, 
                              path_x_spline=path_x_spline, 
                              path_y_spline=path_y_spline, 
                              duration = ts[-1])



class AStar(object):
    """Represents a motion planning problem to be solved using A*"""

    def __init__(self, statespace_lo, statespace_hi, x_init, x_goal, occupancy, resolution=1):
        self.statespace_lo = statespace_lo         # state space lower bound (e.g., [-5, -5])
        self.statespace_hi = statespace_hi         # state space upper bound (e.g., [5, 5])
        self.occupancy = occupancy                 # occupancy grid (a DetOccupancyGrid2D object)
        self.resolution = resolution               # resolution of the discretization of state space (cell/m)
        self.x_offset = x_init                     
        self.x_init = self.snap_to_grid(x_init)    # initial state
        self.x_goal = self.snap_to_grid(x_goal)    # goal state

        self.closed_set = set()    # the set containing the states that have been visited
        self.open_set = set()      # the set containing the states that are condidate for future expension

        self.est_cost_through = {}  # dictionary of the estimated cost from start to goal passing through state (often called f score)
        self.cost_to_arrive = {}    # dictionary of the cost-to-arrive at state from start (often called g score)
        self.came_from = {}         # dictionary keeping track of each state's parent to reconstruct the path

        self.open_set.add(self.x_init)
        self.cost_to_arrive[self.x_init] = 0
        self.est_cost_through[self.x_init] = self.distance(self.x_init,self.x_goal)

        self.path = None        # the final path as a list of states

    def is_free(self, x):
        """
        Checks if a give state x is free, meaning it is inside the bounds of the map and
        is not inside any obstacle.
        Inputs:
            x: state tuple
        Output:
            Boolean True/False

        """
        ########## Code starts here ##########
        return ((x[0],x[1])>self.statespace_lo and (x[0],x[1])<self.statespace_hi and self.occupancy.is_free(np.asarray(x)))
        ########## Code ends here ##########

    def distance(self, x1, x2):
        """
        Computes the Euclidean distance between two states.
        Inputs:
            x1: First state tuple
            x2: Second state tuple
        Output:
            Float Euclidean distance

        """
        ########## Code starts here ##########
        return np.sqrt((x1[0]-x2[0])**2+(x1[1]-x2[1])**2)
        ########## Code ends here ##########

    def snap_to_grid(self, x):
        """ Returns the closest point on a discrete state grid
        Input:
            x: tuple state
        Output:
            A tuple that represents the closest point to x on the discrete state grid
        """
        return (
            self.resolution * round((x[0] - self.x_offset[0]) / self.resolution) + self.x_offset[0],
            self.resolution * round((x[1] - self.x_offset[1]) / self.resolution) + self.x_offset[1],
        )

    def get_neighbors(self, x):
        """
        Gets the FREE neighbor states of a given state x. Assumes a motion model
        where we can move up, down, left, right, or along the diagonals by an
        amount equal to self.resolution.
        Input:
            x: tuple state
        Ouput:
            List of neighbors that are free, as a list of TUPLES
        """
        neighbors = []
        ########## Code starts here ##########
        new_neighbors = [self.snap_to_grid((x[0]+self.resolution,x[1])),
                         self.snap_to_grid((x[0]+self.resolution,x[1]-self.resolution)),
                         self.snap_to_grid((x[0]+self.resolution,x[1]+self.resolution)),
                         self.snap_to_grid((x[0],x[1]+self.resolution)),
                         self.snap_to_grid((x[0],x[1]-self.resolution)),
                         self.snap_to_grid((x[0]-self.resolution,x[1]+self.resolution)),
                         self.snap_to_grid((x[0]-self.resolution,x[1]-self.resolution)),
                         self.snap_to_grid((x[0]-self.resolution,x[1]))]
        for neighbor in new_neighbors:
            if self.is_free(neighbor):
                neighbors.append(neighbor)
        ########## Code ends here ##########
        return neighbors

    def find_best_est_cost_through(self):
        """
        Gets the state in open_set that has the lowest est_cost_through
        Output: A tuple, the state found in open_set that has the lowest est_cost_through
        """
        return min(self.open_set, key=lambda x: self.est_cost_through[x])

    def reconstruct_path(self):
        """
        Use the came_from map to reconstruct a path from the initial location to
        the goal location
        Output:
            A list of tuples, which is a list of the states that go from start to goal
        """
        path = [self.x_goal]
        current = path[-1]
        while current != self.x_init:
            path.append(self.came_from[current])
            current = path[-1]
        return list(reversed(path))

    def plot_path(self, fig_num=0, show_init_label=True):
        """Plots the path found in self.path and the obstacles"""
        if not self.path:
            return

        self.occupancy.plot(fig_num)

        solution_path = np.asarray(self.path)
        plt.plot(solution_path[:,0],solution_path[:,1], color="green", linewidth=2, label="A* solution path", zorder=10)
        plt.scatter([self.x_init[0], self.x_goal[0]], [self.x_init[1], self.x_goal[1]], color="green", s=30, zorder=10)
        if show_init_label:
            plt.annotate(r"$x_{init}$", np.array(self.x_init) + np.array([.2, .2]), fontsize=16)
        plt.annotate(r"$x_{goal}$", np.array(self.x_goal) + np.array([.2, .2]), fontsize=16)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03), fancybox=True, ncol=3)

        plt.axis([0, self.occupancy.width, 0, self.occupancy.height])

    def plot_tree(self, point_size=15):
        plot_line_segments([(x, self.came_from[x]) for x in self.open_set if x != self.x_init], linewidth=1, color="blue", alpha=0.2)
        plot_line_segments([(x, self.came_from[x]) for x in self.closed_set if x != self.x_init], linewidth=1, color="blue", alpha=0.2)
        px = [x[0] for x in self.open_set | self.closed_set if x != self.x_init and x != self.x_goal]
        py = [x[1] for x in self.open_set | self.closed_set if x != self.x_init and x != self.x_goal]
        plt.scatter(px, py, color="blue", s=point_size, zorder=10, alpha=0.2)

    def solve(self):
        """
        Solves the planning problem using the A* search algorithm. It places
        the solution as a list of tuples (each representing a state) that go
        from self.x_init to self.x_goal inside the variable self.path
        Input:
            None
        Output:
            Boolean, True if a solution from x_init to x_goal was found
        """
        ########## Code starts here ##########
        while len(self.open_set)>0:
            x_current = self.find_best_est_cost_through()
            if x_current == self.x_goal :
                self.path=self.reconstruct_path()
                return self.reconstruct_path()
            self.open_set.remove(x_current)
            self.closed_set.add(x_current)
            for x_neigh in self.get_neighbors(x_current):
                if x_neigh in self.closed_set:
                    continue
                tentative_cost_to_arrive = self.cost_to_arrive[x_current]+self.distance(x_current,x_neigh)
                if x_neigh not in self.open_set:
                    self.open_set.add(x_neigh)
                elif tentative_cost_to_arrive > self.cost_to_arrive[x_neigh]:
                    continue
                self.came_from[x_neigh]=x_current
                self.cost_to_arrive[x_neigh]= tentative_cost_to_arrive
                self.est_cost_through[x_neigh]=tentative_cost_to_arrive+self.distance(x_neigh,self.x_goal)
        return False
        ########## Code ends here ##########



if __name__=="__main__":
    rclpy.init()
    navigator = Navigator(2,2,2,2)
    rclpy.spin(navigator)
    rclpy.shutdown()