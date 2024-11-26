'''
Author: Shuhao Bian
Date: 2024-11
Description: 

Copyright (c) 2024 by AIS, All Rights Reserved. 
'''

from detector.abstract_detector import AbstractDetector

import numpy as np
from typing import List
from evaluation.math_utils import normalize
from simModel.common.carFactory import Vehicle
from rich import print

from utils.roadgraph import AbstractLane, JunctionLane, NormalLane, RoadGraph
from utils.obstacles import Rectangle

# this is used for check collision
from shapely.geometry import LineString

def print_cost(cost: float = 0):
    if cost == 0:
        print(f"cost: [blue bold]{cost}[/blue bold]")
    elif cost <= 1:
        print(f"cost: [yellow bold]{cost}[/yellow bold]")
    elif cost <= 2:
        print(f"cost: [orange1 bold]{cost}[/orange1 bold]")
    else:
        print(f"cost: [red1 bold]{cost}[/red1 bold]")

def ConstantV(veh: Vehicle, dt: float = 0.1, predict_step: int = 100) -> list:
    """
    Generate a constant velocity trajectory.

    Args:
        veh (Vehicle): The target vehicle.
        dt (float, optional): The time interval. Defaults to 0.1.

    Returns:
        list: The generated trajectory [(x,y), (x,y) ... ].
    """
    
    traj = []

    last_v = veh.speedQ[-1]
    last_yaw = veh.yawQ[-1]
    last_x = veh.xQ[-1]
    last_y = veh.yQ[-1]
    traj.append((last_x, last_y))

    for i in range(1, predict_step):
        last_x += last_v * np.cos(last_yaw) * dt 
        last_y += last_v * np.sin(last_yaw) * dt 
        traj.append((last_x, last_y))

    return traj

def ConstantVConstantT(veh: Vehicle, dt: float = 0.1, predict_step: int = 100) -> list:
    """
    Generate a constant velocity trajectory and constant turning rate trajectory.

    Args:
        veh (Vehicle): The target vehicle.
        dt (float, optional): The time interval. Defaults to 0.1.

    Returns:
        list: The generated trajectory [(x,y), (x,y) ... ].
    """
    
    traj = []

    last_v = veh.speedQ[-1]
    last_yaw = veh.yawQ[-1]
    last_x = veh.xQ[-1]
    last_y = veh.yQ[-1]
    turning_rate = 0
    if len(veh.yawQ) > 3:
        yaw_diff = ((veh.yawQ[-1] - veh.yawQ[-2]) + (veh.yawQ[-2] - veh.yawQ[-3]))/2
        turning_rate = yaw_diff / dt

    traj.append((last_x, last_y))

    for i in range(1, predict_step):
        last_x += last_v * np.cos(last_yaw) * dt 
        last_y += last_v * np.sin(last_yaw) * dt 
        last_yaw += turning_rate * dt
        traj.append((last_x, last_y))

    return traj

class mDetector(AbstractDetector):
    def __init__(self, dt: float = 0.1) -> None:
        self.dt: float = dt
        self.ego: Vehicle = None
        self.current_lane: AbstractLane = None
        self.agents: List[Vehicle] = None
        self.ref_yaw: float = 0.0
        self.result: np.ndarray = None
        self.new_yaw: float = 0.0
        
        # this is for detecor
        self.threshold = 0.5
        
    def update_data(self, ego: Vehicle, current_lane: AbstractLane,
                    agents: List[Vehicle], roadgraph: RoadGraph):
        self.ego = ego
        self.current_lane = current_lane
        self.agents = agents
        self.roadgraph = roadgraph

    def _calc_path_cost(self) -> float:
        """Calculate the cost of the path

        Args:
            path (List[Vehicle]): the path to evaluate

        Returns:
            float: the cost of the path
        """
        velocity_queue = self.ego.speedQ
        acceleration_queue = self.ego.accelQ
        
        brake_threshold = 0.5
        
        if len(velocity_queue) > 3:
            # check if the vehicle is braking
            if velocity_queue[-1] < velocity_queue[-2] and velocity_queue[-2] < velocity_queue[-3]:
                if acceleration_queue[-1] < -brake_threshold:
                    return 1.0
        else:
            return 0.0
        
        return 0.0
    
    def _calc_traffic_rule_cost(self) -> float:
        """Calculate the cost of the path based on traffic rules

        Args:
            path (List[Vehicle]): the path to evaluate

        Returns:
            float: the cost of the path
        """
        if isinstance(self.current_lane, JunctionLane):
            if self.current_lane.currTlState == 'r':
                return 2.0
            elif self.current_lane.currTlState == 'y':
                return 1.0
                
        return 0.0
    
    def _calc_collision_possibliity_cost(self) -> float:
        """Calculate the cost of the path based on collision possibility

        Args:
            path (List[Vehicle]): the path to evaluate

        Returns:
            float: the cost of the path
        """
        if self.agents:
            # ego_traj = ConstantV(self.ego, self.dt)
            # trajs = [ConstantV(agent, self.dt) for agent in self.agents]
            ego_traj = ConstantVConstantT(self.ego, self.dt)
            trajs = [ConstantVConstantT(agent, self.dt) for agent in self.agents]
            
            # check collision
            for traj in trajs:
                if LineString(ego_traj).intersects(LineString(traj)):
                    return 1.0
                else:
                    return 0.0
        else: 
            return 0.0
            
        return 0.0

    def update_detection_data(self):
        path_cost = self._calc_path_cost()
        traffic_rule_cost = self._calc_traffic_rule_cost()
        collision_possibility_cost = self._calc_collision_possibliity_cost()

        total_cost = path_cost + traffic_rule_cost + collision_possibility_cost
        print_cost(total_cost)
        
