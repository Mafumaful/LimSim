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

from utils.roadgraph import AbstractLane, JunctionLane, NormalLane, RoadGraph
from utils.obstacles import Rectangle

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
        if self.agents:
            number_of_agents = len(self.agents)
        else: 
            number_of_agents = 0

        return 0.0
    
    def _calc_traffic_rule_cost(self) -> float:
        """Calculate the cost of the path based on traffic rules

        Args:
            path (List[Vehicle]): the path to evaluate

        Returns:
            float: the cost of the path
        """
        print("calc_traffic_rule_cost")
        if isinstance(self.current_lane, NormalLane):
            id = self.current_lane.id
            next_lane = self.roadgraph.get_next_lane(id)
            print(next_lane)
        else:
            print(self.current_lane)
                
        return 0.0
    
    def _calc_collision_possibliity_cost(self) -> float:
        """Calculate the cost of the path based on collision possibility

        Args:
            path (List[Vehicle]): the path to evaluate

        Returns:
            float: the cost of the path
        """
        return 0.0

    def update_detection_data(self):
        path_cost = self._calc_path_cost()
        traffic_rule_cost = self._calc_traffic_rule_cost()
        collision_possibility_cost = self._calc_collision_possibliity_cost()

        total_cost = path_cost + traffic_rule_cost + collision_possibility_cost
        print(total_cost)