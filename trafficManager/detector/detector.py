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

from utils.roadgraph import AbstractLane, NormalLane
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
                    agents: List[Vehicle]):
        self.ego = ego
        self.current_lane = current_lane
        self.agents = agents

    def detect(self, **kwargs):
        return super().detect(**kwargs)