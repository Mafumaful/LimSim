'''
Author: Shuhao Bian
Date: 2024-10
Description: 
Copyright (c) 2024 by AIS, All Rights Reserved. 
'''
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List

from common.observation import Observation
from common.vehicle import Vehicle

from utils.roadgraph import RoadGraph
from utils.trajectory import State

class AbstractDetector(ABC):
    @abstractmethod
    def detect(self, **kwargs) -> None:
        pass

