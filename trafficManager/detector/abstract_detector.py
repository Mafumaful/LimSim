'''
Author: Shuhao Bian
Date: 2024-10
Description: 
Copyright (c) 2024 by AIS, All Rights Reserved. 
'''
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

class AbstractDetector(ABC):
    @abstractmethod
    def update_data(self, **kwargs) -> None:
        pass

