'''
Author: Shuhao Bian
Date: 2024-10-22
'''
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

from enum import IntEnum, Enum

class AttackType(Enum):
    NONE = 0
    DOS = 1
    FDI = 2
    REPLAY = 3
    OTHER = 100

class AbstractAttacker(ABC):
    @abstractmethod
    def attack(self, **kwargs) -> None:
        pass
    