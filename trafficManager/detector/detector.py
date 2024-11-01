'''
Author: Shuhao Bian
Date: 2024-11
Description: 

Copyright (c) 2024 by AIS, All Rights Reserved. 
'''

import numpy as np
from detector.abstract_detector import AbstractDetector

class mDetector(AbstractDetector):
    def detect(self, **kwargs):
        return super().detect(**kwargs)