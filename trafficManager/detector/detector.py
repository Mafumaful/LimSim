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
from common.vehicle import Vehicle
from common.observation import Observation
from rich import print

from utils.roadgraph import AbstractLane, JunctionLane, NormalLane, RoadGraph
from utils.obstacles import Rectangle

# this is used for check collision
from shapely.geometry import LineString

import sqlite3
import os
import threading
from queue import Queue
import json
import time

t = time.strftime("%Y%m%d-%H%M%S")
PATH = f"/Users/miakho/Code/LimSim/database/detector_{t}.db"

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

def ConstantVConstantT(veh: Vehicle, dt: float = 0.1, predict_step: int = 80) -> list:
    """
    Generate a constant velocity trajectory and constant turning rate trajectory.

    Args:
        veh (Vehicle): The target vehicle.
        dt (float, optional): The time interval. Defaults to 0.1.
        predict_step (int, optional): Number of prediction steps. Defaults to 100.

    Returns:
        list: The generated trajectory [(x,y), (x,y), ...].
    """
    
    if dt <= 0:
        raise ValueError("Time interval (dt) must be greater than zero.")
    
    traj = []

    # Extract the last known state of the vehicle
    last_v = veh["speedQ"][-1]
    last_yaw = veh["yawQ"][-1]
    last_x = veh["xQ"][-1]
    last_y = veh["yQ"][-1]
    turning_rate = 0  # Default turning rate

    # Calculate turning rate if enough yaw data is available
    if len(veh["yawQ"]) > 3:
        yaw_diff = ((veh["yawQ"][-1] - veh["yawQ"][-2]) + (veh["yawQ"][-2] - veh["yawQ"][-3])) / 2
        turning_rate = yaw_diff / dt

    # Add initial position to the trajectory
    traj.append((last_x, last_y))

    # Iterate to predict future positions
    for i in range(1, predict_step):
        # Straight-line motion if turning rate is zero
        if np.isclose(turning_rate, 0):
            last_x += last_v * np.cos(last_yaw) * dt
            last_y += last_v * np.sin(last_yaw) * dt
        else:
            # Constant turning rate motion
            last_x += last_v * np.cos(last_yaw) * dt
            last_y += last_v * np.sin(last_yaw) * dt
            last_yaw += turning_rate * dt
        
        # Normalize yaw to stay within [-π, π]
        last_yaw = (last_yaw + np.pi) % (2 * np.pi) - np.pi

        # Append the predicted position to the trajectory
        traj.append((last_x, last_y))

    return traj

class mDetector(AbstractDetector):
    def __init__(self, dt: float = 0.1) -> None:
        self.dt: float = dt
        self.timeStep: int = 0
        
        # this is for detecor
        self.threshold = 0.5
        
        self.createDatabase()
        self.dataQueue = Queue()
        self.create_timer()
        
    def update_data(self, vehicles_info: dict, roadgraph: RoadGraph, timeStep: int, ego_id: int, AttackType: str = None, agents: List[Vehicle] = None):
        self.vehicles_info = vehicles_info
        self.roadgraph = roadgraph
        self.timeStep = timeStep
        self.ego_id = ego_id
        self.attack_type = AttackType
        
        self.ego = None
        if "egoCar" in vehicles_info:
            self.ego = vehicles_info["egoCar"]
            
        # update attack type
        self.dataQueue.put(('attack_stats', (self.timeStep, self.attack_type)))

    def _calc_path_cost(self) -> float:
        """Calculate the cost of the path

        Args:
            path (List[Vehicle]): the path to evaluate

        Returns:
            float: the cost of the path
        """
        # if no history track, return 0
        acceleration_queue = self.ego["accelQ"]
        brake_threshold = 0.6
        
        if acceleration_queue[-1] < -brake_threshold:
            return abs(acceleration_queue[-1])
        else:
            return 0.0
    
    def _calc_traffic_rule_cost(self) -> float:
        """Calculate the cost of the path based on traffic rules

        Args:
            path (List[Vehicle]): the path to evaluate

        Returns:
            float: the cost of the path
        """
        self.current_lane = self.roadgraph.get_lane_by_id(self.ego["laneIDQ"][-1])
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
        
        traffic_light_state = "NA"
        current_road_type = "NA"
        traffic_light_time = -1
        
        
        if isinstance(self.current_lane, JunctionLane):
            current_road_type = "junction"
            traffic_light_time = self.current_lane.switchTime
            traffic_light_state = self.current_lane.currTlState
        elif isinstance(self.current_lane, NormalLane):
            current_road_type = "normal"
            next_lane = self.roadgraph.get_next_lane(self.current_lane.id)
            if next_lane:
                traffic_light_time = next_lane.switchTime
                traffic_light_state = next_lane.currTlState
        
        self.dataQueue.put(('trajectory', (self.timeStep, self.ego["id"], self.ego["xQ"][-1], self.ego["yQ"][-1], self.ego["speedQ"][-1], self.ego["accelQ"][-1], traffic_light_time, traffic_light_state, current_road_type)))
        ego_traj = ConstantVConstantT(self.ego, self.dt)
        
        agents = self.vehicles_info["carInAoI"]
        if agents:
            # ego_traj = ConstantV(self.ego, self.dt)
            # trajs = [ConstantV(agent, self.dt) for agent in agents]
            trajs = [ConstantVConstantT(agent, self.dt) for agent in agents]
            
            for agent in agents:
                self.dataQueue.put(('trajectory', (self.timeStep, agent["id"], agent["xQ"][-1], agent["yQ"][-1], agent["speedQ"][-1], agent["accelQ"][-1], -1, "NA", "NA")))

            # check collision
            for traj in trajs:
                if LineString(ego_traj).intersects(LineString(traj)):
                    return 1.0
        else: 
            return 0.0
            
        return 0.0

    def update_detection_data(self):
        traffic_rule_cost = self._calc_traffic_rule_cost()
        path_cost = self._calc_path_cost()
        collision_possibility_cost = self._calc_collision_possibliity_cost()
        
        lane_id = self.current_lane.id
        total_cost = path_cost + traffic_rule_cost + collision_possibility_cost
        self.dataQueue.put(('cost_data', (self.timeStep, lane_id, path_cost, traffic_rule_cost, collision_possibility_cost, total_cost)))
        
        
    def create_timer(self):
        t = threading.Timer(1, self.store_data)
        t.daemon = True
        t.start()
        
    def createDatabase(self):
        if os.path.exists(PATH):
            os.remove(PATH)
        conn = sqlite3.connect(PATH)
        cur = conn.cursor()
        cur.execute('''CREATE TABLE IF NOT EXISTS cost_data
                    (frame INT PRIMARY KEY,
                    lane_id TEXT,
                    path_cost FLOAT,
                    traffic_rule_cost FLOAT,
                    collision_possibility_cost FLOAT,
                    total_cost FLOAT)''')
        
        cur.execute('''CREATE TABLE IF NOT EXISTS trajectory
                    (frame INT,
                    vehicle_id TEXT,
                    x FLOAT,
                    y FLOAT,
                    vel FLOAT,
                    accel FLOAT,
                    traffic_light_time FLOAT,
                    traffic_light_state TEXT,
                    road_type TEXT,
                    PRIMARY KEY (frame, vehicle_id))''')
        
        cur.execute('''CREATE TABLE IF NOT EXISTS attack_stats
                    (frame INT PRIMARY KEY,
                    attack_type TEXT)''')
        
        conn.commit()
        cur.close()
        conn.close()
        
    def store_data(self):
        cnt = 0
        conn = sqlite3.connect(PATH, check_same_thread=False)
        cur = conn.cursor()
        while cnt < 1000 and not self.dataQueue.empty():
            tableName, data = self.dataQueue.get()
            sql = 'INSERT INTO %s VALUES' % tableName + \
                '(' + '?,'*(len(data)-1) + '?)'
            try:
                cur.execute(sql, data)
            except sqlite3.IntegrityError as e:
                print(e)
            cnt += 1
        
        conn.commit()
        cur.close()
        conn.close()
        
        self.create_timer()