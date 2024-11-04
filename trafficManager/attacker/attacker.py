from attacker.abstract_attacker import AbstractAttacker, AttackType

import time

from common.observation import Observation
from common.vehicle import Behaviour, Vehicle
from decision_maker.abstract_decision_maker import EgoDecision, MultiDecision
from trafficManager.planner.abstract_planner import AbstractEgoPlanner
from predictor.abstract_predictor import Prediction

import logger
import trafficManager.planner.trajectory_generator as traj_generator
from utils.obstacles import DynamicObstacle, ObsType, Rectangle
from utils.roadgraph import JunctionLane, NormalLane, RoadGraph
from utils.trajectory import State, Trajectory

logging = logger.get_logger(__name__)

class mAttacker(AbstractEgoPlanner):
    def init(self):
        self.count = 0
        self.attack_location = 100
        self.current_state = "ATK_ON_HARDBRAKE"
        
    def attack(self, 
               attack_type: AttackType = AttackType.DOS):
        self.attack_type = attack_type
        
    def plan(self,
             ego_veh: Vehicle,
             observation: Observation,
             roadgraph: RoadGraph,
             prediction: Prediction,
             T,
             config,
             attack_type: str="ATK_ON_HARDBRAKE") -> Trajectory:

        vehicle_id = ego_veh.id
        start = time.time()
        current_lane = roadgraph.get_lane_by_id(ego_veh.lane_id)

        obs_list = []
        # Process static obstacle
        for obs in observation.obstacles:
            obs_list.append(obs)

        # Process dynamic_obstacles
        for predict_veh, prediction in prediction.results.items():
            if predict_veh.id == vehicle_id:
                continue

            shape = Rectangle(predict_veh.length, predict_veh.width)
            current_state = State(x=prediction[0].x,
                                  y=prediction[0].y,
                                  s=prediction[0].s,
                                  d=prediction[0].d,
                                  yaw=prediction[0].yaw,
                                  vel=prediction[0].vel)
            dynamic_obs = DynamicObstacle(obstacle_id=predict_veh.id,
                                          shape=shape,
                                          obstacle_type=ObsType.CAR,
                                          current_state=current_state,
                                          lane_id=predict_veh.lane_id)
            for i in range(1, len(prediction)):
                state = State(x=prediction[i].x,
                              y=prediction[i].y,
                              s=prediction[i].s,
                              d=prediction[i].d,
                              yaw=prediction[i].yaw,
                              vel=prediction[i].vel)
                dynamic_obs.future_trajectory.states.append(state)

            obs_list.append(dynamic_obs)

        """
        Predict for current vehicle
        """
        next_lane = roadgraph.get_available_next_lane(
            current_lane.id, ego_veh.available_lanes)
        lanes = [current_lane, next_lane] if next_lane != None else [
            current_lane]

        if self.count == 0:
            self.attack_location = ego_veh.current_state.s
            self.current_state = attack_type
        self.count += 1
        
        if self.current_state == "ATK_ON_HARDBRAKE":
            path = traj_generator.stop_trajectory_generator_atk(
                ego_veh, lanes, obs_list, roadgraph, config, T, self.attack_location
            )
        elif self.current_state == "ATK_ON_FULLTHROTTLE":
            path = traj_generator.lanekeeping_trajectory_generator_atk(
                ego_veh, lanes, obs_list, config, T
            )
        else:
            logging.error(
                "Vehicle {} has unknown behaviour {}".format(
                    ego_veh.id, ego_veh.behaviour)
            )
            print(f"{attack_type} is not supported")
    
        logging.debug(
            "Vehicle {} Total planning time: {}".format(
                ego_veh.id, time.time() - start)
        )
        

        return path