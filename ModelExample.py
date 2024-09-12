from simModel.egoTracking.model import Model
from trafficManager.traffic_manager import TrafficManager

import logger

# get path from environment variable
import os
path = os.environ.get("LIMSIM_DIR")
DIRPREFIX = f"{path}"

log = logger.setup_app_level_logger(file_name=f"{DIRPREFIX}/database/app_debug.log")


file_paths = {
    "corridor": (
        f"{DIRPREFIX}/networkFiles/corridor/corridor.net.xml",
        f"{DIRPREFIX}/networkFiles/corridor/corridor.rou.xml",
    ),
    "CarlaTown01": (
        f"{DIRPREFIX}/networkFiles/CarlaTown01/Town01.net.xml",
        f"{DIRPREFIX}/networkFiles/CarlaTown01/carlavtypes.rou.xml,{DIRPREFIX}/networkFiles/CarlaTown01/Town01.rou.xml",
    ),
    "CarlaTown05": (
        f"{DIRPREFIX}/networkFiles/CarlaTown05/Town05.net.xml",
        f"{DIRPREFIX}/networkFiles/CarlaTown05/carlavtypes.rou.xml,{DIRPREFIX}/networkFiles/CarlaTown05/Town05.rou.xml",
    ),
    "bigInter": (
        f"{DIRPREFIX}/networkFiles/bigInter/bigInter.net.xml",
        f"{DIRPREFIX}/networkFiles/bigInter/bigInter.rou.xml",
    ),
    "roundabout": (
        f"{DIRPREFIX}/networkFiles/roundabout/roundabout.net.xml",
        f"{DIRPREFIX}/networkFiles/roundabout/roundabout.rou.xml",
    ),
    "bilbao":   (
        f"{DIRPREFIX}/networkFiles/bilbao/osm.net.xml",
        f"{DIRPREFIX}/networkFiles/bilbao/osm.rou.xml",
    ),
    #######
    # Please make sure you have request the access from https://github.com/ozheng1993/UCF-SST-CitySim-Dataset and put the road network files (.net.xml) in the relevent {DIRPREFIX}/networkFiles/CitySim folder
    "freewayB": (
        f"{DIRPREFIX}/networkFiles/CitySim/freewayB/freewayB.net.xml",
        f"{DIRPREFIX}/networkFiles/CitySim/freewayB/freewayB.rou.xml",
    ),
    "Expressway_A": (
        f"{DIRPREFIX}/networkFiles/CitySim/Expressway_A/Expressway_A.net.xml",
        f"{DIRPREFIX}/networkFiles/CitySim/Expressway_A/Expressway_A.rou.xml",
    ),
    ########
}


def run_model(
    net_file,
    rou_file,
    ego_veh_id="61",
    data_base=f"{DIRPREFIX}/database/egoTrackingTest.db",
    SUMOGUI=0,
    sim_note="example simulation, LimSim-v-0.2.0.",
    carla_cosim=False,
):
    model = Model(
        ego_veh_id,
        net_file,
        rou_file,
        dataBase=data_base,
        SUMOGUI=SUMOGUI,
        simNote=sim_note,
        carla_cosim=carla_cosim,
    )
    model.start()
    planner = TrafficManager(model)

    while not model.tpEnd:
        model.moveStep()
        if model.timeStep % 5 == 0:
            roadgraph, vehicles = model.exportSce()
            if model.tpStart and roadgraph:
                trajectories = planner.plan(
                    model.timeStep * 0.1, roadgraph, vehicles
                )
                model.setTrajectories(trajectories)
            else:
                model.ego.exitControlMode()
        model.updateVeh()

    model.destroy()


if __name__ == "__main__":
    net_file, rou_file = file_paths['CarlaTown05']
    run_model(net_file, rou_file, ego_veh_id="30", carla_cosim=False)
    # run_model(net_file, rou_file, ego_veh_id="4", carla_cosim=True)
