import logger
from trafficManager.traffic_manager import TrafficManager
from simModel.egoTracking import interReplay
# get path from environment variable
import os
path = os.environ.get("LIMSIM_DIR")
DIRPREFIX = "."

# config a logger, set use_stdout=True to output log to terminal
log = logger.setup_app_level_logger(file_name=f"{DIRPREFIX}/database/app_debug.log",
                                    level="DEBUG",
                                    use_stdout=False)

irmodel = interReplay.InterReplayModel(
    dataBase=f"{DIRPREFIX}/database/egoTrackingTest.db", startFrame=0)
planner = TrafficManager(irmodel)

while not irmodel.tpEnd:
    irmodel.moveStep()
    if irmodel.timeStep % 5 == 0:
        roadgraph, vehicles = irmodel.exportSce()
        if roadgraph:
            trajectories = planner.plan(
                irmodel.timeStep * 0.1, roadgraph, vehicles)
        else:
            trajectories = {}
        irmodel.setTrajectories(trajectories)
    else:
        irmodel.setTrajectories({})
irmodel.gui.destroy()