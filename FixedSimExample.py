from simModel.fixedScene.model import Model
from trafficManager.traffic_manager import TrafficManager

import logger
# get path from environment variable
import os
path = os.environ.get("LIMSIM_DIR")
DIRPREFIX = f"{path}"

# config a logger, set use_stdout=True to output log to terminal
log = logger.setup_app_level_logger(file_name=f"{DIRPREFIX}/database/app_debug.log",
                                    level="DEBUG",
                                    use_stdout=False)


carlaNetFile = f'{DIRPREFIX}/networkFiles/CarlaTown05/Town05.net.xml'
carlaRouFile = f'{DIRPREFIX}/networkFiles/CarlaTown05/Town05.rou.xml'
carlaVtypeFile = f'{DIRPREFIX}/networkFiles/CarlaTown05/carlavtypes.rou.xml'
carlaRouFile = carlaVtypeFile + ',' + carlaRouFile

if __name__ == '__main__':
    fmodel = Model(
        (300, 198),
        50,
        carlaNetFile,
        carlaRouFile,
        dataBase=f'{DIRPREFIX}/database/fixedSceneTest.db',
        SUMOGUI=0,
        simNote='local model first testing.',
    )

    fmodel.start()
    planner = TrafficManager(fmodel)

    while not fmodel.simEnd:
        fmodel.moveStep()
        if fmodel.timeStep % 5 == 0:
            roadgraph, vehicles = fmodel.exportSce()
            if roadgraph:
                trajectories = planner.plan(
                    fmodel.timeStep * 0.1, roadgraph, vehicles)
                fmodel.setTrajectories(trajectories)

        fmodel.updateVeh()

    fmodel.destroy()
