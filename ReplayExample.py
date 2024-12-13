from rich import print

from simModel.egoTracking.replay import ReplayModel
# get path from environment variable
import os
path = os.environ.get("LIMSIM_DIR")
DIRPREFIX = "."

dataBase = f'{DIRPREFIX}/database/egoTrackingTest.db'

rmodel = ReplayModel(dataBase=dataBase,
                     startFrame=0)

while not rmodel.tpEnd:
    rmodel.moveStep()

rmodel.gui.destroy()
