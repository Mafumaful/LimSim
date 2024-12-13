from simModel.fixedScene.replay import ReplayModel

# get path from environment variable
import os
path = os.environ.get("LIMSIM_DIR")
DIRPREFIX = "."

dataBase = f'{DIRPREFIX}/database/fixedSceneTest.db'

frmodel = ReplayModel(dataBase)

while not frmodel.tpEnd:
    frmodel.moveStep()

frmodel.gui.destroy()
