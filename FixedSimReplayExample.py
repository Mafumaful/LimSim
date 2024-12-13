from simModel.fixedScene.replay import ReplayModel
import os

# get path from environment variable
path = os.environ.get("LIMSIM_DIR")
DIRPREFIX = "."

dataBase = f'{DIRPREFIX}/database/fixedSceneTest.db'

frmodel = ReplayModel(dataBase)

while not frmodel.tpEnd:
    frmodel.moveStep()

frmodel.gui.destroy()
