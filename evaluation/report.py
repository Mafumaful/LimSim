import os
import sqlite3
import argparse
import numpy as np
from rich import print
from matplotlib import pyplot as plt
# get path from environment variable
import os
path = os.environ.get("LIMSIM_DIR")
DIRPREFIX = f"{path}"
plt.style.use('ggplot')

def create_if_not_exist(path: str):
    # Ensure the parent directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Create the file if it doesn't exist
    open(path, 'a').close()
    
    return path


class Analysis:
    def __init__(self, database: str, outputPath: str, criteria: float) -> None:
        self.database = database
        self.outputPath = outputPath
        self.figPath = os.path.join(outputPath, 'figs/')
        self.criteria = criteria

    def getData(self, sql: str) -> list[tuple]:
        conn = sqlite3.connect(self.database)
        cur = conn.cursor()
        cur.execute(sql)
        data = cur.fetchall()
        conn.close()
        return list(zip(*data))
    
    def getCollisionStages(
            self, frame: list[int], collision: list[float]
        ) -> list[list[int]]:
        stageStart = 0
        stageEnd = frame[-1]
        stages = []
        for i in range(len(frame)-1):
            if collision[i] > self.criteria and collision[i+1] < self.criteria:
                stageStart = frame[i]
            if collision[i] < self.criteria and collision[i+1] > self.criteria:
                stageEnd = frame[i]
                stages.append([stageStart, stageEnd])
        if stageEnd < stageStart:
            stages.append([stageStart, frame[-1]])

        return stages
    
    def collisionAnalysis(self):
        sql = """SELECT frame, collision from evaluationINFO;"""
        frame, collision = self.getData(sql)

        stages = self.getCollisionStages(frame, collision)

        plt.figure(figsize=(10, 6))
        plt.plot(frame, collision, color='#54a0ff')
        plt.plot(frame, [self.criteria for _ in range(len(frame))], color='#ff6b6b')
        plt.legend(['TTC', 'Criteria'])
        plt.xlabel('Frame')
        plt.ylabel('TTC (s)')
        plt.ylim((0, 21))
        collisionPath = create_if_not_exist(f"{self.figPath}/collision.svg")
        plt.savefig(collisionPath, bbox_inches='tight')
        plt.close()

        header = '# Collision reports\n\n'
        comments = '''The collision criteria for TTC is {} s, a total of {} 
        potential collisions occurred in this simulation.\n\n'''.format(
            self.criteria, len(stages)
        )
        if stages:
            table = '|Stage|Start frame|End frame|\n'
            table += '|:----:|----:|----:|\n'
            for i in range(len(stages)):
                table += '|{}|{}|{}|\n'.format(i, stages[i][0], stages[i][1])
            table += '\n'
        else:
            table = '\n'
        fig = '![img](figs/collision.svg)\n\n'
        return header + comments + table + fig

    def velocityDistributionAnalysis(self):
        sql = """SELECT speed FROM frameINFO WHERE vtag='ego' OR 'AoI';"""
        data = self.getData(sql)
        velocity = data[0]
        plt.figure(figsize=(10, 6))
        plt.hist(velocity, bins=100, density=True, color='#54a0ff')
        plt.xlabel('Velocity (m/s)')
        plt.ylabel('Density')
        figPath = create_if_not_exist(f"{self.figPath}/velocityDistribution.svg")
        plt.savefig(figPath, bbox_inches='tight')
        plt.close()

        header = '# Velocity distribution reports\n\n'
        comments = '''The velocity distribution of this simulation is shown as follow:\n\n'''
        fig = '![img](figs/velocityDistribution.svg)\n\n'
        return header + comments + fig


def createPath(outputPath: str):
    if not os.path.exists(outputPath):
        os.mkdir(outputPath)

    figPath = os.path.join(outputPath, 'figs/')
    if not os.path.exists(figPath):
        os.mkdir(figPath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Database report tool'
    )
    parser.add_argument('-d', '--database', type=str, help='Filename of the database.',
                        default=f'{DIRPREFIX}/database/egoTrackingTest.db')
    parser.add_argument(
        '-o', '--output', type=str, help='path to generate the report.',
        default=f'{DIRPREFIX}/report/'
        )
    parser.add_argument(
        '--ttc-criteria', type=float, 
        help="TTC's criteria for evaluating whether a scenario is dangerous.", 
        default=3.0
    )
    
    args = parser.parse_args()

    outputPath = args.output

    ana = Analysis(args.database, outputPath, args.ttc_criteria)

    # write the analysis comments of each section.
    outputPath = create_if_not_exist(f"{outputPath}/report.md")
    with open(outputPath, 'w') as rf:
        collsionComments = ana.collisionAnalysis()
        rf.write(collsionComments)
        velocityDistributionComments = ana.velocityDistributionAnalysis()
        rf.write(velocityDistributionComments)