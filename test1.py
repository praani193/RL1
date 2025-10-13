import pybullet as p
import pybullet_data
import time, os

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-9.81)

p.loadURDF("plane.urdf")

urdf_path = os.path.join(os.path.dirname(__file__), "humanoid.urdf")
robot = p.loadURDF(urdf_path, [0,0,1.0])

print("\nJoints in humanoid:")
for j in range(p.getNumJoints(robot)):
    print(j, p.getJointInfo(robot, j)[1])

while True:
    p.stepSimulation()
    time.sleep(1/240)
