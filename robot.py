import pybullet as p, pybullet_data, os, time

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-9.8)

p.loadURDF("plane.urdf")

urdf_path = os.path.join(os.getcwd(), "humanoid.urdf")
robot = p.loadURDF(urdf_path, [0,0,1.0])

print("Number of joints:", p.getNumJoints(robot))
for j in range(p.getNumJoints(robot)):
    print(j, p.getJointInfo(robot,j)[1])
