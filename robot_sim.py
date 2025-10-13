import os, time, math
import pybullet as p
import pybullet_data

# URDF path
URDF = "humanoid.urdf"
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
urdf_path = os.path.join(PROJECT_DIR, URDF)

# Connect to PyBullet GUI
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# Add ground
p.loadURDF("plane.urdf")

# Load robot
robot = p.loadURDF(urdf_path, [0, 0, 0.9], useFixedBase=False)

# Find actuated joints
def actuated_joints(robot_id):
    ids = []
    names = []
    for j in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, j)
        if info[2] in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
            ids.append(j)
            names.append(info[1].decode())
    return ids, names

act, names = actuated_joints(robot)
print("Actuated joints:", list(zip(act, names)))

# Reset to initial bent-leg posture
init_pose = {
    "left_hip": 0.2,
    "left_knee": -0.5,
    "right_hip": -0.2,
    "right_knee": -0.5
}
for j_idx, j_name in zip(act, names):
    if j_name in init_pose:
        p.resetJointState(robot, j_idx, init_pose[j_name])

# Simulation loop
SIM_STEP = 1. / 240.
try:
    t0 = time.time()
    while True:
        t = time.time() - t0

        # Simple sinusoidal gait
        left_hip = 0.3 * math.sin(2 * math.pi * 0.5 * t)
        left_knee = -0.5 + 0.2 * math.sin(2 * math.pi * 0.5 * t + math.pi / 2)
        right_hip = -0.3 * math.sin(2 * math.pi * 0.5 * t)
        right_knee = -0.5 + 0.2 * math.sin(2 * math.pi * 0.5 * t + math.pi / 2)

        desired = [left_hip, left_knee, right_hip, right_knee]

        # Apply motor control
        for j_idx, q_des in zip(act, desired):
            p.setJointMotorControl2(
                robot, j_idx,
                p.POSITION_CONTROL,
                targetPosition=q_des,
                force=40
            )

        p.stepSimulation()
        time.sleep(SIM_STEP)

except KeyboardInterrupt:
    p.disconnect()
