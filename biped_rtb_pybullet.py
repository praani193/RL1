"""
biped_rtb_pybullet.py
- loads simple_biped.urdf using RTB's PyBullet backend
- runs a simple open-loop sinusoidal gait on 4 actuated joints
"""

import os, time, math
import numpy as np
import roboticstoolbox as rtb
from roboticstoolbox.backends.PyBullet import PyBullet
import pybullet as p

# ---------- configuration ----------
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
URDF_PATH = os.path.join(PROJECT_DIR, "simple_biped.urdf")
SIM_STEP = 1.0 / 240.0
MAX_TORQUE = 40.0
# gait parameters
HIP_AMP = 0.8     # radians
KNEE_AMP = 0.7    # radians
HIP_FREQ = 1.0    # Hz
KNEE_FREQ = 1.0   # Hz (can be different)
# ---------- end configuration ----------

def find_actuated_joints(pb_robot_id):
    """Return indices of revolute joints that are actuated (non-fixed)"""
    actuated = []
    for j in range(p.getNumJoints(pb_robot_id)):
        info = p.getJointInfo(pb_robot_id, j)
        jtype = info[2]
        if jtype == p.JOINT_REVOLUTE or jtype == p.JOINT_PRISMATIC:
            actuated.append(j)
    return actuated

def main():
    # Launch PyBullet (via RTB backend)
    backend = PyBullet()
    backend.launch(gui=True)      # display GUI
    world = backend.world         # pybullet world handle

    # Add plane (PyBullet built-in)
    p.setAdditionalSearchPath(rtb.robotpy.__path__[0])  # ensure search path; fallback below if not needed
    plane = p.loadURDF("plane.urdf")

    # Load URDF into the PyBullet world (RTB wrapper will also track it)
    pb_robot_id = backend.add_urdf(URDF_PATH, base_pos=[0,0,0.9], use_fixed_base=False)
    time.sleep(0.05)

    # Detect actuated joint indices in PyBullet's numbering
    actuated = find_actuated_joints(pb_robot_id)
    print("Actuated joints (pybullet indices):", actuated)
    n_act = len(actuated)
    if n_act == 0:
        print("No actuated joints found — check URDF.")
        return

    # Optional: print joint names
    for j in actuated:
        info = p.getJointInfo(pb_robot_id, j)
        print(j, info[1].decode('utf-8'), "type", info[2])

    t0 = time.time()
    step = 0
    try:
        while True:
            t = time.time() - t0
            # construct desired torques for 4 joints: [L_hip, L_knee, R_hip, R_knee]
            # use simple sinusoids with phase shift between left/right
            hip_phase = 0.0
            knee_phase = math.pi/2
            left_hip = HIP_AMP * math.sin(2*math.pi*HIP_FREQ * t + hip_phase)
            right_hip = HIP_AMP * math.sin(2*math.pi*HIP_FREQ * t + hip_phase + math.pi)  # opposite phase for walking
            left_knee = KNEE_AMP * math.sin(2*math.pi*KNEE_FREQ * t + knee_phase)
            right_knee = KNEE_AMP * math.sin(2*math.pi*KNEE_FREQ * t + knee_phase + math.pi)

            # pack into torque commands (scale small to torques)
            # here we simply use PD-like torque proportional to desired angle (no derivative) for simplicity
            desired_angles = [left_hip, left_knee, right_hip, right_knee]

            # Ensure ordering: actuated must map to the four joints in your URDF order.
            # If ordering mismatches, inspect the printed joint names and rearrange desired_angles accordingly.
            # We'll apply motor position control to follow desired_angles (easier than torque direct).
            for j_idx, q_des in zip(actuated, desired_angles):
                p.setJointMotorControl2(pb_robot_id, j_idx,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPosition=q_des,
                                        force=MAX_TORQUE)

            # step simulation
            backend.step()   # steps once (backend.step calls p.stepSimulation internally)
            time.sleep(SIM_STEP)
            step += 1

    except KeyboardInterrupt:
        print("Interrupted by user — closing.")
    finally:
        backend.shutdown()

if __name__ == "__main__":
    main()
