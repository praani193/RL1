import os
import time
import numpy as np
import pybullet as p
import pybullet_data

class BipedEnv:
    def __init__(self, urdf_path="simple_biped.urdf", render=False, dt=1/240., substeps=4):
        self.render = render
        if render:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        self.dt = dt
        self.substeps = substeps
        self.urdf_path = urdf_path
        self.robot = None
        self.plane = None
        self.reset()

    def reset(self):
        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        self.plane = p.loadURDF("plane.urdf", physicsClientId=self.client)
        start_pos = [0, 0, 0.8]
        start_orn = p.getQuaternionFromEuler([0, 0, 0])
        cwd = os.path.dirname(os.path.abspath(__file__))
        urdf_full = os.path.join(cwd, self.urdf_path)
        self.robot = p.loadURDF(urdf_full, start_pos, start_orn, useFixedBase=False, physicsClientId=self.client)

        # disable default motors and set joint limits etc.
        self.joint_indices = []
        self.joint_names = []
        for i in range(p.getNumJoints(self.robot, physicsClientId=self.client)):
            info = p.getJointInfo(self.robot, i, physicsClientId=self.client)
            jtype = info[2]
            # we only actuate revolute joints (type 0 = revolute, 1 = prismatic)
            if jtype == p.JOINT_REVOLUTE or jtype == p.JOINT_PRISMATIC:
                self.joint_indices.append(i)
                self.joint_names.append(info[1].decode('utf-8'))
                # disable default position control
                p.setJointMotorControl2(bodyIndex=self.robot, jointIndex=i,
                                        controlMode=p.VELOCITY_CONTROL, force=0, physicsClientId=self.client)
        self.num_joints = len(self.joint_indices)
        # small random init pose
        for idx in self.joint_indices:
            p.resetJointState(self.robot, idx, targetValue=np.random.uniform(-0.1, 0.1), targetVelocity=0, physicsClientId=self.client)

        # step a few frames for stability
        for _ in range(10):
            p.stepSimulation(physicsClientId=self.client)
        return self.get_state()

    def get_state(self):
        base_pos, base_orn = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.client)
        base_euler = p.getEulerFromQuaternion(base_orn)
        base_lin_vel, base_ang_vel = p.getBaseVelocity(self.robot, physicsClientId=self.client)
        joint_angles = []
        joint_vels = []
        for idx in self.joint_indices:
            js = p.getJointState(self.robot, idx, physicsClientId=self.client)
            joint_angles.append(js[0])
            joint_vels.append(js[1])
        state = np.concatenate([
            np.array(base_pos), np.array(base_euler),
            np.array(base_lin_vel), np.array(base_ang_vel),
            np.array(joint_angles), np.array(joint_vels)
        ], axis=0)
        return state

    def step(self, action):
        """
        action: numpy array of length self.num_joints
        treated as desired torque for each actuated joint
        """
        # clip action
        action = np.clip(action, -1.0, 1.0)
        # scale to reasonable torque range
        MAX_TORQUE = 40.0
        torques = (action * MAX_TORQUE).tolist()

        for _ in range(self.substeps):
            for j_idx, tau in zip(self.joint_indices, torques):
                p.setJointMotorControl2(self.robot, j_idx, controlMode=p.TORQUE_CONTROL, force=tau, physicsClientId=self.client)
            p.stepSimulation(physicsClientId=self.client)
            if self.render:
                time.sleep(self.dt)

        state = self.get_state()
        reward = self._compute_reward(state, action)
        done = self._check_done(state)
        return state, reward, done, {}

    def _compute_reward(self, state, action):
        # simple reward: forward progress (x of base) minus energy penalty
        base_x = state[0]
        # we want incremental forward progress; compute via base linear velocity x
        vx = state[6]  # base lin vel x
        energy_penalty = 0.0
        # approximate energy = sum(|torque * joint_vel|)
        joint_vels = state[10 + self.num_joints : 10 + 2 * self.num_joints] if self.num_joints>0 else []
        for tau, vel in zip(action, joint_vels):
            energy_penalty += abs(tau * vel)
        reward = 1.0 * vx - 0.01 * energy_penalty
        return reward

    def _check_done(self, state):
        # done when torso height too low (fell) or too large pitch/roll
        base_z = state[2]
        roll = state[4]
        pitch = state[5]
        if base_z < 0.35 or abs(roll) > 1.0 or abs(pitch) > 1.0:
            return True
        return False

    def close(self):
        p.disconnect(physicsClientId=self.client)
