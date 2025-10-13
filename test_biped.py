import numpy as np
from biped_env import BipedEnv
import time

def main():
    env = BipedEnv(render=True)
    print("Joint indices:", env.joint_indices)
    print("Joint names:", env.joint_names)
    print("Num joints:", env.num_joints)

    obs = env.reset()
    total_reward = 0.0
    steps = 0
    try:
        for ep in range(50):
            obs = env.reset()
            done = False
            ep_reward = 0.0
            while not done and steps < 2000:
                # random sinusoidal-ish actions to give some motion
                t = steps * 0.02
                action = 0.6 * np.array([np.sin(t), np.cos(t), np.cos(t*1.2), np.sin(t*1.2)])
                obs, reward, done, info = env.step(action)
                ep_reward += reward
                steps += 1
            print(f"Episode {ep} reward {ep_reward:.2f}, steps {steps}")
            total_reward += ep_reward
            time.sleep(0.5)
    finally:
        env.close()
    print("Total reward:", total_reward)

if __name__ == "__main__":
    main()
