import numpy as np
from LHW.envs.h1.h1_env import H1Env
import time

# Create environment
env = H1Env()
obs = env.reset_model()

# Initialize viewer
env.render()
env.viewer_setup()

# Visualization loop
for _ in range(500):
    action = np.zeros(len(env.leg_names))  # Convert to np.array
    obs, reward, done, info = env.step(action)
    env.render()
    time.sleep(env.cfg.control_dt)

env.close()
