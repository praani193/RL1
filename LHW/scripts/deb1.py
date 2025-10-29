import numpy as np
from LHW.envs.jvrc.jvrc_walk import JvrcWalkEnv
import time
import mujoco
from mujoco import viewer

# Create environment
env = JvrcWalkEnv()
obs = env.reset_model()

# Initialize model and data
model = env.model
data = env.data

print("Launching MuJoCo live viewer...")

try:
    # Launch live interactive viewer
    with viewer.launch_passive(model, data) as v:
        # Patch: Assign viewer safely (not old mujoco_py viewer)
        env.viewer = v

        # Try to call viewer_setup() only if it supports new API
        try:
            if hasattr(env, "viewer_setup"):
                env.viewer_setup()
        except Exception as e:
            print(f"Ignoring viewer_setup() incompatibility: {e}")

        # Main visualization loop
        for _ in range(1000):
            # Use model.nu for number of actuators
            action = np.zeros(env.model.nu)

            # Step environment
            obs, reward, done, info = env.step(action)

            # Forward simulation and sync viewer
            mujoco.mj_forward(model, data)
            v.sync()

            # Slow down to real time
            time.sleep(getattr(env.cfg, "control_dt", 0.01))

except Exception as e:
    print(f"Viewer failed: {e}")
    print("Falling back to dummy render (no viewer)...")
    try:
        if hasattr(env, "render"):
            env.render()
        for _ in range(500):
            action = np.zeros(env.model.nu)
            obs, reward, done, info = env.step(action)
            if hasattr(env, "render"):
                env.render()
            time.sleep(getattr(env.cfg, "control_dt", 0.01))
    except Exception as inner:
        print(f"Render fallback failed: {inner}")

env.close()
