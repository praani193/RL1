import mujoco

# Load the binary .0 file
model = mujoco.MjModel.from_binary_path("C:/Users/Lenovo/Desktop/RL1/LHW/RL_logsr1/events.out.tfevents.1762358436.DESKTOP-MA5G4HE.3792.0")

# Inspect data
print("Number of bodies:", model.nbody)
print("Number of joints:", model.njnt)
print("Number of geoms:", model.ngeom)

# Access part names
for i in range(model.nbody):
    print(i, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i))
