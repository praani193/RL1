import mujoco
from pathlib import Path

# Path to your MuJoCo XML
xml_path = Path("C:\Users\Lenovo\Desktop\RL1\LHW\tmp\mjcf-export")  # e.g., "LHW/envs/jvrc/assets/jvrc_step.xml"

# Load the model
model = mujoco.MjModel.from_xml_path(str(xml_path))
data = mujoco.MjData(model)

# Print all bodies and indices
print("Checking bodies in the MuJoCo model...")
for i in range(model.nbody):
    print(f"Index {i}: {model.body(i).name}")
print(f"Total bodies: {model.nbody}")

# Check if required bodies exist
required_bodies = ["R_ANKLE_P_S", "L_ANKLE_P_S"]
for rb in required_bodies:
    if rb not in [model.body(i).name for i in range(model.nbody)]:
        print(f"Warning: {rb} not found in model!")
