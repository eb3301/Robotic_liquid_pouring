import mujoco
import mujoco.viewer
import os, sys
import numpy as np
from scipy.spatial.transform import Rotation as R

import numpy as np

def rotm_to_quaternion(R):
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2,1] - R[1,2]) * s
        y = (R[0,2] - R[2,0]) * s
        z = (R[1,0] - R[0,1]) * s
    else:
        if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
            s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
            w = (R[2,1] - R[1,2]) / s
            x = 0.25 * s
            y = (R[0,1] + R[1,0]) / s
            z = (R[0,2] + R[2,0]) / s
        elif R[1,1] > R[2,2]:
            s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
            w = (R[0,2] - R[2,0]) / s
            x = (R[0,1] + R[1,0]) / s
            y = 0.25 * s
            z = (R[1,2] + R[2,1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
            w = (R[1,0] - R[0,1]) / s
            x = (R[0,2] + R[2,0]) / s
            y = (R[1,2] + R[2,1]) / s
            z = 0.25 * s

    # Restituisci nel formato x, y, z, w come richiesto
    return np.array([w, x, y, z])

# Path to your XML file
base_dir = "/home/barutta/projects/src/" # os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(base_dir)
model_path = os.path.join(base_dir, "ur5e_utils_mujoco/ur5e/ur5e.xml")

# Load the model
model = mujoco.MjModel.from_xml_path(model_path)
# Create data structure
data = mujoco.MjData(model)
# Launch the viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    q = np.array([1.5, 4.5500, -1.5874, 0.5281, -4.0680, 2.2158])
    data.qpos[:6] = q.tolist()
    print(data)
    #tool_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'tool_site')
    tool_body_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_frame")

    mujoco.mj_forward(model, data)
    viewer.sync()

    pos = data.xpos[tool_body_id]  # shape: (3,)
    # rot = data.xmat[tool_body_id].reshape(3, 3)  # shape: (3, 3)

#quat = rotm_to_quaternion(rot)
#print(f"FK wrist 3: pos={np.round(pos, 3)}, quat={np.round(quat, 3)}")
print(f"FK wrist 3: pos={np.round(pos, 3)}")
input("Press Enter to continue...")
    