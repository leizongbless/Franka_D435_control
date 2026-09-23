from pathlib import Path
import mujoco
import mujoco.viewer

model_path = Path(__file__).with_name("scene.xml")
model = mujoco.MjModel.from_xml_path(str(model_path))
data = mujoco.MjData(model)
home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
if home_id >= 0:
    mujoco.mj_resetDataKeyframe(model, data, home_id)
mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:
    # Focus the initial view on the assembled hand, mount, camera, and open fingers.
    viewer.cam.lookat[:] = [0.565, 0.035, 0.615]
    viewer.cam.distance = 0.38
    viewer.cam.azimuth = 225
    viewer.cam.elevation = -20
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
