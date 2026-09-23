import mujoco
from pathlib import Path
from PIL import Image
m=mujoco.MjModel.from_xml_path('mujoco_fr3/scene.xml'); d=mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m,d,mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_KEY,'home'))
mujoco.mj_forward(m,d)
r=mujoco.Renderer(m,600,600)
cam=mujoco.MjvCamera(); cam.lookat[:]=[0.56,0.035,0.59]; cam.distance=0.30; cam.azimuth=225; cam.elevation=-20
r.update_scene(d, camera=cam)
Image.fromarray(r.render()).save('mujoco_fr3/fr3_d435i_preview.png'); print('saved')

