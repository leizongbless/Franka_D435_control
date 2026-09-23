# FR3 + D435i wrist camera mount

这是已经装配的 FR3 + 官方 Franka Hand/两指夹爪 + CNC 腕部连接件 + Intel RealSense D435i 模型。

MuJoCo 端采用了两个上游模型的做法：Franka Hand 使用 DeepMind MuJoCo Menagerie 的材质分离网格，D435i 使用 kyokochi/mujoco-model 的材质分离网格；这比直接把 ROS 的单个 DAE 转成 OBJ 更不容易透明或丢材质。

Dependencies:

```bash
sudo apt install ros-$ROS_DISTRO-franka-description
```

完整模型是 `urdf/fr3_d435i_official.urdf`。它保留官方 `franka_description` 的 FR3、`fr3_hand`、`fr3_leftfinger` 和 `fr3_rightfinger` 网格与惯量；连接件挂在 `fr3_hand`，相机挂在连接件上。

相机现在按连接件底部斜面安装：后侧参考点仍是 CAD 的 `x=72.566 mm, y=31.620 mm`，底面法向按 `25°` 斜面计算，并留 `0.5 mm` 装配间隙；相机宽度方向与夹爪侧面平行。MuJoCo 场景见 `mujoco_fr3/scene.xml`，Windows 运行 `python .\\mujoco_fr3\\view_fr3_d435i.py`。
