# 📸 Visualization and Rendering

The Genesis visualization system is managed by the `visualizer` of the newly created scene (i.e., `scene.visualizer`). There are two ways to visualize the scene: 1) using an interactive viewer that runs in a separate thread, and 2) manually adding a camera to the scene and rendering images with the camera.

## Viewer

If a monitor is connected, an interactive viewer can be used to visualize the scene. Genesis uses different`options`Groups are used to configure different components in the scenario. To configure the viewer, you can change it during scene creation.`viewer_options`parameters in . Additionally, we use`vis_options`Specify the attributes related to visualization. These attributes will be shared by the viewer and the camera (we will add the camera soon).
Create a scene with more detailed viewer and visualization settings (this seems a bit complex, but it is just for illustrative purposes):

```python
scene = gs.Scene(
    show_viewer    = True,
    viewer_options = gs.options.ViewerOptions(
        res           = (1280, 960),
        camera_pos    = (3.5, 0.0, 2.5),
        camera_lookat = (0.0, 0.0, 0.5),
        camera_fov    = 40,
        max_FPS       = 60,
    ),
    vis_options = gs.options.VisOptions(
        show_world_frame = True, # 可视化`world`在其原点的坐标系
        world_frame_size = 1.0, # 世界坐标系的长度（米）
        show_link_frame  = False, # 不可视化实体链接的坐标系
        show_cameras     = False, # 不可视化添加的相机的网格和视锥
        plane_reflection = True, # 打开平面反射
        ambient_light    = (0.1, 0.1, 0.1), # 环境光设置
    ),
    renderer = gs.renderers.Rasterizer(), # 使用光栅化器进行相机渲染
)
```

1. We can specify the pose and field of view angle of the viewer camera here. If `max_FPS` is set to `None`, the viewer will run as quickly as possible. If `res` is set to None, Genesis will automatically create a 4:3 window with the height set to half the monitor height. Note that in these settings, we use a rasterization backend for camera rendering. Genesis offers two rendering backends: `gs.renderers.Rasterizer()` and `gs.renderers.RayTracer()`. The viewer always uses a rasterizer. By default, the camera also uses a rasterizer.
Once the scene is created, the viewer object can be accessed via `scene.visualizer.viewer` or its abbreviation `scene.viewer`. The viewer camera pose can be queried or set:

```python
cam_pose = scene.viewer.camera_pose()
scene.viewer.set_camera_pose(cam_pose)
```

## Camera and Headless Rendering

Now let's manually add a camera object to the scene. The camera is not connected to a viewer or monitor; it only returns the rendered image when needed. Therefore, the camera works in headless mode.

```python
cam = scene.add_camera(
    res    = (1280, 960),
    pos    = (3.5, 0.0, 2.5),
    lookat = (0, 0, 0.5),
    fov    = 30,
    GUI    = False
)
```

If`GUI=True`Each camera will create an OpenCV window to dynamically display the rendered image. Note that this is different from the viewer GUI.
Then, once we build the scene, we can use the camera to render images. Our camera supports rendering RGB images, depth maps, segmentation masks, and surface normals. By default, only RGB is rendered; this can be changed by calling`camera.render()`Set parameters to enable other modes:

```python
scene.build()
# 渲染rgb、深度、分割掩码和法线图
rgb, depth, segmentation, normal = cam.render(depth=True, segmentation=True, normal=True)
```

If `GUI=True` is used and a monitor is connected, 4 windows should now be visible. (Sometimes, OpenCV windows have additional latency; therefore, if the windows are black, you can call additional `cv2.waitKey(1)`, or simply call `render()` again to refresh the windows.)

```{figure}

```

**Use the camera to record video**
Now, let's render only the RGB images, move the camera, and record video. Genesis provides a convenient tool for recording video:

```python
# 开始相机录制。一旦开始，所有渲染的rgb图像将被内部记录
cam.start_recording()
import numpy as np
for i in range(120):
    scene.step()
    # 改变相机位置
    cam.set_pose(
        pos    = (3.0 * np.sin(i / 60), 3.0 * np.cos(i / 60), 2.5),
        lookat = (0, 0, 0.5),
    )

    cam.render()
# 停止录制并保存视频。如果未指定`filename`，将使用调用文件名自动生成名称。
cam.stop_recording(save_to_filename='video.mp4', fps=60)
```

Save the video to `video.mp4`:
<video preload="auto" controls="True" width="100%">


```python
import genesis as gs
gs.init(backend=gs.cpu)
scene = gs.Scene(
    show_viewer = True,
    viewer_options = gs.options.ViewerOptions(
        res           = (1280, 960),
        camera_pos    = (3.5, 0.0, 2.5),
        camera_lookat = (0.0, 0.0, 0.5),
        camera_fov    = 40,
        max_FPS       = 60,
    ),
    vis_options = gs.options.VisOptions(
        show_world_frame = True,
        world_frame_size = 1.0,
        show_link_frame  = False,
        show_cameras     = False,
        plane_reflection = True,
        ambient_light    = (0.1, 0.1, 0.1),
    ),
    renderer=gs.renderers.Rasterizer(),
)
plane = scene.add_entity(
    gs.morphs.Plane(),
)
franka = scene.add_entity(
    gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'),
)
cam = scene.add_camera(
    res    = (640, 480),
    pos    = (3.5, 0.0, 2.5),
    lookat = (0, 0, 0.5),
    fov    = 30,
    GUI    = False,
)
scene.build()
# Render RGB, depth, segmentation masks, and normal maps.
# rgb, depth, segmentation, normal = cam.render(rgb=True, depth=True, segmentation=True, normal=True)
cam.start_recording()
import numpy as np
for i in range(120):
    scene.step()
    cam.set_pose(
        pos    = (3.0 * np.sin(i / 60), 3.0 * np.cos(i / 60), 2.5),
        lookat = (0, 0, 0.5),
    )
    cam.render()
cam.stop_recording(save_to_filename='video.mp4', fps=60)
```




## Parallel Simulation

```python
import torch
import genesis as gs

gs.init(backend=gs.gpu)

scene = gs.Scene(
    show_viewer   = False,
    rigid_options = gs.options.RigidOptions(
        dt                = 0.01,
    ),
)

plane = scene.add_entity(
    gs.morphs.Plane(),
)

franka = scene.add_entity(
    gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'),
)

scene.build(n_envs=30000)

# 控制所有机器人
franka.control_dofs_position(
    torch.tile(
        torch.tensor([0, 0, 0, -1.0, 0, 0, 0, 0.02, 0.02], device=gs.device), (30000, 1)
    ),
)

for i in range(1000):
    scene.step()
```

## test_genesis_bottle.py

```py
import argparse
import time

import genesis as gs
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument("-B", type=int, default=1) # batch size
parser.add_argument("-v", action="store_true", default=False) # visualize
parser.add_argument("-r", action="store_true", default=False) # random action

args = parser.parse_args()

########################## init ##########################
gs.init(backend=gs.gpu)

########################## create a scene ##########################
scene = gs.Scene(
    show_viewer=args.v,
    rigid_options=gs.options.RigidOptions(
        dt=0.01,
        constraint_solver=gs.constraint_solver.Newton,
        enable_self_collision=True,
    ),
)

########################## entities ##########################
plane = scene.add_entity(
    gs.morphs.Plane(),
)

obj = scene.add_entity(
    morph=gs.morphs.URDF(
        file="../assets/urdf/bottle/bottle.urdf",
        scale=0.09,
        pos=(0.65, 0.0, 0.036),
        euler=(0, 90, 0),
    ),
)
franka = scene.add_entity(
    gs.morphs.MJCF(file="../assets/xml/franka_emika_panda/panda.xml")
)

########################## build ##########################
n_envs = args.B
scene.build(n_envs=n_envs)

# set control gains (official value taken from mujoco franka xml)
franka.set_dofs_kp(
    np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
)
franka.set_dofs_kv(
    np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
)
franka.set_dofs_force_range(
    np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
    np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
)

motors_dof = np.arange(7)
fingers_dof = np.arange(7, 9)

grasp_qpos = np.array([-0.9937,  1.4588,  1.3058, -1.6924, -1.4882,  1.8461,  1.4577,  0.04, 0.04])
lift_qpos = np.array([-1.0411,  1.2861,  1.5200, -1.7065, -1.2946,  1.6572,  1.4315,  0.04, 0.04])
franka.set_dofs_position(grasp_qpos)

# grasp
franka.control_dofs_position(grasp_qpos[:-2], motors_dof)
franka.control_dofs_force(np.array([-4, -4]), fingers_dof)


for i in range(100):
    scene.step()

# lift
franka.control_dofs_position(lift_qpos[:-2], motors_dof)
for i in range(50):
    scene.step()

ref_pos = torch.tile(torch.tensor(lift_qpos[:7]), [n_envs, 1]).cuda()

t0 = time.perf_counter()
for i in range(500):
    if args.r and i % 2 == 0: # match maniskill's 50hz control freq
        # match maniskill's random action
        franka.control_dofs_position(ref_pos + torch.rand((n_envs, 7), device='cuda')*0.05 - 0.025, motors_dof)
    scene.step()
t1 = time.perf_counter()

print(f'per env: {500 / (t1 - t0):,.2f} FPS')
print(f'total  : {500 / (t1 - t0) * n_envs:,.2f} FPS')
```



### 03test_genesis_cube.py

```py
import argparse
import time

import genesis as gs
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument("-B", type=int, default=1) # batch size
parser.add_argument("-v", action="store_true", default=False) # visualize
parser.add_argument("-r", action="store_true", default=False) # random action
parser.add_argument("-m", action="store_true", default=False) # move along a traj
parser.add_argument("--mjcf", action="store_true", default=False) # use mjcf

args = parser.parse_args()

########################## init ##########################
gs.init(backend=gs.gpu)

########################## create a scene ##########################
scene = gs.Scene(
    show_viewer=args.v,
    rigid_options=gs.options.RigidOptions(
        dt=0.01,
        constraint_solver=gs.constraint_solver.Newton,
        enable_self_collision=True,
    ),
)

########################## entities ##########################
plane = scene.add_entity(
    gs.morphs.Plane(),
)

cube = scene.add_entity(
    gs.morphs.Box(
        size=(0.04, 0.04, 0.04),
        pos=(0.65, 0.0, 0.02),
    ),
)
franka = scene.add_entity(
    gs.morphs.MJCF(file="../assets/xml/franka_emika_panda/panda.xml") if args.mjcf else gs.morphs.URDF(file="../assets/urdf/franka_description/robots/franka_panda.urdf", fixed=True),
)

########################## build ##########################
n_envs = args.B
scene.build(n_envs=n_envs)

# set control gains (official value taken from mujoco franka xml)
franka.set_dofs_kp(
    np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
)
franka.set_dofs_kv(
    np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
)
franka.set_dofs_force_range(
    np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
    np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
)

motors_dof = np.arange(7)
fingers_dof = np.arange(7, 9)

grasp_qpos = np.array([-1.0104,  1.5623,  1.3601, -1.6840, -1.5863,  1.7810,  1.4598,  0.0400, 0.0400])
lift_qpos = np.array([-1.0426,  1.4028,  1.5634, -1.7114, -1.4055,  1.6015,  1.4510,  0., 0.])
franka.set_dofs_position(grasp_qpos)

# grasp
franka.control_dofs_position(grasp_qpos[:-2], motors_dof)
franka.control_dofs_force(np.array([-0.5, -0.5]), fingers_dof)


for i in range(100):
    scene.step()

# lift
franka.control_dofs_position(lift_qpos[:-2], motors_dof)
for i in range(50):
    scene.step()

ref_pos = torch.tile(torch.tensor(lift_qpos[:7]), [n_envs, 1]).cuda()

t0 = time.perf_counter()
if args.m:
    init_qpos = torch.tile(torch.tensor([-1.0426,  1.4028,  1.5634, -1.7114, -1.4055,  1.6015,  1.4510]), [n_envs, 1]).cuda()
    target_qpos = torch.tile(torch.tensor([1.0426,  -1.4028,  1.5634, -1.7114, -1.4055,  1.6015,  1.4510]), [n_envs, 1]).cuda()
    for i in range(500):
        if i % 2 == 0: # match maniskill's 50hz control freq
            franka.control_dofs_position(init_qpos + (target_qpos - init_qpos) * i / 500, motors_dof)
        scene.step()
else:
    for i in range(500):
        if args.r and i % 2 == 0: # match maniskill's 50hz control freq
            # match maniskill's random action
            franka.control_dofs_position(ref_pos + torch.rand((n_envs, 7), device='cuda')*0.05 - 0.025, motors_dof)
        scene.step()
t1 = time.perf_counter()

print(f'per env: {500 / (t1 - t0):,.2f} FPS')
print(f'total  : {500 / (t1 - t0) * n_envs:,.2f} FPS')
```

## Inverse Kinematics

```py

```
