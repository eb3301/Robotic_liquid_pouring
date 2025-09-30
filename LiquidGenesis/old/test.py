import os
os.environ['PYOPENGL_PLATFORM'] = 'glx'
os.environ['MUJOCO_GL'] = 'glx'
import genesis as gs
#import cv2
import numpy as np

########################## init ##########################

gs.init(
    seed                = None,
    precision           = '32',
    debug               = False,
    eps                 = 1e-12,
    logging_level       = None,
    backend             = gs.cpu,
    theme               = 'dark',
    logger_verbose_time = 'Debug'
)

########################## create a scene ##########################
scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt=0.01,
        gravity=(0, 0, -9.81),
    ),
    show_viewer    = True,
    viewer_options = gs.options.ViewerOptions(
        res           = (1280, 960),
        camera_pos    = (3.5, 0.0, 2.5),
        camera_lookat = (0.0, 0.0, 0.5),
        camera_fov    = 40,
        max_FPS       = 60,
    ),
    vis_options = gs.options.VisOptions(
        show_world_frame = True, # visualize the coordinate frame of `world` at its origin
        world_frame_size = 1.0, # length of the world frame in meter
        show_link_frame  = False, # do not visualize coordinate frames of entity links
        show_cameras     = False, # do not visualize mesh and frustum of the cameras added
        plane_reflection = True, # turn on plane reflection
        ambient_light    = (0.1, 0.1, 0.1), # ambient light setting
    ),
    renderer = gs.renderers.Rasterizer(), # using rasterizer for camera rendering
    #renderer=gs.renderers.RayTracer()
)

# It is possible to change the camera position in the following way:
# cam_pose = scene.viewer.camera_pose
# scene.viewer.set_camera_pose(cam_pose)

# Camera & Headless Rendering:
cam = scene.add_camera(
    res    = (1280, 960),
    pos    = (3.5, 0.0, 2.5),
    lookat = (0, 0, 0.5),
    fov    = 30,
    GUI    = True
)

########################## entities ##########################
# Premade entities:
plane = scene.add_entity(gs.morphs.Plane())
# plane = scene.add_entity(gs.morphs.Box())
# plane = scene.add_entity(gs.morphs.Cylinder())
# plane = scene.add_entity(gs.morphs.Sphere())


franka = scene.add_entity(
    gs.morphs.MJCF(
        file  = 'xml/franka_emika_panda/panda.xml',
        pos   = (0, 0, 0),
        euler = (0, 0, 90), # we follow scipy's extrinsic x-y-z rotation convention, in degrees,
        # quat  = (1.0, 0.0, 0.0, 0.0), # we use w-x-y-z convention for quaternions,
        scale = 1.0,
    ),
)
jnt_names = [
    'joint1',
    'joint2',
    'joint3',
    'joint4',
    'joint5',
    'joint6',
    'joint7',
    'finger_joint1',
    'finger_joint2',
]


dofs_idx = [franka.get_joint(name).dof_idx_local for name in jnt_names]


# Supported entity types:
# gs.morphs.MJCF: mujoco .xml robot configuration files
# gs.morphs.URDF: robot description files that end with .urdf (Unified Robotics Description Format)
# gs.morphs.Mesh: non-articulated mesh assets, supporting extensions including: *.obj, *.ply, *.stl, *.glb, *.gltf

########################## build ##########################
# create 20 parallel environments
B = 1 # number of environments (entities)
scene.build(n_envs=B, env_spacing=(1.0, 1.0))



########################## main ##########################
# render rgb, depth, segmentation mask and normal map
#cv2.waitKey(1)
# rgb, depth, segmentation, normal = cam.render(depth=True, segmentation=True, normal=True)
# cam.render()

# start camera recording. Once this is started, all the rgb images rendered will be recorded internally
#cam.start_recording()

# TEST 1 - CAMERA MOTION
# for i in range(120):
#     scene.step()

#     # change camera position
#     cam.set_pose(
#         pos    = (3.0 * np.sin(i / 60), 3.0 * np.cos(i / 60), 2.5),
#         lookat = (0, 0, 0.5),
#     )
    
#     cam.render()

# TEST 2 - HARDE RESET ROBOT POS
# ############ Optional: set control gains ############

# # set positional gains
# franka.set_dofs_kp(
#     kp             = np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
#     dofs_idx_local = dofs_idx,
# )
# # set velocity gains
# franka.set_dofs_kv(
#     kv             = np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
#     dofs_idx_local = dofs_idx,
# )
# # set force range for safety
# franka.set_dofs_force_range(
#     lower          = np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
#     upper          = np.array([ 87,  87,  87,  87,  12,  12,  12,  100,  100]),
#     dofs_idx_local = dofs_idx,
# )
# for i in range(150):
#     if i < 50:
#         franka.set_dofs_position(np.array([1, 1, 0, 0, 0, 0, 0, 0.04, 0.04]), dofs_idx)
#     elif i < 100:
#         franka.set_dofs_position(np.array([-1, 0.8, 1, -2, 1, 0.5, -0.5, 0.04, 0.04]), dofs_idx)
#     else:
#         franka.set_dofs_position(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]), dofs_idx)

#     scene.step()

# TEST 3 - PD ROBOT POS CONTROL
############ Optional: set control gains ############

# set positional gains
franka.set_dofs_kp(
    kp             = np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
    dofs_idx_local = dofs_idx,
)
# set velocity gains
franka.set_dofs_kv(
    kv             = np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
    dofs_idx_local = dofs_idx,
)
# set force range for safety
franka.set_dofs_force_range(
    lower          = np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
    upper          = np.array([ 87,  87,  87,  87,  12,  12,  12,  100,  100]),
    dofs_idx_local = dofs_idx,
)
# PD control
for i in range(1250):
    if i == 0:
        franka.control_dofs_position(
            np.array([1, 1, 0, 0, 0, 0, 0, 0.04, 0.04]),
            dofs_idx,
        )
    elif i == 250:
        franka.control_dofs_position(
            np.array([-1, 0.8, 1, -2, 1, 0.5, -0.5, 0.04, 0.04]),
            dofs_idx,
        )
    elif i == 500:
        franka.control_dofs_position(
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
            dofs_idx,
        )
    elif i == 750:
        # control first dof with velocity, and the rest with position
        franka.control_dofs_position(
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])[1:],
            dofs_idx[1:],
        )
        franka.control_dofs_velocity(
            np.array([1.0, 0, 0, 0, 0, 0, 0, 0, 0])[:1],
            dofs_idx[:1],
        )
    elif i == 1000:
        franka.control_dofs_force(
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
            dofs_idx,
        )

    # In case of using GPU, to reduce data transfer between CPU and GPU, it is possible to do the following:
    # control only 3 environments: 1, 5, and 7.
    # franka.control_dofs_position(
    #     position = torch.zeros(3, 9, device=gs.device), # zero-position command
    #     envs_idx = torch.tensor([1, 5, 7], device=gs.device),
    # )

    # This is the control force computed based on the given control command
    # If using force control, it's the same as the given control command
    print('control force:', franka.get_dofs_control_force(dofs_idx))

    # This is the actual force experienced by the dof
    print('internal force:', franka.get_dofs_force(dofs_idx))

    scene.step()

    # change camera position
    cam.set_pose(
        pos    = (4.0 * np.sin(i / 60*0.5), 4.0 * np.cos(i / 60*0.5), 2.5),
        lookat = (0, 0, 0.5),
    )
    
    cam.render()

# stop recording and save video. If `filename` is not specified, a name will be auto-generated using the caller file name.
#cam.stop_recording(save_to_filename='video.mp4', fps=60)