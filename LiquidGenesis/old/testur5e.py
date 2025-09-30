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

ur5e=scene.add_entity(gs.morphs.URDF(
        file = '/home/edo/thesis/LiquidGenesis/urdf/urdf_files_dataset/urdf_files/matlab/ur_description/urdf/ur5e.urdf',
        fixed=True,
        pos   = (0, 0, 0),
        euler = (0, 0, 0), # we follow scipy's extrinsic x-y-z rotation convention, in degrees,
        # quat  = (1.0, 0.0, 0.0, 0.0), # we use w-x-y-z convention for quaternions,
        scale = 1.0,
    ),
)
jnt_names = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint"
]


dofs_idx = [ur5e.get_joint(name).dof_idx_local for name in jnt_names]


########################## build ##########################
# create parallel environments
B = 1 # number of environments (entities)
scene.build(n_envs=B, env_spacing=(1.0, 1.0))

ur5e.set_dofs_position([0, -np.pi/2, 0, 0, 0, 0])
# set positional gains
ur5e.set_dofs_kp(
    kp             = np.array([4500, 3500, 2000, 2000, 100, 100]),
    dofs_idx_local = dofs_idx,
)

########################## main ##########################

# start camera recording. Once this is started, all the rgb images rendered will be recorded internally
cam.start_recording()

# TEST 1 - CAMERA MOTION
for i in range(1000):
    ur5e.control_dofs_position(
        np.array([0, (-np.pi/2+np.sin(i/(2*np.pi))*0.5), 0, 0, 0, 0])[1:],
        dofs_idx[1:],
    )
        
    
    scene.step()

    # change camera position
    cam.set_pose(
        pos    = (3.0 * np.sin(i / 60), 3.0 * np.cos(i / 60), 2.5),
        lookat = (0, 0, 0.5),
    )
    
    cam.render()


# stop recording and save video. If `filename` is not specified, a name will be auto-generated using the caller file name.
cam.stop_recording(save_to_filename='video.mp4', fps=60)