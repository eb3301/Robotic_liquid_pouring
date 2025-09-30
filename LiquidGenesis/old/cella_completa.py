import os
os.environ['PYOPENGL_PLATFORM'] = 'glx'
os.environ['MUJOCO_GL'] = 'glx'
import genesis as gs
import numpy as np
import trimesh  

gs.init(
    seed                = None,
    precision           = '32',
    debug               = False,
    eps                 = 1e-12,
    logging_level       = None,
    backend             = gs.cpu,
    theme               = 'dark',
    logger_verbose_time = 'Debug',
)

scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt=1e-3,
        substeps=100,  # Increased substeps for better stability
        gravity=(0, 0, -9.81),
    ),
    sph_options=gs.options.SPHOptions(
        # position of the bounding box for the liquid
        lower_bound   = (-1.5, -1.5, 0.0), 
        upper_bound   = (1.5, 1.5, 2),
        particle_size = 0.009,  
    ),
    viewer_options = gs.options.ViewerOptions(
        res           = (1280, 960),
        camera_pos    = (3.5, 0.0, 2.5),
        camera_lookat = (0.0, 0.0, 0.5),
        camera_fov    = 40,
        max_FPS       = 60,
    ),
    vis_options = gs.options.VisOptions(
        show_world_frame = True, # visualize the coordinate frame of `world` at its origin
        world_frame_size = 0.5, # length of the world frame in meter
        show_link_frame  = False, # do not visualize coordinate frames of entity links
        show_cameras     = False, # do not visualize mesh and frustum of the cameras added
        plane_reflection = False, # turn on plane reflection
        ambient_light    = (0.1, 0.1, 0.1), # ambient light setting
    ),
    show_viewer = True,
    renderer = gs.renderers.Rasterizer(), # using rasterizer for camera rendering
    show_FPS = False,
    #renderer=gs.renderers.RayTracer()
)

plane = scene.add_entity(gs.morphs.Plane())


ur5e=scene.add_entity(gs.morphs.URDF(
        file = '/home/edo/thesis/ur5e_urdf/urdf/ur5e_complete.urdf',
        fixed=True,
        pos   = (0, 0, 0),
        euler = (0, 0, 0), # we follow scipy's extrinsic x-y-z rotation convention, in degrees,
        # quat  = (1.0, 0.0, 0.0, 0.0), # we use w-x-y-z convention for quaternions,
        scale = 1.0,
    ),
    material=gs.materials.Rigid(),
)

jnt_names = []
dofs_idx = []
for joint in ur5e.joints:
    if joint.name not in ["joint_world"]:
        jnt_names.append(joint.name)
        dofs_idx.append(joint.dof_idx_local)


# scene.build()

# for i in range(1000):
#     scene.step()
#     if i % 10 == 0:
#         print(f"Step {i}")

