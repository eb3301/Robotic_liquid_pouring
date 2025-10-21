import genesis as gs
import numpy as np
import trimesh
import random 
import torch
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

gs.init(
        seed                = None,
        precision           = '32',
        debug               = True,
        eps                 = 1e-12,
        logging_level       = 'debug',
        backend             = gs.gpu,
        theme               = 'dark',
        logger_verbose_time = 'warning',
    )

dt=1e-2
debug=False
view=True
global scene
scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt=dt,
        substeps= 200, #10000*dt,  # Increased substeps for better stability
        gravity=(0, 0, -9.81),
    ),
    rigid_options=gs.options.RigidOptions(
    enable_collision=True,
    enable_self_collision=True,
    enable_adjacent_collision=False,
    # constraint_timeconst=0.001,
    # max_dynamic_constraints=10,
    ),
    sph_options=gs.options.SPHOptions(
        # position of the bounding box for the liquid
        lower_bound   = (-1.5, -1.5, 0.0), 
        upper_bound   = (1.5, 1.5, 2),
        particle_size = 0.01, #0.002  
    ),
    viewer_options = gs.options.ViewerOptions(
        res           = (640, 480),
        camera_pos    = (0.5, 0, 0.3),
        camera_lookat = (0, 0, 0.1),
        camera_fov    = 40,
        max_FPS       = 60,
    ),
    vis_options = gs.options.VisOptions(
        show_world_frame = debug, # visualize the coordinate frame of `world` at its origin
        world_frame_size = 0.5, # length of the world frame in meter
        show_link_frame  = debug, #  visualize coordinate frames of entity links
        show_cameras     = False, # visualize mesh and frustum of the cameras added
        plane_reflection = False, # turn on plane reflection
        ambient_light    = (0.1, 0.1, 0.1), # ambient light setting
        shadow=False,
    ),
    show_viewer = view,
    renderer = gs.renderers.Rasterizer(), # using rasterizer for camera rendering
    profiling_options = gs.options.ProfilingOptions(show_FPS = True),
    #renderer=gs.renderers.RayTracer()
)

DIR="/home/barutta/Robotic_liquid_pouring"
container_mesh_path = DIR + '/becher/becher1.obj'

plane = scene.add_entity(gs.morphs.Plane())

becher = scene.add_entity(
    gs.morphs.Mesh(
        file=container_mesh_path,
        fixed=True,
        pos=(0,0,0),
        euler=(90, 0, 180),
        scale=0.015,
        decimate=False,
        convexify=False,
        decompose_object_error_threshold=float("inf"),
        #decompose_nonconvex=False,
        # contype=0b011,
        # conaffinity=0b011,
        coacd_options=gs.options.CoacdOptions(),
        merge_submeshes_for_collision=True,
    ),
    material=gs.materials.Rigid(needs_coup=True),
    surface=gs.surfaces.Rough(
            diffuse_texture=gs.textures.ColorTexture()
    ),
    # vis_mode = "collision",
    visualize_contact=debug,
)

# bbox = mesh.bounding_box.extents
#     center = mesh.bounding_box.centroid
#     aabb = mesh.bounding_box
#     aabb_min = aabb.bounds[0]  # Minimum (x, y, z) of the bounding box
#     aabb_max = aabb.bounds[1]  # Maximum (x, y, z) of the bounding box

container_mesh = trimesh.load(container_mesh_path)
container_bounds = container_mesh.bounds
global container_size
container_size = (container_bounds[1] - container_bounds[0])*0.015

liquid_radius = min(container_size[0], container_size[1])/2*0.7
init_volume=100*1e-6
liquid_height = init_volume/(np.pi*liquid_radius**2)
print(f"Radius: {liquid_radius*10**3} mm, Height: {liquid_height*10**3} mm")
#liquid_height = container_size[2]*container_scale*np.sqrt(2)*0.5
#print(liquid_radius, liquid_height)
# Position liquid relative to container center
liqpos = (0,0,liquid_height/2+0.01) 


liquid = scene.add_entity(
    # viscous liquid
    #material=gs.materials.SPH.Liquid(mu=0.02, gamma=0.02),
    material=gs.materials.SPH.Liquid( 
        rho= 1000.0,
        stiffness=5000.0,
        exponent=7.0,
        mu= 0.001002,       # viscosità dinamica dell'acqua a 20 °C [Pa·s]
        gamma= 0.0728,  # tensione superficiale dell'acqua a 20 °C [N/m]),
        sampler='regular',
    ),
    morph=gs.morphs.Cylinder(
        pos  = liqpos,
        radius = liquid_radius,
        height = liquid_height,     
    ),
    surface=gs.surfaces.Default(
        color    = (0.4, 0.8, 1.0),
        vis_mode = 'particle', #recon / particle
    ),
)

scene.build()

# n_envs n_part pos3x
#vel =[]


for _ in range(1000):
   # liquid.set_vel(0, vel)
    scene.step()