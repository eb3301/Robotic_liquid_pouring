import os
os.environ['PYOPENGL_PLATFORM'] = 'glx'
os.environ['MUJOCO_GL'] = 'glx'
import genesis as gs
import numpy as np
import trimesh  

def euler_to_quaternion(euler_angles):
    """
    Convert Euler angles (in radians) to quaternion using ZYX convention.
    
    Args:
        euler_angles: numpy array or list of [yaw, pitch, roll] in radians
        
    Returns:
        numpy array of quaternion [w, x, y, z]
    """
    yaw, pitch, roll = euler_angles
    
    # Pre-compute trigonometric values
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    
    # Compute quaternion components
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return np.array([w, x, y, z])

def quaternion_to_euler(quaternion):
    """
    Convert quaternion to Euler angles (in radians) using ZYX convention.
    
    Args:
        quaternion: numpy array or list of [w, x, y, z]
        
    Returns:
        numpy array of Euler angles [yaw, pitch, roll] in radians
    """
    w, x, y, z = quaternion
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = np.sqrt(1 + 2 * (w * y - x * z))
    cosp = np.sqrt(1 - 2 * (w * y - x * z))
    pitch = 2 * np.arctan2(sinp, cosp) - np.pi / 2
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return np.array([yaw, pitch, roll])

def quaternion_multiply(q1, q2):
    """
    Multiply two quaternions.
    
    Args:
        q1: First quaternion [w, x, y, z]
        q2: Second quaternion [w, x, y, z]
        
    Returns:
        Resulting quaternion [w, x, y, z]
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return np.array([w, x, y, z])

def set_liquid_pose(becher):
    becher_pos = becher.get_pos().cpu().numpy()
    n_particles = liquid.get_particles().shape[0]
    spacing = scene.sph_options.particle_size/2*0.9
    
    # Create a grid of positions around the becher position
    grid_size = int(np.ceil(np.cbrt(n_particles)))
    x = np.linspace(-spacing*grid_size/2, spacing*grid_size/2, grid_size)
    y = np.linspace(-spacing*grid_size/2, spacing*grid_size/2, grid_size)
    z = np.linspace(0, spacing*grid_size, grid_size)
    xx, yy, zz = np.meshgrid(x, y, z)
 
    offsets = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1)[:n_particles]

    pos = np.array([becher_pos[0], becher_pos[1], becher_pos[2]+0.01], dtype=np.float32) + offsets.astype(np.float32)
    liquid.set_pos(-1, pos=pos)

########################## init ##########################

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

########################## create a scene ##########################
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
    show_viewer = False,
    renderer = gs.renderers.Rasterizer(), # using rasterizer for camera rendering
    show_FPS = False,
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
    GUI    = False
)

########################## entities ##########################
plane = scene.add_entity(gs.morphs.Plane())

ur5e=scene.add_entity(gs.morphs.URDF(
        file = '/home/edo/thesis/LiquidGenesis/urdf/urdf_files_dataset/urdf_files/matlab/ur_description/urdf/ur5e.urdf',
        fixed=True,
        pos   = (0, 0, 0),
        euler = (0, 0, 0), # we follow scipy's extrinsic x-y-z rotation convention, in degrees,
        # quat  = (1.0, 0.0, 0.0, 0.0), # we use w-x-y-z convention for quaternions,
        scale = 1.0,
    ),
    material=gs.materials.Rigid(),
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

# Position container relative to end effector
contpos=(0.0, -0.04, 0.04)  # Initial position, will be updated after build
container_scale = 0.001
container_mesh_path = '/home/edo/thesis/LiquidGenesis/containers/test-Body.obj'

becher = scene.add_entity(
    gs.morphs.Mesh(
        file=container_mesh_path,
        fixed=True,
        pos=contpos,
        euler=(-90, 0, 0),
        scale=container_scale,
        decimate=False,
        convexify=False,
        decompose_nonconvex=True,
        coacd_options=gs.options.CoacdOptions(),
        merge_submeshes_for_collision=True,
    ),
    material=gs.materials.Rigid(needs_coup=True),
)

scene.link_entities(
        ur5e,
        becher,
        parent_link_name="wrist_3_link",
        child_link_name="test-Body_obj_baselink",
    )

# Load and analyze container mesh
container_mesh = trimesh.load(container_mesh_path)
container_bounds = container_mesh.bounds
container_size = container_bounds[1] - container_bounds[0]
container_center = container_mesh.center_mass
# Calculate liquid dimensions based on container size
liquid_radius = min(container_size[0], container_size[1])*container_scale*1/np.sqrt(2)*0.9
liquid_height = container_size[2]*container_scale*np.sqrt(2)*0.5
# Position liquid relative to container center
liqpos = (contpos[0], contpos[1], contpos[2] + liquid_height/2)
#liqpos = (0.0, 0.0, 0.05)


liquid = scene.add_entity(
    # viscous liquid
    #material=gs.materials.SPH.Liquid(mu=0.02, gamma=0.02),
    material=gs.materials.SPH.Liquid( 
        rho=1000.0,
        stiffness=50000.0,
        exponent=7.0,
        mu=0.001002,       # viscosità dinamica dell'acqua a 20 °C [Pa·s]
        gamma=0.0728,       # tensione superficiale dell'acqua a 20 °C [N/m]),
    ),
    morph=gs.morphs.Box(
        pos  = liqpos,
        size = (liquid_radius, liquid_radius, liquid_height),
    ),
    surface=gs.surfaces.Default(
        color    = (0.4, 0.8, 1.0),
        vis_mode = 'particle', #recon / particle
    ),
)

container = scene.add_entity(
    gs.morphs.Mesh(
        file=container_mesh_path,
        fixed=True,
        pos=(0.5,0.5,0.05),
        euler=(0, 0, 0),
        scale=container_scale,
        decimate=False,
        convexify=False,
        decompose_nonconvex=True,
        coacd_options=gs.options.CoacdOptions(),
        merge_submeshes_for_collision=True,
    ),
    material=gs.materials.Rigid(needs_coup=True),
)


########################## build ##########################
scene.build()
#import IPython; IPython.embed()

# Set initial robot position
ur5e.set_dofs_position([0, -np.pi/2, np.pi/2, 0, 0, 0])
ur5e.set_dofs_kp(
    kp             = np.array([2000, 1500, 1000, 1000, 50, 50]),
    dofs_idx_local = dofs_idx,
)

# Increase velocity gains for better damping
ur5e.set_dofs_kv(
    kv             = np.array([100, 100, 100, 100, 20, 20]), 
    dofs_idx_local = dofs_idx,
)

# Set force limits for safety
ur5e.set_dofs_force_range(
    np.array([-100, -100, -100, -100, -20, -20]),
    np.array([100, 100, 100, 100, 20, 20]),
    dofs_idx_local = dofs_idx,
)
set_liquid_pose(becher)
########################## main ##########################

# start camera recording. Once this is started, all the rgb images rendered will be recorded internally
cam.start_recording()

# Get initial end effector position for reference
end_effector = ur5e.get_link("wrist_3_link")
initial_pos = end_effector.get_pos().cpu().numpy()
initial_quat = end_effector.get_quat().cpu().numpy()

# Part 1 - Reach steady state of the liquid
for i in range(100):
    ur5e.control_dofs_position(
        np.array([0, (-np.pi/2), np.pi/2, 0, 0, 0])[1:],
        dofs_idx[1:],
    )
    scene.step()
    cam.render()
    if i % 10 == 0:
        print(f"Step {i}")

# Part 2 - Bring the bottle around
for i in range(200):
    # Calculate target position in horizontal motion
    distance = i * 0.001  # Slowly increasing distance
    
    # Target position in world frame
    target_pos = np.array([
        initial_pos[0], 
        initial_pos[1]+ distance,             
        initial_pos[2],            
    ])
    
    # Target orientation (keep container orientation)
    target_quat = initial_quat
    
    # Use inverse kinematics to get joint angles
    try:
        qpos = ur5e.inverse_kinematics(
            link=end_effector,
            pos=target_pos,
            quat=target_quat,
        )
        
        # Control joint positions with velocity limits
        ur5e.control_dofs_position(
            position=qpos,
            dofs_idx_local=dofs_idx,
        )
            
    except Exception as e:
        print(f"IK failed: {e}")
        continue
        
    scene.step()
    cam.render()
    if i % 10 == 0:
        print(f"Step {i+100}")

# Part 3 - Lower the arm
for i in range(100):
    # Calculate lowering motion
    height_offset = i * -0.001  # Slowly decreasing height
    
    # Target position in world frame 
    target_pos = np.array([
        initial_pos[0],
        initial_pos[1] + distance, # Keep final horizontal position
        initial_pos[2] + height_offset,
    ])
    
    # Keep initial orientation
    target_quat = initial_quat
    
    try:
        qpos = ur5e.inverse_kinematics(
            link=end_effector,
            pos=target_pos,
            quat=target_quat,
        )
        
        ur5e.control_dofs_position(
            position=qpos,
            dofs_idx_local=dofs_idx,
        )
            
    except Exception as e:
        print(f"IK failed: {e}")
        continue
        
    scene.step()
    cam.render()
    if i % 10 == 0:
        print(f"Step {i+300}")

# Part 4 - Rotate to pour SOLO L'END EFFECTOR (wrist_3_joint)
current_qpos = ur5e.get_dofs_position().cpu().numpy()
for i in range(150):
    # Calcola l'angolo di versamento
    pour_angle = i * (np.pi/33)  # Fino a ~90 gradi

    # Copia la posizione attuale dei giunti
    qpos = current_qpos.copy()
    # Modifica solo l'ultimo giunto (wrist_3_joint)
    qpos[dofs_idx[-1]] = pour_angle  # Muove solo l'end effector

    ur5e.control_dofs_position(
        position=qpos,
        dofs_idx_local=dofs_idx,
    )
    scene.step()
    cam.render()
    if i % 10 == 0:
        print(f"Step {i+400}")

current_qpos = ur5e.get_dofs_position().cpu().numpy()
for i in range(1000):
    ur5e.control_dofs_position(
        position=current_qpos,
        dofs_idx_local=dofs_idx,
    )
    scene.step()
    cam.render()
    if i % 10 == 0:
        print(f"Step {i+550}")


# particles = liquid.get_particles()
# stop recording and save video. If `filename` is not specified, a name will be auto-generated using the caller file name.
cam.stop_recording(save_to_filename='bringing_becher_around_2.mp4', fps=60)