import os

import ompl.base
import ompl.geometric
import ompl.util
os.environ['PYOPENGL_PLATFORM'] = 'glx'
os.environ['MUJOCO_GL'] = 'glx'
import genesis as gs
import numpy as np
import trimesh
import random 
import torch 
# import ompl
# from ompl import base as ob
# from ompl import geometric as og
# from ompl import util as ou

####################### functions #######################
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

def generate_parameters(parameters_range):
    parameters = {}
    for key, value in parameters_range.items():
        if isinstance(value, list) and len(value) == 2 and all(isinstance(v, (int, float)) for v in value):
            # If the range is a list of two numbers, sample uniformly in the range
            parameters[key] = random.uniform(value[0], value[1])
        else:
            # If the range is not specified, set to None or empty
            parameters[key] = None
    return parameters

# def tensor_to_cpu(x):
#     if isinstance(x, torch.Tensor):
#         x = x.cpu()
#     return x

# def tensor_to_array(x):
#     return np.array(tensor_to_cpu(x))

# def plan_path1(
#         entity,
#         qpos_goal,
#         qpos_start=None,
#         timeout=5.0,
#         smooth_path=True,
#         num_waypoints=100,
#         ignore_collision=False,
#         ignore_joint_limit=False,
#         planner="RRTConnect",
#     ):
#         """
#         Plan a path from `qpos_start` to `qpos_goal`.

#         Parameters
#         ----------
#         qpos_goal : array_like
#             The goal state.
#         qpos_start : None | array_like, optional
#             The start state. If None, the current state of the rigid entity will be used. Defaults to None.
#         timeout : float, optional
#             The maximum time (in seconds) allowed for the motion planning algorithm to find a solution. Defaults to 5.0.
#         smooth_path : bool, optional
#             Whether to smooth the path after finding a solution. Defaults to True.
#         num_waypoints : int, optional
#             The number of waypoints to interpolate the path. If None, no interpolation will be performed. Defaults to 100.
#         ignore_collision : bool, optional
#             Whether to ignore collision checking during motion planning. Defaults to False.
#         ignore_joint_limit : bool, optional
#             Whether to ignore joint limits during motion planning. Defaults to False.
#         planner : str, optional
#             The name of the motion planning algorithm to use. Supported planners: 'PRM', 'RRT', 'RRTConnect', 'RRTstar', 'EST', 'FMT', 'BITstar', 'ABITstar'. Defaults to 'RRTConnect'.

#         Returns
#         -------
#         waypoints : list
#             A list of waypoints representing the planned path. Each waypoint is an array storing the entity's qpos of a single time step.
#         """

#         if qpos_start is None:
#             qpos_start = entity.get_qpos()
#         qpos_start = tensor_to_array(qpos_start)
#         qpos_goal = tensor_to_array(qpos_goal)

#         if qpos_start.shape != (entity.n_qs,) or qpos_goal.shape != (entity.n_qs,):
#             gs.raise_exception("Invalid shape for `qpos_start` or `qpos_goal`.")
    
#         ######### process joint limit ##########
#         if ignore_joint_limit:
#             q_limit_lower = np.full_like(entity.q_limit[0], -1e6)
#             q_limit_upper = np.full_like(entity.q_limit[1], 1e6)
#         else:
#             q_limit_lower = entity.q_limit[0]
#             q_limit_upper = entity.q_limit[1]

#         if (qpos_start < q_limit_lower).any() or (qpos_start > q_limit_upper).any():
#             gs.logger.warning(
#                 "`qpos_start` exceeds joint limit. Relaxing joint limit to contain `qpos_start` for planning."
#             )
#             q_limit_lower = np.minimum(q_limit_lower, qpos_start)
#             q_limit_upper = np.maximum(q_limit_upper, qpos_start)

#         if (qpos_goal < q_limit_lower).any() or (qpos_goal > q_limit_upper).any():
#             gs.logger.warning(
#                 "`qpos_goal` exceeds joint limit. Relaxing joint limit to contain `qpos_goal` for planning."
#             )
#             q_limit_lower = np.minimum(q_limit_lower, qpos_goal)
#             q_limit_upper = np.maximum(q_limit_upper, qpos_goal)
#         ou=ompl.util
#         ob=ompl.base
#         og=ompl.geometric
#         ######### setup OMPL ##########
#         ou.setLogLevel(ou.LOG_ERROR)
#         space = ob.RealVectorStateSpace(entity.n_qs)
#         bounds = ob.RealVectorBounds(entity.n_qs)

#         for i_q in range(entity.n_qs):
#             bounds.setLow(i_q, q_limit_lower[i_q])
#             bounds.setHigh(i_q, q_limit_upper[i_q])
#         space.setBounds(bounds)
#         ss = og.SimpleSetup(space)

#         geoms_idx = list(range(entity._geom_start, entity._geom_start + len(entity._geoms)))
#         mask_collision_pairs = set(
#             (i_ga, i_gb) for i_ga, i_gb in entity.detect_collision() if i_ga in geoms_idx or i_gb in geoms_idx
#         )
#         if not ignore_collision and mask_collision_pairs:
#             gs.logger.info("Ingoring collision pairs already active for starting pos.")

#         def is_ompl_state_valid(state):
#             if ignore_collision:
#                 return True
#             qpos = torch.tensor([state[i] for i in range(entity.n_qs)], dtype=gs.tc_float, device=gs.device)
#             entity.set_qpos(qpos, zero_velocity=False)
#             collision_pairs = set(map(tuple, entity.detect_collision()))
#             return not (collision_pairs - mask_collision_pairs)

#         ss.setStateValidityChecker(ob.StateValidityCheckerFn(is_ompl_state_valid))

#         try:
#             planner_cls = getattr(og, planner)
#             if not issubclass(planner_cls, ob.Planner):
#                 raise ValueError
#             ss.setPlanner(planner_cls(ss.getSpaceInformation()))
#         except (AttributeError, ValueError) as e:
#             gs.raise_exception_from(f"'{planner}' is not a valid planner. See OMPL documentation for details.", e)

#         state_start = ob.State(space)
#         state_goal = ob.State(space)
#         for i_q in range(entity.n_qs):
#             state_start[i_q] = float(qpos_start[i_q])
#             state_goal[i_q] = float(qpos_goal[i_q])
#         ss.setStartAndGoalStates(state_start, state_goal)

#         ######### solve ##########
#         solved = ss.solve(timeout)
#         waypoints = []
#         if solved:
#             gs.logger.info("Path solution found successfully.")
#             path = ss.getSolutionPath()
#             if smooth_path:
#                 ps = og.PathSimplifier(ss.getSpaceInformation())
#                 # simplify the path
#                 try:
#                     ps.partialShortcutPath(path)
#                     ps.ropeShortcutPath(path)
#                 except:
#                     ps.shortcutPath(path)
#                 ps.smoothBSpline(path)

#             if num_waypoints is not None:
#                 path.interpolate(num_waypoints)
#             waypoints = [
#                 torch.as_tensor([state[i] for i in range(entity.n_qs)], dtype=gs.tc_float, device=gs.device)
#                 for state in path.getStates()
#             ]
#         else:
#             gs.logger.warning("Path planning failed. Returning empty path.")

#         ########## restore original state #########
#         entity.set_qpos(qpos_start, zero_velocity=False)

#         return waypoints

####################### parameters #######################
parameters_range={
"viscosità":[0.0008,0.0012], # range viscosità liquido
"densità":[995.0,1001.0], # range densità liquido
"tens_sup":[0.070,0.073], # range di tensione superficiale 
"vol_init":[],
"vol_target":[],
"delay_control":[0,50e-6], # range delay di controllo
"friction":[0.01,0.05], # range attrito giunti (da def per ogni giunto)
"posizione_iniziale":[],
"err_pos_target":[-0.01,0.01],
"kp":[50,200],
"kv":[10,50],
"max_F":[10,100],
}

parameters = generate_parameters(parameters_range)

#### guess gain e limiti

# Posizione
kp = [300, 300, 300, 150, 100, 80]  # N·m/rad o N/m
# Velocità (smorzamento critico: kv = 2 * sqrt(kp * inertia))
kv = [60, 60, 60, 40, 30, 20]       # N·m·s/rad o N·s/m
max_torque = [150, 150, 150, 28, 28, 28]  # Nm
max_velocity = [3.14] * 6                 # rad/s
max_force = 150                           # N (sul tool)

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
dt=1e-3
scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt=dt,
        substeps=100,  # Increased substeps for better stability
        gravity=(0, 0, -9.81),
    ),
    sph_options=gs.options.SPHOptions(
        # position of the bounding box for the liquid
        lower_bound   = (-1.5, -1.5, 0.0), 
        upper_bound   = (1.5, 1.5, 2),
        particle_size = 0.02,  
    ),
    viewer_options = gs.options.ViewerOptions(
        res           = (1280, 960),
        camera_pos    = (3.5, 0.0, 2.5),
        camera_lookat = (0.0, 0.0, 0.5),
        camera_fov    = 40,
        max_FPS       = 60,
    ),
    vis_options = gs.options.VisOptions(
        show_world_frame = False, # visualize the coordinate frame of `world` at its origin
        world_frame_size = 0.5, # length of the world frame in meter
        show_link_frame  = False, # do not visualize coordinate frames of entity links
        show_cameras     = False, # do not visualize mesh and frustum of the cameras added
        plane_reflection = False, # turn on plane reflection
        ambient_light    = (0.1, 0.1, 0.1), # ambient light setting
        shadow=False,
    ),
    show_viewer = False,
    renderer = gs.renderers.Rasterizer(), # using rasterizer for camera rendering
    show_FPS = False,
    #renderer=gs.renderers.RayTracer()
)
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
        file = '/home/edo/thesis/ur5e_urdf/urdf/ur5e_complete.urdf',
        fixed=True,
        pos   = (0, 0, 0),
        euler = (0, 0, 0), # we follow scipy's extrinsic x-y-z rotation convention, in degrees,
        # quat  = (1.0, 0.0, 0.0, 0.0), # we use w-x-y-z convention for quaternions,
        scale = 1.0,
        links_to_keep=['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint', 'hand_e_link','hande_left_finger_joint', 'hande_right_finger_joint','tool0'],
    ),
    material=gs.materials.Rigid(),

)

jnt_names = []
dofs_idx = []
for joint in ur5e.joints:
    if joint.name not in ["joint_world","flange-tool0","robotiq_hande_base_joint"]:
        jnt_names.append(joint.name)
        dofs_idx.append(joint.dof_idx_local)       
#print(jnt_names)
#print(dofs_idx)
link_names=[]
for link in ur5e.links:
    link_names.append(link.name)
#print(link_names)

contpos=(0.85,0.2, 0.92)  # Initial position, will be updated after build
container_scale = 0.013
container_mesh_path = '/home/edo/thesis/becher/becher.obj'

becher = scene.add_entity(
    gs.morphs.Mesh(
        file=container_mesh_path,
        fixed=False,
        pos=contpos,
        euler=(90, 0, 180),
        scale=container_scale,
        decimate=False,
        convexify=False,
        decompose_nonconvex=True,
        coacd_options=gs.options.CoacdOptions(),
        merge_submeshes_for_collision=True,
    ),
    material=gs.materials.Rigid(needs_coup=True),
)
for link in becher.links:
    link_becher = link.name


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

# enter IPython's interactive mode for debug
# import IPython; IPython.embed()

########################## build ##########################
scene.build()

ur5e.set_dofs_kp(
    kp             = np.array([300, 300, 300, 150, 100, 80, 80, 80]),
    dofs_idx_local = dofs_idx,
)

# Increase velocity gains for better damping
ur5e.set_dofs_kv(
    kv             = np.array([60, 60, 60, 40, 30, 20, 10, 10]), 
    dofs_idx_local = dofs_idx,
)

# Set force limits for safety
ur5e.set_dofs_force_range(
    np.array([-100, -100, -100, -80, -80, -80, -100, -100]),
    np.array([ 100,  100,  100,  80,  80,  80,  100,  100]),
    dofs_idx_local = dofs_idx,
)
# ########################## main ##########################

# # start camera recording. Once this is started, all the rgb images rendered will be recorded internally
# cam.start_recording()

# Reach steady state of the liquid
# Set initial robot position
end_effector = ur5e.get_link("hand_e_link")
init_pos=np.array([0.6, contpos[1], 0.97]) # da modificare con param set
init_quat=np.array([0.5, 0.5, 0.5, 0.5]) # idem
    
# Use inverse kinematics to get joint angles
init_qpos = ur5e.inverse_kinematics(
            link=end_effector,
            pos=init_pos,
            quat=init_quat,
    )
ur5e.set_dofs_position(init_qpos)
for i in range(10):
    ur5e.control_dofs_position(
        position=init_qpos,
        dofs_idx_local=dofs_idx,
    )
    scene.step()
    # cam.render()
print("Steady state reached")

# Part 2 - Bring the bottle around
# Get end effector position
end_effector = ur5e.get_link("hand_e_link")

target_pos = np.array([
        contpos[0]-0.25, 
        contpos[1],             
        contpos[2]+0.05,            
    ])
# Target orientation
target_quat=np.array([0.5, 0.5, 0.5, 0.5])
# Use inverse kinematics to get joint angles
qpos = ur5e.inverse_kinematics(
        link=end_effector,
        pos=target_pos,
        quat=target_quat,
)
print("banana")
# Plan the path
# path = plan_path1(ur5e,
#     qpos_start         = None, # None = current state
#     qpos_goal          = qpos,
#     num_waypoints      = 20/dt, # 2s duration
#     smooth_path        = True,
#     ignore_collision   = False,
#     ignore_joint_limit = False,
#     planner            = "RRTConnect",
# )
# print(path)
# # execute the planned path
# for waypoint in path:
#     ur5e.control_dofs_position(waypoint)
#     scene.step()

# # allow robot to reach the last waypoint
# for i in range(100):
#     scene.step()


