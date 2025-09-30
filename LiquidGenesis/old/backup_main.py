import os
os.environ['PYOPENGL_PLATFORM'] = 'glx'
os.environ['MUJOCO_GL'] = 'glx'
import genesis as gs
import numpy as np
import trimesh
import random 
import torch
from ompl import util as ou
from ompl import base as ob
from ompl import geometric as og
 
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

def set_liquid_pose(becher): # da sistemare
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

def obtain_range():
    """
    #### guess gain e limiti (REDDIT, VA BENE?)
    # Posizione
    kp = [300, 300, 300, 150, 100, 80]  # N·m/rad o N/m
    # Velocità (smorzamento critico: kv = 2 * sqrt(kp * inertia))
    kv = [60, 60, 60, 40, 30, 20]       # N·m·s/rad o N·s/m
    max_torque = [150, 150, 150, 28, 28, 28]  # Nm
    max_velocity = [3.14] * 6                 # rad/s
    max_force = 150                           # N (sul tool)
    """

    parameters_range={
    "viscosità":[0.0008,0.0012], # range viscosità liquido
    "densità":[995.0,1001.0], # range densità liquido
    "tens_sup":[0.070,0.073], # range di tensione superficiale 
    "vol_init":[1e-5,5e-5], 
    "vol_target":[0.5e-5,1e-5],
    "delay_control":[10e-6,10e-6], # range delay di controllo
    "friction":[
        [0.01, 0.05],
        [0.01, 0.05],
        [0.01, 0.05],
        [0.01, 0.05],
        [0.01, 0.05],
        [0.01, 0.05],
        [0.01, 0.05],
        [0.01, 0.05],
        ], # range attrito links (da def per ogni giunto) TOGLIERE
    "pos_init_cont":[
        [0.8,0.9],
        [0.15,0.25],
        [0.92,0.92]
    ],
    "pos_init_ee":[
        [0.5,0.7],
        [0.10,0.3],
        [0.97, 1.1],
        [0.5,0.5],
        [0.5,0.5],
        [0.5,0.5],
        [0.5,0.5],
    ],
    "err_target":[5e-6,5e-6],
    "pos_cont_goal":[
        [0.8,0.9],
        [0.65,0.75],
        [0.92,0.92]
    ],
    "goal":[
        [0.5,0.7],
        [0.6,0.8],
        [0.97, 1.1],
        [0.5,0.5],
        [0.5,0.5],
        [0.5,0.5],
        [0.5,0.5],
    ],
    "kp":[
        [200, 400],
        [200, 400],
        [200, 400],
        [70, 230],
        [30,170],
        [40,120],
        [40,120],
        [40,120],
    ],
    "kv":[
        [20, 100],
        [20, 100],
        [20, 100],
        [20, 60],
        [20, 40],
        [10, 30],
        [5, 15],
        [5, 15],
    ],
    "max_F":[
        [50,150],
        [50,150],
        [50,150],
        [50,110],
        [50,110],
        [50,110],
        [50,150],
        [50,150],
    ],
    }
    return parameters_range

def generate_parameters(parameters_range):
    parameters = {}
    for key, value in parameters_range.items():
        if isinstance(value, list) and len(value) == 2 and all(isinstance(v, (int, float)) for v in value):
            # Singolo range: [min, max]
            parameters[key] = random.uniform(value[0], value[1])
        elif isinstance(value, list) and all(isinstance(v, list) and len(v) == 2 for v in value):
            # Lista di range: [[min, max], [min, max], ...]
            sampled_values = []
            for v in value:
                if v[0] == v[1]:
                    sampled_values.append(v[0])  # valore fisso
                else:
                    sampled_values.append(random.uniform(v[0], v[1]))
            parameters[key] = sampled_values
        else:
            # Caso non gestito o vuoto
            parameters[key] = None
    return parameters

def init_sim():
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

def generate_sim(parameters,view=False,video=False):    
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
    if video==True:
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
    link_idx=[]
    for link in ur5e.links:
        if link.name not in ["world","tool0", "hand_e_link"]: # rimuovi fixed links
            link_names.append(link.name)
            link_idx.append(link.idx_local)
    #print(link_names,link_idx)

    contpos= (parameters['pos_init_cont'][0],parameters['pos_init_cont'][1],parameters['pos_init_cont'][2]) # (0.85,0.2, 0.92) # Initial position
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

    contpos2= (parameters['pos_cont_goal'][0],parameters['pos_cont_goal'][1],parameters['pos_cont_goal'][2])
    container_scale2 = 0.013
    container_mesh_path2 = '/home/edo/thesis/becher/becher.obj'

    becher2 = scene.add_entity(
        gs.morphs.Mesh(
            file=container_mesh_path2,
            fixed=False,
            pos=contpos2,
            euler=(90, 0, 180),
            scale=container_scale2,
            decimate=False,
            convexify=False,
            decompose_nonconvex=True,
            coacd_options=gs.options.CoacdOptions(),
            merge_submeshes_for_collision=True,
        ),
        material=gs.materials.Rigid(needs_coup=True),
    )

    # Load and analyze container mesh
    container_mesh = trimesh.load(container_mesh_path)
    container_bounds = container_mesh.bounds
    container_size = container_bounds[1] - container_bounds[0]
    container_center = container_mesh.center_mass
    # Calculate liquid dimensions based on container size
    liquid_radius = min(container_size[0], container_size[1])*container_scale*1/np.sqrt(2)*0.9
    init_volume = parameters['vol_init']
    liquid_height = init_volume/(np.pi*liquid_radius**2)
    #liquid_height = container_size[2]*container_scale*np.sqrt(2)*0.5
    #print(liquid_radius, liquid_height)
    # Position liquid relative to container center
    liqpos = (contpos[0], contpos[1], contpos[2] + liquid_height/2)

    liquid = scene.add_entity(
        # viscous liquid
        #material=gs.materials.SPH.Liquid(mu=0.02, gamma=0.02),
        material=gs.materials.SPH.Liquid( 
            rho= parameters['densità'], # 1000.0
            stiffness=50000.0,
            exponent=7.0,
            mu= parameters['viscosità'], # 0.001002       # viscosità dinamica dell'acqua a 20 °C [Pa·s]
            gamma=parameters['tens_sup'], # 0.0728       # tensione superficiale dell'acqua a 20 °C [N/m]),
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
    #import IPython; IPython.embed()

    ########################## build ##########################
    scene.build()
    # Set dofs kp:
    ur5e.set_dofs_kp(
        kp = np.array([
            parameters['kp'][0],
            parameters['kp'][1],
            parameters['kp'][2],
            parameters['kp'][3],
            parameters['kp'][4],
            parameters['kp'][5],
            parameters['kp'][6],
            parameters['kp'][7],
        ]), # np.array([300, 300, 300, 150, 100, 80, 80, 80]),
        dofs_idx_local = dofs_idx,
    )
    # Set dofs kv: (Increase velocity gains for better damping)
    ur5e.set_dofs_kv(
        kv = np.array([
            parameters['kv'][0],
            parameters['kv'][1],
            parameters['kv'][2],
            parameters['kv'][3],
            parameters['kv'][4],
            parameters['kv'][5],
            parameters['kv'][6],
            parameters['kv'][7],
         ]), #np.array([60, 60, 60, 40, 30, 20, 10, 10]), 
        dofs_idx_local = dofs_idx,
    )
    # Set force limits:
    ur5e.set_dofs_force_range(
        np.array([
            -parameters['max_F'][0],
            -parameters['max_F'][1],
            -parameters['max_F'][2],
            -parameters['max_F'][3],
            -parameters['max_F'][4],
            -parameters['max_F'][5],
            -parameters['max_F'][6],
            -parameters['max_F'][7],
        ]), # np.array([-100, -100, -100, -80, -80, -80, -100, -100]),
        np.array([
            parameters['max_F'][0],
            parameters['max_F'][1],
            parameters['max_F'][2],
            parameters['max_F'][3],
            parameters['max_F'][4],
            parameters['max_F'][5],
            parameters['max_F'][6],
            parameters['max_F'][7],
        ]), #np.array([ 100,  100,  100,  80,  80,  80,  100,  100]),
        dofs_idx_local = dofs_idx,
    )
    
    ########################## TO DO #################
    # for i, link_idx in enumerate link_idx:
    #     ur5e.set_friction_ratio(parameters['friction'][i],link_i)
    
    ########################## main ##########################

    # # start camera recording. Once this is started, all the rgb images rendered will be recorded internally
    if video==True:
        cam.start_recording()

    # Reach steady state of the liquid
    # Set initial robot position
    end_effector = ur5e.get_link("hand_e_link")
    init_pos=np.array([parameters['pos_init_ee'][0], parameters['pos_init_ee'][1],parameters['pos_init_ee'][2]])
    init_quat=np.array([parameters['pos_init_ee'][3], parameters['pos_init_ee'][4], parameters['pos_init_ee'][5], parameters['pos_init_ee'][6]])
        
    # Use inverse kinematics to get joint angles
    init_qpos = ur5e.inverse_kinematics(
                link=end_effector,
                pos=init_pos,
                quat=init_quat,
        )
    ur5e.set_dofs_position(init_qpos)
    for i in range(100):
        ur5e.control_dofs_position(
            position=init_qpos,
            dofs_idx_local=dofs_idx,
        )
        scene.step()
        # cam.render()
    print("Steady state reached")
    
    return scene, ur5e, becher, becher2, liquid, dt

def reset_sim(scene, ur5e, becher, becher2, liquid):
    # Set initial robot position
    dofs_idx = []
    for joint in ur5e.joints:
        if joint.name not in ["joint_world","flange-tool0","robotiq_hande_base_joint"]:
            dofs_idx.append(joint.dof_idx_local)
    end_effector = ur5e.get_link("hand_e_link")
    init_pos=np.array([parameters['pos_init_ee'][0], parameters['pos_init_ee'][1],parameters['pos_init_ee'][2]])
    init_quat=np.array([parameters['pos_init_ee'][3], parameters['pos_init_ee'][4], parameters['pos_init_ee'][5], parameters['pos_init_ee'][6]])
    init_qpos = ur5e.inverse_kinematics(
                link=end_effector,
                pos=init_pos,
                quat=init_quat,
        )
    ur5e.set_dofs_position(init_qpos)

    contpos= (parameters['pos_init_cont'][0],parameters['pos_init_cont'][1],parameters['pos_init_cont'][2]) # (0.85,0.2, 0.92) # Initial position, will be updated after build
    becher.set_pos(contpos)
    contpos2= (parameters['pos_cont_goal'][0],parameters['pos_cont_goal'][1],parameters['pos_cont_goal'][2])
    becher2.set_pos(contpos2)

    set_liquid_pose(becher)

    for i in range(100):
        ur5e.control_dofs_position(
            position=init_qpos,
            dofs_idx_local=dofs_idx,
        )
        scene.step()
        # cam.render()
    print("Steady state reached")

def tensor_to_cpu(x):
    if isinstance(x, torch.Tensor):
        x = x.cpu()
    return x

def tensor_to_array(x):
    return np.array(tensor_to_cpu(x))

# to be updated for full path (grasping becher + motion + final rotation)
def plan_path(
        entity,
        cont,
        qpos_goal,
        qpos_start=None,
        timeout=5.0,
        smooth_path=True,
        num_waypoints=100,
        ignore_collision=False,
        ignore_joint_limit=False,
        planner="RRTConnect",
    ):
        """
        Plan a path from `qpos_start` to `qpos_goal`.

        Parameters
        ----------
        qpos_goal : array_like
            The goal state.
        qpos_start : None | array_like, optional
            The start state. If None, the current state of the rigid entity will be used. Defaults to None.
        timeout : float, optional
            The maximum time (in seconds) allowed for the motion planning algorithm to find a solution. Defaults to 5.0.
        smooth_path : bool, optional
            Whether to smooth the path after finding a solution. Defaults to True.
        num_waypoints : int, optional
            The number of waypoints to interpolate the path. If None, no interpolation will be performed. Defaults to 100.
        ignore_collision : bool, optional
            Whether to ignore collision checking during motion planning. Defaults to False.
        ignore_joint_limit : bool, optional
            Whether to ignore joint limits during motion planning. Defaults to False.
        planner : str, optional
            The name of the motion planning algorithm to use. Supported planners: 'PRM', 'RRT', 'RRTConnect', 'RRTstar', 'EST', 'FMT', 'BITstar', 'ABITstar'. Defaults to 'RRTConnect'.

        Returns
        -------
        waypoints : list
            A list of waypoints representing the planned path. Each waypoint is an array storing the entity's qpos of a single time step.
        """

        if qpos_start is None:
            qpos_start = entity.get_qpos()
        qpos_start = tensor_to_array(qpos_start)
        qpos_goal = tensor_to_array(qpos_goal)

        if qpos_start.shape != (entity.n_qs,) or qpos_goal.shape != (entity.n_qs,):
            gs.raise_exception("Invalid shape for `qpos_start` or `qpos_goal`.")
    
        ######### process joint limit ##########
        if ignore_joint_limit:
            q_limit_lower = np.full_like(entity.q_limit[0], -1e6)
            q_limit_upper = np.full_like(entity.q_limit[1], 1e6)
        else:
            q_limit_lower = entity.q_limit[0]
            q_limit_upper = entity.q_limit[1]

        if (qpos_start < q_limit_lower).any() or (qpos_start > q_limit_upper).any():
            gs.logger.warning(
                "`qpos_start` exceeds joint limit. Relaxing joint limit to contain `qpos_start` for planning."
            )
            q_limit_lower = np.minimum(q_limit_lower, qpos_start)
            q_limit_upper = np.maximum(q_limit_upper, qpos_start)

        if (qpos_goal < q_limit_lower).any() or (qpos_goal > q_limit_upper).any():
            gs.logger.warning(
                "`qpos_goal` exceeds joint limit. Relaxing joint limit to contain `qpos_goal` for planning."
            )
            q_limit_lower = np.minimum(q_limit_lower, qpos_goal)
            q_limit_upper = np.maximum(q_limit_upper, qpos_goal)
        
        ######### setup OMPL ##########
        ou.setLogLevel(ou.LOG_ERROR)
        space = ob.RealVectorStateSpace(entity.n_qs)
        bounds = ob.RealVectorBounds(entity.n_qs)

        for i_q in range(entity.n_qs):
            bounds.setLow(i_q, q_limit_lower[i_q])
            bounds.setHigh(i_q, q_limit_upper[i_q])
        space.setBounds(bounds)
        ss = og.SimpleSetup(space)

        geoms_idx = list(range(entity._geom_start, entity._geom_start + len(entity._geoms)))
        mask_collision_pairs = set(
            (i_ga, i_gb) for i_ga, i_gb in entity.detect_collision() if i_ga in geoms_idx or i_gb in geoms_idx
        )
        if not ignore_collision and mask_collision_pairs:
            gs.logger.info("Ingoring collision pairs already active for starting pos.")

        def is_ompl_state_valid(state):
            if ignore_collision:
                return True
            qpos = torch.tensor([state[i] for i in range(entity.n_qs)], dtype=gs.tc_float, device=gs.device)
            entity.set_qpos(qpos, zero_velocity=False)
            collision_pairs = set(map(tuple, entity.detect_collision()))
            return not (collision_pairs - mask_collision_pairs)

        ss.setStateValidityChecker(ob.StateValidityCheckerFn(is_ompl_state_valid))

        try:
            planner_cls = getattr(og, planner)
            if not issubclass(planner_cls, ob.Planner):
                raise ValueError
            ss.setPlanner(planner_cls(ss.getSpaceInformation()))
        except (AttributeError, ValueError) as e:
            gs.raise_exception_from(f"'{planner}' is not a valid planner. See OMPL documentation for details.", e)

        state_start = ob.State(space)
        state_goal = ob.State(space)
        for i_q in range(entity.n_qs):
            state_start[i_q] = float(qpos_start[i_q])
            state_goal[i_q] = float(qpos_goal[i_q])
        ss.setStartAndGoalStates(state_start, state_goal)

        ######### solve ##########
        solved = ss.solve(timeout)
        waypoints = []
        if solved:
            gs.logger.info("Path solution found successfully.")
            path = ss.getSolutionPath()
            if smooth_path:
                ps = og.PathSimplifier(ss.getSpaceInformation())
                # simplify the path
                try:
                    ps.partialShortcutPath(path)
                    ps.ropeShortcutPath(path)
                except:
                    ps.shortcutPath(path)
                ps.smoothBSpline(path)

            if num_waypoints is not None:
                path.interpolate(num_waypoints)
            waypoints = [
                torch.as_tensor([state[i] for i in range(entity.n_qs)], dtype=gs.tc_float, device=gs.device)
                for state in path.getStates()
            ]
        else:
            gs.logger.warning("Path planning failed. Returning empty path.")

        ########## restore original state #########
        entity.set_qpos(qpos_start, zero_velocity=False)

        return waypoints

def simulate_action(ur5e, parameters, path): 
    ############# Reset env ################
    # Reach steady state of the liquid
    # Set robot position
    dofs_idx = []
    for joint in ur5e.joints:
        if joint.name not in ["joint_world","flange-tool0","robotiq_hande_base_joint"]:
            dofs_idx.append(joint.dof_idx_local) 
    end_effector = ur5e.get_link("hand_e_link")
    init_pos=np.array([parameters['pos_init_ee'][0], parameters['pos_init_ee'][1],parameters['pos_init_ee'][2]])
    init_quat=np.array([parameters['pos_init_ee'][3], parameters['pos_init_ee'][4], parameters['pos_init_ee'][5], parameters['pos_init_ee'][6]])
    init_qpos = ur5e.inverse_kinematics(
                link=end_effector,
                pos=init_pos,
                quat=init_quat,
        )
    ur5e.set_dofs_position(init_qpos)
    for i in range(100):
        ur5e.control_dofs_position(
            position=init_qpos,
            dofs_idx_local=dofs_idx,
        )
        scene.step()
        # cam.render()
    print("Steady state reached")

    # Esegui il path
    score=0
    for qpos in path:
        ur5e.control_dofs_position(qpos, dofs_idx_local=dofs_idx) # bisogna aggiungere il delay di controllob (delay_control)
        particles = liquid.get_particles().cpu().numpy()
        for particle in particles:
            if particle[2] < parameters['pos_init_cont'][2]: # da cambiare con un collision detection
                score-=5/len(particles) # to be tuned
                
        scene.step()

    # Valuta successo
    particles = liquid.get_particles().cpu().numpy()
    contpos = np.array(parameters['pos_cont_goal'])
    err = parameters['err_target']
    target_vol=parameters['vol_target']

    ck1 = abs(np.mean(particles[:, 0])-contpos[0])< err # err x
    ck2 = abs(np.mean(particles[:, 1])-contpos[1])< err # err y
    ck3 = abs(np.mean(particles[:, 2])-contpos[2])< err # err z
    if ck1 and ck2 and ck3:
        score+=4/len(particles) # to be tuned

    mask = (
        (np.abs(particles[:, 0] - contpos[0]) < err) &
        (np.abs(particles[:, 1] - contpos[1]) < err) &
        (np.abs(particles[:, 2] - contpos[2]) < err)
    )
    num_particles_in_target = np.sum(mask)
    vol=num_particles_in_target*liquid.particle_size
    if abs(vol-target_vol)<err:
        score+=1 # to be tuned

    return score

def is_success(score, threshold=0.5):
    return score > threshold

def update_parameters(param, scale=0.1): # Aggiunta rumore gaussiano ai parametri
    new_param = {}
    for key, val in param.items():
        if isinstance(val, float):
            noise = np.random.normal(0, scale * abs(val))
            new_param[key] = val + noise
        elif isinstance(val, list) and all(isinstance(v, float) for v in val):
            new_param[key] = [v + np.random.normal(0, scale * abs(v)) for v in val]
        else:
            new_param[key] = val
    return new_param


########################## main ##########################

N = 20                    # Numero di modelli simulati
M = 5                     # Numero di traiettorie
delta = 0.7*M             # Threshold di successo 
MAX_ITERS = 10            # Numero massimo di iterazioni

parameters_range=obtain_range()
param_real = generate_parameters(parameters_range)  # Simulazione della realtà

init_sim()

parameters_set=[]
for n in range(N):
    parameters = generate_parameters(parameters_range) # ottiene un dizionario di parametri in modo randomico sui range iniziali dati
    scene, ur5e, becher, becher2, liquid, dt = generate_sim(parameters,True) # genera l'ambiente di simulazione
    parameters_set.append(parameters)

    q_start = ur5e.get_dofs_position().cpu().numpy() # posizione iniziale (definita attraverso i parametri in generate_sim)
    goal_pos = np.array([parameters['goal'][0],parameters['goal'][1],parameters['goal'][2]]) # posizione del goal
    goal_quat = np.array([parameters['goal'][3],parameters['goal'][4],parameters['goal'][5],parameters['goal'][6]]) # orientazione del goal

    try:
        q_goal = ur5e.inverse_kinematics(
            link=ur5e.get_link("hand_e_link"),
            pos=goal_pos,
            quat=goal_quat
        )
    except Exception as e:
        print(f"errore nella IK goal")
        continue
                
    for t in range(MAX_ITERS):
        print(f"\n[ITER {t}] Belief size: {len(parameters)}")

        candidate_paths = []
        for i in range(M):
            try:
                # path = plan_path(
                #     ur5e,
                #     becher,
                #     qpos_goal=q_goal,
                #     qpos_start=q_start,
                #     timeout=5.0,
                #     num_waypoints=100+50*i, # per ottenere traiettorie diverse: diversi tempi di percorrenza di traj simili
                #     ignore_collision=False,
                #     planner="RRT*"
                # )

                path, is_invalid = ur5e.plan_path(
                    qpos_goal=q_goal,
                    qpos_start=q_start,
                    timeout=5.0,
                    num_waypoints=100+50*i, # per ottenere traiettorie diverse: diversi tempi di percorrenza di traj simili
                    ignore_collision=False,
                    planner="RRTstar"
                )

                if not is_invalid:  # se valido
                    candidate_paths.append(path)
            except Exception as e:
                print(f"Errore nel planning: {e}")
                continue

    # Valuta ogni traiettoria su ogni set di param
    best_path = None
    best_score = -1
    for path in candidate_paths:
        score = sum(simulate_action(ur5e, parameters, path) for parameters in parameters_set)
        if score > best_score:
            best_score = score
            best_path = path

    if best_score < delta:
        print("Nessuna traiettoria soddisfa il delta succ")
        break
    else:
        print("Esiste traj che soddisfa req succ")

    # Esegui su realtà (simulata con param_real)
    real_score = simulate_action(ur5e, param_real, best_path)
    real_result = is_success(real_score)
    #print(f"Risultato nel mondo reale (parameters*): {real_score}")

    # Aggiorna belief set
    param_new = [p for p in parameters if is_success(simulate_action(ur5e, p, best_path)) == real_result]

    if len(param_new) == 0:
            print("Tutte le ipotesi eliminate.")
            break
    
    # Resample
    new_samples = [update_parameters(p) for p in param_new]
    parameters = param_new + new_samples

    max_models = 30
    if len(parameters) > max_models:
        parameters = random.sample(parameters, max_models)


       


#da mettere nel planner!!!

# # Get becher position
# contpos=becher.get_pos()
# # Get end effector position
# end_effector = ur5e.get_link("hand_e_link")

# target_pos = np.array([
#         contpos[0]-0.25, 
#         contpos[1],             
#         contpos[2]+0.05,            
#     ])
# # Target orientation
# target_quat=np.array([0.5, 0.5, 0.5, 0.5])
# # Use inverse kinematics to get joint angles
# qpos = ur5e.inverse_kinematics(
#         link=end_effector,
#         pos=target_pos,
#         quat=target_quat,
# )
# # Plan the path
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
