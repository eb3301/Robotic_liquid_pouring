import os
clear = lambda: os.system('clear')
clear()
import rclpy
from rclpy.node import Node
from rclpy.logging import get_logger
import yaml
import numpy as np
import genesis as gs
import trimesh
import random 
import torch
from scipy.spatial.transform import Rotation as R
from interfaces.srv import Simplan
import sys
from scipy.special import expit

def progress_bar(i, total, msg, length=30):
    percent = (i + 1) / total
    bar = '#' * int(percent * length)
    sys.stdout.write(f"\r[{bar:<{length}}] {percent*100:5.1f}% [{msg}]")
    sys.stdout.flush()

def to_device_tensor(x):
    """Converte in torch.Tensor solo se Genesis usa GPU."""
    if gs.backend == gs.gpu:
        if isinstance(x, np.ndarray):
            return torch.as_tensor(x, dtype=torch.float32, device="cuda")
        elif isinstance(x, torch.Tensor):
            return x.to("cuda", dtype=torch.float32)
    else:
        if isinstance(x, np.ndarray):
            return torch.as_tensor(x, dtype=torch.float32)
        elif isinstance(x, torch.Tensor):
            return x.to("cpu", dtype=torch.float32)
    return torch.tensor(x, dtype=torch.float32)

def to_numpy_cpu(x):
    """Converte tensor → numpy solo se è su GPU."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x, dtype=np.float32)

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

def quat_inverse(q):
    """
    Inversa di un quaternione unitario.
    q: [w, x, y, z]
    """
    w, x, y, z = q
    return np.array([w, -x, -y, -z], dtype=np.float32)

def quat_multiply(q1, q2):
    """
    Moltiplicazione di due quaternioni.
    q1, q2: [w, x, y, z]
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.array([w, x, y, z], dtype=np.float32)

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
        backend             = gs.gpu,
        theme               = 'dark',
        logger_verbose_time = 'warning',
        performance_mode=True,
    )

def generate_sim(parameters, view=False, liq=True, debug=False, video=False, approach=False):    
    ########################## create a scene ##########################
    DIR="/home/barutta/Robotic_liquid_pouring"
    dt=1e-2
    global scene
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=dt,
            substeps= 100,  # Increased substeps for better stability
            gravity=(0, 0, -9.81),
        ),
        rigid_options=gs.options.RigidOptions(
        enable_collision=True,
        enable_self_collision=True,
        enable_adjacent_collision=False,
        # constraint_timeconst=0.0001,
        # max_dynamic_constraints=10,
        ),
        sph_options=gs.options.SPHOptions(
            # position of the bounding box for the liquid
            lower_bound   = (0.5, 0.0, 0.9), 
            upper_bound   = (1.0, 1.2, 1.3),
            particle_size = 0.01, #0.002  
        ),
        viewer_options = gs.options.ViewerOptions(
            res           = (640, 480),
            camera_pos    = (2.5, 0.55, 1.5),
            camera_lookat = (0.8, 0.55, 1.0),
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
            visualize_sph_boundary=True,
        ),
        show_viewer = view,
        renderer = gs.renderers.Rasterizer(), # using rasterizer for camera rendering
        profiling_options = gs.options.ProfilingOptions(show_FPS = False),
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
    # mat_rigid = gs.materials.Rigid(coup_friction=0.1,
    #                                coup_softness=0.0001,
    #                                coup_restitution=0.001,
    #                                sdf_cell_size=0.0001,
    #                                sdf_min_res=32,
    #                                sdf_max_res=512)
    
    plane = scene.add_entity(gs.morphs.Plane())

    ur5e=scene.add_entity(gs.morphs.URDF(
            file = DIR + '/ur5e_urdf/urdf/ur5e_complete.urdf',
            fixed=True,
            collision=True,
            visualization=True,
            pos   = (0, 0, 0),
            euler = (0, 0, 0), # we follow scipy's extrinsic x-y-z rotation convention, in degrees,
            # quat  = (1.0, 0.0, 0.0, 0.0), # we use w-x-y-z convention for quaternions,
            decimate=False,
            # convexify=True,
            # decompose_robot_error_threshold=0.0,
            # contype=0b001,
            # conaffinity=0b001,
            scale = 1.0,
            links_to_keep=['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint', 'hand_e_link','hande_left_finger_joint', 'hande_right_finger_joint','tool0'],
        ),
        material=gs.materials.Rigid(),
        # vis_mode = "collision",
        visualize_contact=debug,
    )
    jnt_names = []
    dofs_idx = []
    for joint in ur5e.joints:
        if joint.name not in ["joint_world","flange-tool0","robotiq_hande_base_joint"]:
            jnt_names.append(joint.name)
            dofs_idx.extend(joint.dofs_idx_local)
            # dofs_idx.append(joint.dof_idx_local)

    link_names=[]
    link_idx=[]
    for link in ur5e.links:
        #if link.name not in ["world","tool0", "hand_e_link"]: # rimuovi fixed links
            link_names.append(link.name)
            link_idx.append(link.idx_local)
    if debug:
        print(f"joint names: {jnt_names}")
        print(f"joint indexes: {dofs_idx}")
        print(f"link names {link_names}, link indexes: {link_idx}")

    plane1 = scene.add_entity(gs.morphs.Plane(pos=(0,0,0.92), visualization=False))

    # Real z position is for sure in contact with the table
    parameters['pos_cont_goal'][2] = 0.92
    parameters['pos_init_cont'][2] = 0.92

    if approach:
        contpos= (parameters['pos_init_cont'][0],parameters['pos_init_cont'][1],parameters['pos_init_cont'][2]) # (0.85,0.2, 0.92) # Initial position
        container_scale = 0.016
        container_mesh_path = DIR + '/becher/becher1.obj'

        becher = scene.add_entity(
            gs.morphs.Mesh(
                file=container_mesh_path,
                fixed=False,
                pos=contpos,
                euler=(90, 0, 180),
                scale=container_scale,
                decimate=False,
                convexify=True,
                decompose_object_error_threshold=0.0,
                # decompose_nonconvex=True,
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
    else:
        contpos= (parameters['offset'][0], -parameters['offset'][2], parameters['offset'][1]) # ordine y, (-)z, x # np.array([0.0,-0.04,0.13]) # Offset di presa tool0 --> becher
        container_scale = 0.016
        container_mesh_path = DIR + '/becher/becher1.obj'

        becher = scene.add_entity(
            gs.morphs.Mesh(
                file=container_mesh_path,
                fixed=True,
                pos=contpos,
                euler=(180, 0, 0),
                scale=container_scale,
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


    for link in becher.links:
        link_becher = link.name
    
    # Load and analyze container mesh
    container_mesh = trimesh.load(container_mesh_path)
    container_bounds = container_mesh.bounds
    global container_size
    container_size = (container_bounds[1] - container_bounds[0])*container_scale
    #container_center = container_mesh.center_mass

    contpos2= (parameters['pos_cont_goal'][0],parameters['pos_cont_goal'][1], parameters['pos_cont_goal'][2])
    container_scale2 = 0.013
    container_mesh_path2 = DIR + '/becher/becher1.obj'

    becher2 = scene.add_entity(
        gs.morphs.Mesh(
            file=container_mesh_path2,
            fixed=False,
            pos=contpos2,
            euler=(90, 0, 180),
            scale=container_scale2,
            decimate=False,
            convexify=True,
            decompose_object_error_threshold=0.0,
            # decompose_nonconvex=True,
            # contype=0b011,
            # conaffinity=0b011,
            coacd_options=gs.options.CoacdOptions(),
            merge_submeshes_for_collision=True,
        ),
        material=gs.materials.Rigid(needs_coup=True),
    )

    if debug:
        print(f"ur5e - geom start: {ur5e.geom_start} - geom end: {ur5e.geom_end}")
        print(f"becher - geom start: {becher.geom_start} - geom end: {becher.geom_end}")
        print(f"becher2 - geom start: {becher2.geom_start} - geom end: {becher2.geom_end}")
    
    # Load and analyze container 2 mesh
    container2_mesh = trimesh.load(container_mesh_path2)
    container2_bounds = container2_mesh.bounds
    global container2_size
    container2_size = (container2_bounds[1] - container2_bounds[0])*container_scale2
    
    # Calculate liquid dimensions based on container size
    liquid_radius = min(container_size[0], container_size[1])/2*0.7
    init_volume = parameters['vol_init'] if parameters['vol_init']<1 else parameters['vol_init']*1e-6 
    liquid_height = init_volume/(np.pi*liquid_radius**2)
    num_part=init_volume/(0.01**3*0.7) #vol/(part_size^3*efficiency)

    print(f"Init Volume: {init_volume}")
    print(f"Radius: {liquid_radius*10**3} mm, Height: {liquid_height*10**3} mm")
    print(f"Th. num of part: {num_part}")
    #liquid_height = container_size[2]*container_scale*np.sqrt(2)*0.5
    #print(liquid_radius, liquid_height)
    # Position liquid relative to container center
    if approach:
        liqpos = (parameters['pos_init_cont'][0]-0.01,parameters['pos_init_cont'][1],parameters['pos_init_cont'][2]+container_size[2]+liquid_height/2) 
    else:
        liqpos = (parameters['pos_grip_ee'][0]+parameters['offset'][1],parameters['pos_grip_ee'][1]-parameters['offset'][0], parameters['pos_grip_ee'][2]+parameters['offset'][2]+container_size[2]+liquid_height/2)

    if liq:
        liquid = scene.add_entity(
            # viscous liquid
            #material=gs.materials.SPH.Liquid(mu=0.02, gamma=0.02),
            material=gs.materials.SPH.Liquid( 
                #rho= parameters['densità'], # 1000.0
                #stiffness=50000.0,
                #exponent=7.0,
                #mu= parameters['viscosità'], # 0.001002       # viscosità dinamica dell'acqua a 20 °C [Pa·s]
                #gamma=parameters['tens_sup'], # 0.0728       # tensione superficiale dell'acqua a 20 °C [N/m]),
                sampler='regular'
            ),
            morph=gs.morphs.Cylinder(
                pos  = liqpos,
                radius = liquid_radius,
                height = liquid_height,  
                # contype=0b010,
                # conaffinity=0b010,      
            ),
            surface=gs.surfaces.Default(
                color    = (0.4, 0.8, 1.0),
                vis_mode = 'particle', #recon / particle
            ),
        )
    else:
        liquid=[]

    if approach:
        # timeconst, dampratio, dmin, dmax, width, mid, power
        unactive_sol_params = np.array([float("inf"), 1.0, 0.0, 0.0, float("inf"), 0, 0], dtype=np.float32)
        # active_sol_params = np.array([0.02, 1.0, 0.95, 0.9999, 1.0, 0.1, 6.0], dtype=np.float32)
        eq_data = np.array([-0.04,0,0.04, 1,0,0,0], dtype=np.float32)  # offset nullo
        #eq_data = np.array([-0.25905615, -0.12823507,  0.16667026, -0.5989425,  -0.03569368,  0.7960116,  0.07974595])
        ur5e.add_equality_between_entities(
            name="grasp_weld",
            type=gs.EQUALITY_TYPE.WELD,
            entity1=ur5e,
            obj1_name="tool0",
            entity2=becher,
            obj2_name=link_becher,
            data=eq_data,
            sol_params=unactive_sol_params,
        )
    else:
        scene.link_entities(
            ur5e,
            becher,
            parent_link_name="tool0",
            child_link_name=link_becher,
        )

    # enter IPython's interactive mode for debug
    # import IPython; IPython.embed()
    
    ########################## build ##########################
    scene.build()

    # sol_param=scene.rigid_solver.get_sol_params()
    # sol_param=sol_param[0]
    # grasp_eq=ur5e.add_equality_between_entities(name="grasp", type=gs.EQUALITY_TYPE.CONNECT, entity1=ur5e, obj1_name="tool0", entity2=becher, obj2_name=link_becher,data=None,sol_params=sol_param)
    # scene.rigid_solver.constraint_solver.add_equality_constraints()
    # print(grasp_eq)

    # Set dofs kp:
    ur5e.set_dofs_kp(
        kp = np.array([5500, 5500, 5500, 4500, 4500, 4500, 20, 20]),
        dofs_idx_local = dofs_idx,
    )
    # Set dofs kv: (Increase velocity gains for better damping)
    ur5e.set_dofs_kv(
        kv = np.array([550,550,550,450,450,450,2,2]),
        dofs_idx_local = dofs_idx,
    )
    # Set force limits:
    ur5e.set_dofs_force_range(
        np.array([-100, -100, -100, -80, -80, -80, -100, -100]),
        np.array([ 100,  100,  100,  80,  80,  80,  100,  100]),
        dofs_idx_local = dofs_idx,
    )
    
    friction=5
    ur5e.set_friction(friction)
    becher.set_friction(friction)
    becher2.set_friction(friction)

    ########################## main ##########################

    # start camera recording. Once this is started, all the rgb images rendered will be recorded internally
    if video==True:
        cam.start_recording()

    # Set initial robot position
    if approach:
        end_effector = ur5e.get_link("tool0")
        init_pos=np.array([parameters['pos_init_ee'][0], parameters['pos_init_ee'][1],parameters['pos_init_ee'][2]])
        init_quat=np.array([parameters['pos_init_ee'][6], parameters['pos_init_ee'][3], parameters['pos_init_ee'][4], parameters['pos_init_ee'][5]]) # xyzw-> wxyz
    else:     
        end_effector = ur5e.get_link("tool0")
        init_pos=np.array([parameters['pos_grip_ee'][0], parameters['pos_grip_ee'][1],parameters['pos_grip_ee'][2]])
        init_quat=np.array([parameters['pos_grip_ee'][6], parameters['pos_grip_ee'][3], parameters['pos_grip_ee'][4], parameters['pos_grip_ee'][5]]) # xyzw-> wxyz
        # x_shift=0.13
        # z_min=0.967
        # quat_orizz = np.array([0.5,0.5,0.5,0.5])
        # init_pos = np.array([parameters['pos_init_cont'][0],parameters['pos_init_cont'][1],parameters['pos_init_cont'][2]])
        # init_pos[0]-=x_shift 
        # init_pos[2]+=container_size[2]-0.01
        # init_pos[2]=max(init_pos[2],z_min)
        # init_quat = quat_orizz
    
    # Use inverse kinematics to get joint angles
    init_qpos = ur5e.inverse_kinematics(
                link=end_effector,
                pos=init_pos,
                quat=init_quat,
                init_qpos=np.deg2rad([-114.0,-138.0,-90.0,-134.0,-116.0,-182.0,0.0,0.0]),
        )
    if approach:
        init_qpos[-2:]=-0.02
    else:
        init_qpos[-2:]=0.005

    ur5e.set_dofs_position(init_qpos)
    scene.visualizer.update(force=True, auto=True)

    # Reach steady state of the liquid
    logger = get_logger('steady_state')
    if liq:
        n=70
        for i in range(n):
            ur5e.control_dofs_position(
                position=init_qpos,
                dofs_idx_local=dofs_idx,
            )
            scene.step()
            # percent = (i + 1) / n
            # bar = ('#' * int(percent * 20)).ljust(20)
            # logger.info(f"[{bar}] {percent*100:.1f}% completato")
            progress_bar(i,n, "steady state")
                
            # cam.render()
        #print("Scene ready to use (steady state reached)")
        logger.info("Scene ready to use (steady state reached)")

    global init_scene
    init_scene = scene.get_state()

    return scene, ur5e, becher, becher2, liquid, dt

def surface_normal(points, ratio_threshold=0.8):
    """Stima la normale a una superficie con SVD."""
    if points.shape[0]<5: return np.array([0,0,1.])
    c = np.mean(points, axis=0)
    _, S, Vt = np.linalg.svd(points - c)
    if S[-1] < ratio_threshold * S[0] and S[-1] < ratio_threshold * S[1]:
        normal = Vt[-1]
        return normal / np.linalg.norm(normal)
    else:
        return np.array([0,0,1.])

def ransac_plane_normal(p, iters=50, tol=0.01):
    # opzionale: robustezza
    best_n, best_in = None, -1
    N = p.shape[0]
    if N < 3: return np.array([0,0,1.])
    idx = np.arange(N)
    for _ in range(iters):
        J = np.random.choice(idx, 3, replace=False)
        n = np.cross(p[J[1]]-p[J[0]], p[J[2]]-p[J[0]])
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-9: continue
        n = n / n_norm
        d = -np.dot(n, p[J[0]])
        dist = np.abs(p @ n + d)
        inliers = (dist < tol).sum()
        if inliers > best_in:
            best_in, best_n = inliers, n
    if best_n is None: return surface_normal(p)
    if best_n[2] < 0: best_n = -best_n
    return best_n / np.linalg.norm(best_n)

class exp_filt_rot:
    def __init__(self, alpha=0.2):
        self.R = R.identity()
        self.alpha = alpha
        self.init = False
    def update(self, R_new):
        if not self.init:
            self.R = R_new; self.init = True; return self.R
        # log/exp smoothing
        R_err = R_new * self.R.inv()
        v = R_err.as_rotvec()
        self.R = R.from_rotvec(self.alpha*v) * self.R
        return self.R

def liq_compensate(particles_world, quat_init_wxyz,top_percent=10,
                   use_ransac=False, alpha=0.2, omega_max=0.8):

    z = particles_world[:,2]
    thr = np.percentile(z, 100 - top_percent)
    surf = particles_world[z >= thr]
    if surf.shape[0] < 3:
        return quat_init_wxyz

    n = ransac_plane_normal(surf) if use_ransac else surface_normal(surf)

    zhat = np.array([0.,0.,1.])
    cos = np.clip(np.dot(n, zhat), -1.0, 1.0)
    angle = np.arccos(cos)
    axis = np.cross(n, zhat)
    s = np.linalg.norm(axis)
    R_corr = R.identity() if (s < 1e-9 or angle < 1e-6) else R.from_rotvec((axis / s) * angle)

    # Solo attorno a x utensile
    rotvec = R_corr.as_rotvec()
    # orientazione corrente dell’utensile
    q_xyzw = np.roll(quat_init_wxyz, -1)
    R0 = R.from_quat(q_xyzw)
    
    # porta l’asse x utensile nel mondo
    motion_axis_tool=np.array([1.,0.,0.])
    axis_world = R0.apply(motion_axis_tool)

    # proietta il rotvec sull’asse x utensile (espresso nel mondo)
    proj = np.dot(rotvec, axis_world) * axis_world
    R_corr = R.from_rotvec(proj)

    # filtro esponenziale su SO(3) con stato persistente per sessione
    static_lpf = getattr(liq_compensate, "_lpf", None)
    if static_lpf is None:
        static_lpf = exp_filt_rot(alpha=alpha)
        liq_compensate._lpf = static_lpf
    R_corr_f = static_lpf.update(R_corr)

    # rate limit per passo
    rotvec = R_corr_f.as_rotvec()
    nrm = np.linalg.norm(rotvec)
    if nrm > omega_max:
        rotvec *= (omega_max / nrm)
    R_step = R.from_rotvec(rotvec)

    R_des = R_step * R0
    q_out_xyzw = R_des.as_quat()
    q_out_wxyz = np.roll(q_out_xyzw, 1)
    return q_out_wxyz

def plan_path(
        ur5e,
        theta_f,
        parameters,
        timeout=5.0,
        smooth_path=True,
        num_waypoints=1000,
        ignore_collision=False,
        planner= "RRTstar", # "RRTConnect"
        return_valid_mask=True,
        debug=False,
        approach=False,
        max_retry=20,
    ):
    logger=get_logger("path logger")
    old=False
    path=np.empty((0, 8))
    print(f"planning started")
    
    #################################
    x_shift=0.15
    z_min=0.967
    lip_height = parameters['pos_cont_goal'][2] + container2_size[2]+0.07
    quat_orizz = np.array([0.5,-0.5,0.5,-0.5])
    if approach:
        # q0 (foto)
        q0 = ur5e.get_qpos()
        collisions0 = ur5e.detect_collision()
        if debug: print(f"Collisioni 0: {collisions0}")
        pos0=np.array([parameters['pos_init_ee'][0], parameters['pos_init_ee'][1],parameters['pos_init_ee'][2]])
        quat0=np.array([parameters['pos_init_ee'][3], parameters['pos_init_ee'][4], parameters['pos_init_ee'][5], parameters['pos_init_ee'][6]])

        q0_test = ur5e.inverse_kinematics(
                    link=ur5e.get_link("tool0"),
                    pos=pos0,
                    quat=quat0,
            )
        
        links_p0, _ = ur5e.forward_kinematics(q0)      
        p0 = links_p0[-1]
        links_p0_test, _ = ur5e.forward_kinematics(q0_test)
        p0_test = links_p0_test[-1]
        pos_error = np.linalg.norm(p0 - p0_test)
        if pos_error>5e-2:
            print(f"Errore: {pos_error}")
            raise RuntimeError(f"Errore nella posizione iniziale troppo grosso")
        
        # q01 (pregrasp)
        pos01 = np.array([parameters['pos_init_cont'][0],parameters['pos_init_cont'][1],parameters['pos_init_cont'][2]])
        pos01[0]-=x_shift 
        pos01[2]+=container_size[2]+0.05
        pos01[2]=max(pos01[2],z_min)
        quat01 = quat_orizz
        try:
            q01 = ur5e.inverse_kinematics(
                link=ur5e.get_link("tool0"),
                pos=pos01,
                quat=quat01
            ) 
        except Exception as e:
            raise RuntimeError(f"errore nella IK q1")
        ur5e.set_qpos(q01)
        scene.visualizer.update(force=True, auto=True)
        collisions01 = ur5e.detect_collision()
        if debug: print(f"Collisioni 01: {collisions01}")

        # Pianifica da posizione iniziale ee a posizione grasping contenitore (0->1): movimento principale nel piano X-Z
        path01, valid = ur5e.plan_path(
            #ee_link_name="tool0",
            qpos_goal=q01,
            qpos_start=q0,
            timeout=timeout,
            num_waypoints=int(num_waypoints*0.8),
            smooth_path=smooth_path,
            ignore_collision=ignore_collision,
            planner=planner,
            return_valid_mask=return_valid_mask,
            max_retry=max_retry,
        )
        if not valid:  # se invalido
            raise RuntimeError(f"path da posizione iniziale a posizione grasping è invalido")
        path01=path01.cpu().numpy()

        # q1 (grasp)
        pos1 = np.array([parameters['pos_init_cont'][0],parameters['pos_init_cont'][1],parameters['pos_init_cont'][2]])
        pos1[0]-=x_shift 
        pos1[2]+=container_size[2]-0.01
        pos1[2]=max(pos1[2],z_min)
        quat1 = quat_orizz
        try:
            q1 = ur5e.inverse_kinematics(
                link=ur5e.get_link("tool0"),
                pos=pos1,
                quat=quat1
            ) 
        except Exception as e:
            raise RuntimeError(f"errore nella IK q1")
        ur5e.set_qpos(q1)
        scene.visualizer.update(force=True, auto=True)
        collisions1 = ur5e.detect_collision()
        if debug: print(f"Collisioni 1: {collisions1}")

        # Pianifica da posizione iniziale ee a posizione grasping contenitore (0->1): movimento principale nel piano X-Z
        path1, valid = ur5e.plan_path(
            #ee_link_name="tool0",
            qpos_goal=q1,
            qpos_start=q01,
            timeout=timeout,
            num_waypoints=int(num_waypoints*0.1),
            smooth_path=smooth_path,
            ignore_collision=ignore_collision,
            planner=planner,
            return_valid_mask=return_valid_mask,
            max_retry=max_retry,
        )
        if not valid:  # se invalido
            raise RuntimeError(f"path da posizione iniziale a posizione grasping è invalido")
        path1=path1.cpu().numpy()
        path1=np.concatenate((path01,path1))
        path = np.concatenate((path, path1))
    else:
        # q1 (grasp)
        # pos1 = np.array([parameters['pos_init_cont'][0],parameters['pos_init_cont'][1],parameters['pos_init_cont'][2]])
        # pos1[0]-=x_shift 
        # pos1[2]+=container_size[2]-0.01
        # pos1[2]=max(pos1[2],z_min)
        # quat1 = quat_orizz
        pos1=np.array([parameters['pos_grip_ee'][0], parameters['pos_grip_ee'][1],parameters['pos_grip_ee'][2]])
        quat1=np.array([parameters['pos_grip_ee'][6], parameters['pos_grip_ee'][3], parameters['pos_grip_ee'][4], parameters['pos_grip_ee'][5]]) # xyzw-> wxyz
        
        try:
            q1 = ur5e.inverse_kinematics(
                link=ur5e.get_link("tool0"),
                pos=pos1,
                quat=quat1
            ) 
        except Exception as e:
            raise RuntimeError(f"errore nella IK q1")
        ur5e.set_qpos(q1)
        scene.visualizer.update(force=True, auto=True)
        collisions1 = ur5e.detect_collision()
        if debug: print(f"Collisioni 1: {collisions1}")
   
    ################################# 
    # q2 (sollevam)
    pos2 = pos1.copy()
    pos2[2]+=0.10
    pos2[2]=max(pos2[2],z_min)
    quat2 = quat_orizz
    try:
        q2 = ur5e.inverse_kinematics(
            link=ur5e.get_link("tool0"),
            pos=pos2,
            quat=quat2
        ) 
    except Exception as e:
        raise RuntimeError(f"errore nella IK q2")
    ur5e.set_qpos(q2)
    scene.visualizer.update(force=True, auto=True)
    collisions2 = ur5e.detect_collision()
    if debug: print(f"Collisioni 2: {collisions2}")

    # Sollevamento (1->2): movimento verticale lungo Z
    path2, valid = ur5e.plan_path(
        #ee_link_name="tool0",
        qpos_goal=q2,
        qpos_start=q1,
        timeout=timeout,
        num_waypoints= 100, #int(num_waypoints/3),
        smooth_path=smooth_path,
        ignore_collision=ignore_collision,
        planner=planner,
        return_valid_mask=return_valid_mask,
        max_retry=max_retry,
    )
    if not valid:  
        raise RuntimeError(f"path di sollevamento è invalido")
    path2=path2.cpu().numpy()
    path = np.concatenate((path, path2))
    logger.info(f"Pianificazione salita eseguita")
   
    #################################
    # q3 (approach cont target): movimento principale nel piano Y-Z
    pos3 = np.array([parameters['pos_cont_goal'][0],parameters['pos_cont_goal'][1],parameters['pos_cont_goal'][2]])
    pos3[0]-=x_shift
    pos3[1] -= (0.01+container2_size[0]/2+container_size[0]/2)
    pos3[2] = pos2[2]
    pos3[2]=max(pos3[2],z_min)
    quat3 = quat_orizz
    try:
        q3 = ur5e.inverse_kinematics(
            link=ur5e.get_link("tool0"),
            pos=pos3,
            quat=quat3
        ) 
    except Exception as e:
        raise RuntimeError(f"errore nella IK q3")
    ur5e.set_qpos(q3)
    scene.visualizer.update(force=True, auto=True)
    collisions3 = ur5e.detect_collision()
    if debug: print(f"Collisioni 3: {collisions3}")

    # Trasporto liquido (2->3) ((modificato in simulazione per cambio orientaz in funzione del liquido))
    path3, valid = ur5e.plan_path(
        #ee_link_name="tool0",
        qpos_goal=q3,
        qpos_start=q2,
        timeout=timeout*10,
        max_nodes=10000,
        num_waypoints=num_waypoints,
        smooth_path=smooth_path,
        ignore_collision=ignore_collision,
        planner=planner,
        return_valid_mask=return_valid_mask,
        max_retry=max_retry,
    )
    if not valid:  
        raise RuntimeError(f"path di trasporto è invalido")
    path3 = path3.cpu().numpy()
    path = np.concatenate((path, path3))
    logger.info(f"Pianificazione trasporto eseguito")
    #################################
    # q4 (pre vers)
    pos4 = np.array([parameters['pos_cont_goal'][0],parameters['pos_cont_goal'][1],parameters['pos_cont_goal'][2]])
    pos4[0]-=x_shift
    if old: 
        pos4[1] -= (0.01+container2_size[0]/2+container_size[0]/2)
        pos4[2] += container2_size[2]+0.05
    else:
        pos4[1] -= (container2_size[0]/2+0.03)
        pos4[2] += container2_size[2]
    
    pos4[2]=max(pos4[2],z_min,lip_height)
    quat4 = quat_orizz
    try:
        q4 = ur5e.inverse_kinematics(
            link=ur5e.get_link("tool0"),
            pos=pos4,
            quat=quat4
        ) 
    except Exception as e:
        raise RuntimeError(f"errore nella IK q4")
    ur5e.set_qpos(q4)
    scene.visualizer.update(force=True, auto=True)
    collisions4 = ur5e.detect_collision()
    if debug: print(f"Collisioni 4: {collisions4}")

    # Posizione pre versamento (3->4): movimento verticale lungo Z
    path4, valid = ur5e.plan_path(
        #ee_link_name="tool0",
        qpos_goal=q4,
        qpos_start=q3,
        timeout=timeout,
        num_waypoints=100, #int(num_waypoints/3),
        smooth_path=smooth_path,
        ignore_collision=ignore_collision,
        planner=planner,
        return_valid_mask=return_valid_mask,
        max_retry=max_retry,
    )
    if not valid:  
        raise RuntimeError(f"path da fine trasporto a preversamento è invalido")
    path4=path4.cpu().numpy()
    path = np.concatenate((path, path4))   
    logger.info(f"Pianificazione discesa eseguita")

    ################################# OLD (TO BE REMOVED IF TESTING OF NEW IS SUCCESSFUL)
    if old:
        # Versamento (4->5) [Da paper: Vision-based robot manipulation of transparent liquid containers in a laboratory setting]
        # La rotazione avviene nel piano Y-Z
        CoR3D = np.array([parameters['pos_cont_goal'][0]-x_shift,parameters['pos_cont_goal'][1]-container2_size[0]/4,parameters['pos_cont_goal'][2]+container2_size[2]/2])    
        _, y_c, z_c = CoR3D
        x0, y0, z0 = pos4
        yaw,pitch,roll=quaternion_to_euler(quat4)
        l = np.sqrt((y0 - y_c)**2 + (z0 - z_c)**2)
        alpha_start = np.arctan2(z0 - z_c, y0 - y_c)
        path5 = []
        n_steps = int(num_waypoints/2.5)
        for theta in np.linspace(0,theta_f,n_steps):
            x = x0 # fixed
            y = y_c + l * np.cos(alpha_start - theta)
            z = z_c + l * np.sin(alpha_start - theta)
            z = max(z, z_min)
        
            pos5 = [x, y, z]
            quat5 = R.from_euler('zxy', [yaw + theta, pitch, roll]).as_quat()
            try:
                q5 = ur5e.inverse_kinematics(
                    link=ur5e.get_link("tool0"),
                    pos=pos5,
                    quat=quat5
                ) 
            except Exception as e:
                raise RuntimeError(f"errore nella IK q5")
            path5.append(q5)
        
        ur5e.set_qpos(q5)
        scene.visualizer.update(force=True, auto=True)
        collisions5 = ur5e.detect_collision()
        if debug: print(f"Collisioni 5: {collisions5}")
        path5 = np.array(path5)
        path5 = path5.cpu().numpy()
        path = np.concatenate((path, path5))    

        #################################
        path6 = []
        for theta in np.linspace(theta_f, 0, n_steps):
            x = x0  # fisso
            y = y_c + 1.5 * l * np.cos(alpha_start - theta)  
            z = z_c + 1.0 * l * np.sin(alpha_start - theta)
            z = max(z, z_min)

            pos6 = [x, y, z]
            quat6 = R.from_euler('zxy', [yaw + theta, pitch, roll]).as_quat()
            try:
                q6 = ur5e.inverse_kinematics(
                    link=ur5e.get_link("tool0"),
                    pos=pos6,
                    quat=quat6
                )
            except Exception:
                raise RuntimeError("errore nella IK q6")
            path6.append(q6)

        ur5e.set_qpos(q6)
        scene.visualizer.update(force=True, auto=True)
        collisions6 = ur5e.detect_collision()
        if debug:
            print(f"Collisioni 6: {collisions6}")
        path6 = np.array(path6)
        path6 = path6.cpu().numpy()
        path = np.concatenate((path, path6))

    else:
        ######################################
        # Versamento (4->5)
        CoR3D = np.array([
            parameters['pos_cont_goal'][0] + parameters['dCoR'][0], # 0.0
            parameters['pos_cont_goal'][1] - 0.005 + parameters['dCoR'][1], # - 0.01 
            parameters['pos_cont_goal'][2] + parameters['dCoR'][2], # + 0.04
        ])
        p_tcp0 = pos4.copy()
        scene.draw_debug_sphere(CoR3D, radius=0.005, color=(1.0, 0.0, 0.0, 1.0))
        R0 = R.from_quat(quat4) # matrice rot init
        l = R0.inv().apply(CoR3D - p_tcp0) # offset tool0 --> CoR3D
        tool_x_axis = np.array([1.0, 0.0, 0.0])             # asse x nel frame tool
        axis_world=tool_x_axis
      
        path5 = []
        n_steps = int(num_waypoints/2)
        for theta in np.linspace(0, theta_f, n_steps):
            R_theta = R.from_rotvec(theta * axis_world) * R0 # matrice rotazione lungo x
            quat5 = R_theta.as_quat()
            delta_pos=R_theta.apply(l)
            p_tcp = CoR3D - delta_pos
            p_tcp[2] = max(pos4[2],z_min, lip_height) 
        
            try:
                q5 = ur5e.inverse_kinematics(
                    link=ur5e.get_link("tool0"),
                    pos=p_tcp,
                    quat=quat5
                )
            except Exception:
                raise RuntimeError("errore nella IK q5 (pour)")

            path5.append(q5)

        ur5e.set_qpos(q5)
        scene.visualizer.update(force=True, auto=True)
        collisions5 = ur5e.detect_collision()
        if debug: print(f"Collisioni 5: {collisions5}")
        path5 = np.stack([q.cpu().numpy() if isinstance(q, torch.Tensor) else np.array(q) for q in path5])
        path = np.concatenate((path, path5))

        ###########################################
        # Ritorno dal versamento (5->6)
        path6 = []
        for theta in np.linspace(theta_f, 0.0, n_steps):
            R_theta = R.from_rotvec(theta * axis_world) * R0
            quat6 = R_theta.as_quat()

            p_tcp = CoR3D - R_theta.apply(l)
            p_tcp[2] = max(pos4[2],z_min,lip_height)

            try:
                q6 = ur5e.inverse_kinematics(
                    link=ur5e.get_link("tool0"),
                    pos=p_tcp,
                    quat=quat6
                )
            except Exception:
                raise RuntimeError("errore nella IK q6 (unpour)")

            path6.append(q6)
        pos6=p_tcp
        ur5e.set_qpos(q6)
        scene.visualizer.update(force=True, auto=True)
        collisions6 = ur5e.detect_collision()
        if debug: print(f"Collisioni 6: {collisions6}")
        path6 = np.stack([q.cpu().numpy() if isinstance(q, torch.Tensor) else np.array(q) for q in path6])
        path = np.concatenate((path, path6))
    logger.info(f"Pianificazione pouring e unpouring eseguita")
    #################################
    # q7
    pos7 = pos6    
    pos7[2] =parameters['pos_init_cont'][2]+container_size[2]
    pos7[2]=max(pos7[2],z_min)
    quat7 = quat_orizz
    try:
        q7 = ur5e.inverse_kinematics(
            link=ur5e.get_link("tool0"),
            pos=pos7,
            quat=quat7
        ) 
    except Exception as e:
        raise RuntimeError(f"errore nella IK q7")
    ur5e.set_qpos(q7)
    scene.visualizer.update(force=True, auto=True)
    collisions7 = ur5e.detect_collision()
    if debug: print(f"Collisioni 7: {collisions7}")
    # Termina rilasciando contenitore sul tavolo
    path7, valid = ur5e.plan_path(
        #ee_link_name="tool0",
        qpos_goal=q7,
        qpos_start=q6,
        timeout=timeout,
        num_waypoints= 30, #int(num_waypoints/10),
        smooth_path=smooth_path,
        ignore_collision=ignore_collision,
        planner=planner,
        return_valid_mask=return_valid_mask,
        max_retry=max_retry,
    )
    if not valid:  
        print(path7)
        raise RuntimeError(f"path di release")
    path7=path7.cpu().numpy()
    path = np.concatenate((path, path7))

    if debug: print(path)
    print(f"Planning complete")
    
    if approach:
        return {
        "init_to_grasp": path1,
        "lift": path2,
        "transport": path3,
        "pre_pour": path4,
        "pour": path5,
        "unpour": path6,
        "release": path7,
        "all": path
        }
    else:
        return {
        "lift": path2,
        "transport": path3,
        "pre_pour": path4,
        "pour": path5,
        "unpour": path6,
        "release": path7,
        "all": path
        }

def compute_reward(liquid, becher2, parameters, t0, scene):
    particles = np.squeeze(liquid.get_particles())
    target_vol = parameters['vol_target']
    vol_tol = 0.1 * target_vol  # ±10%

    #  Bounding box reale del becher target 
    aabb = becher2.get_AABB().cpu().numpy().squeeze()  # shape (2, 3)
    lower, upper = aabb[0], aabb[1]

    print(f"AABB: {lower} - {upper}")
    #  Particelle dentro il volume target (AABB) 
    inside_mask = np.all((particles >= lower) & (particles <= upper), axis=1)
    num_inside = np.sum(inside_mask)
    frac_inside = num_inside / len(particles)
    
    # Errore volume:
    actual_vol = num_inside * liquid.particle_size
    vol_err = abs(actual_vol - target_vol)

    #  Perdite (particelle cadute sotto piano tavolo) 
    loss_frac = 1 - frac_inside

    #  Tempo 
    Dt = scene.get_state().scene.t - t0 

    #  Pesi
    w_vol, w_loss, w_time = 3.0, 3.0, 0.5

    reward = 0.0

    # Volume
    reward1 = w_vol * max(0, 1 - vol_err / vol_tol) # Alternativa: reward1 = w_vol * np.exp(-vol_err / (vol_tol + 1e-12))
    print(f"reward errore volume: {reward1}")

    # Perdite
    reward2 = w_loss * (1-loss_frac)
    print(f"reward perdite: {reward2}")

    # Penalità tempo (normalizzata) 
    reward3 = w_time * Dt / 10.0 
    print(f"reward tempo: {reward3}")

    reward = reward1 + reward2 + reward3
    return reward

def compute_fake_reward(parameters, theta_f, num_wp):
    
    contpos2 = np.array([parameters['pos_cont_goal'][0], parameters['pos_cont_goal'][1], parameters['pos_cont_goal'][2]])
    real_contpos2 = np.array([0.746, 0.961, 0.960])
    pos_err = np.linalg.norm(contpos2 - real_contpos2)
    pos_tol = 0.05

    CoR3D = np.array([
            parameters['pos_cont_goal'][0] + parameters['dCoR'][0], # 0.0
            parameters['pos_cont_goal'][1] - 0.005 + parameters['dCoR'][1], # - 0.01 
            parameters['pos_cont_goal'][2] + parameters['dCoR'][2], # + 0.04
        ])
    real_CoR3d=np.array([real_contpos2[0], real_contpos2[1] - 0.005, real_contpos2[2]])
    cor_err = np.linalg.norm(CoR3D - real_CoR3d)
    cor_tol = 0.1

    theta_opt=90
    err_theta=np.linalg.norm(theta_f-theta_opt)
    err_max_theta=10

    num_wp_opt=320
    err_num_wp=np.linalg.norm(num_wp-num_wp_opt)
    err_max_num_wp=50

    w_pos, w_cor = 3, 1
    w_theta, w_num_wp = 1,1

    reward1 = w_pos * max(0,1-pos_err/pos_tol)
    reward2 = w_cor * max(0,1-cor_err/cor_tol)
    reward3 = w_theta * max (0,1-err_theta/err_max_theta)
    reward4 = w_num_wp * max (0,1-err_num_wp/err_max_num_wp)

    print(f"reward cont pos err: {reward1}")
    print(f"reward pos CoR: {reward2}")
    print(f"reward theta: {reward3}")
    print(f"reward num_wp: {reward4}")

    reward = reward1 + reward2 + reward3 + reward4
    
    return reward

def simulate_action(ur5e, parameters, paths, scene, becher, becher2, liquid, liq, approach=False, antisloshing=False): 
    # Reset env:
    # reset_sim(scene, ur5e, becher, becher2, liquid, parameters)
    scene.reset(init_scene)
    print("Simulation started")
    t0=scene.get_state().scene.t

    # Ottieni indici locali dei giunti
    dofs_idx = []
    for joint in ur5e.joints:
        if joint.name not in ["joint_world","flange-tool0","robotiq_hande_base_joint"]:
            dofs_idx.extend(joint.dofs_idx_local)
    motors_dof = dofs_idx[:-2]
    fingers_dof = dofs_idx[-2:]
    
    path_debug = scene.draw_debug_path(torch.from_numpy(paths["all"]), ur5e)
    ################################################################################################################################### 
    # Esegui il path
    score=0
    excluded=[]
    logger=get_logger("sim logger")
    if liq:
        particles = np.squeeze(liquid.get_particles())
        h_min=np.min(particles[:,2])

    opening_force=np.array([-0.5, -0.5])
    closing_force=np.array([5, 5])

    if approach:
        # Init to grasp:
        path1=paths["init_to_grasp"]
        for qpos in path1:
            # ur5e.control_dofs_position(qpos, dofs_idx_local=dofs_idx)
            ur5e.control_dofs_position(qpos[:-2], motors_dof)
            ur5e.control_dofs_force(opening_force, fingers_dof)
            scene.step()
        for _ in range(10): scene.step() # per raggiungere ultimo waypoint

        # Grasping
        eq = next(e for e in ur5e.equalities if e.name=="grasp_weld")

        # q = ur5e.get_qpos()
        # tool0=ur5e.get_link("tool0")
        # p1=tool0.get_pos().cpu().numpy()
        # p2=becher.get_pos().cpu().numpy()
        # pos=p1-p2
        # q1=tool0.get_quat().cpu().numpy()
        # q2=becher.get_quat().cpu().numpy()
        # quat = quat_multiply(quat_inverse(q2), q1)  # q_rel = q2⁻¹ * q1
        # eq_data = np.concatenate([pos, quat]).astype(np.float32)
        # eq.set_eq_data(eq_data)
        #active_sol_params = np.array([0.02, 1.0, 1e-4, 0.9999, 0.0, 0.5, 2.0], dtype=np.float32)
        
        # timeconst, dampratio 
        # dmin, dmax --> impedance (0,1) = constraint’s ability to generate force (Small values of dd correspond to weak constraints while large values of dd correspond to strong constraints)    
        # width, mid, power (midpoint and power control the shape of the sigmoidal function that interpolates between dmin​ and dmax, vedi img su desktop)

        active_sol_params = np.array([2, 1.0, 1e-4, 0.9999, 1.0, 0.1, 1.0], dtype=np.float32)
        eq.set_sol_params(active_sol_params)
        
        qpos=path1[-1]
        for i in range(100):
            ur5e.control_dofs_position(qpos[:-2], motors_dof)
            # ur5e.control_dofs_velocity(np.array([0.1,0.1]),fingers_dof)
            ur5e.control_dofs_force(closing_force/2, fingers_dof)
            #active_sol_params = np.array([(100-i/5)*0.02+0.02, 1.0, 1e-4, 0.9999, 1.0, 0.1, 1.0], dtype=np.float32)
            #eq.set_sol_params(active_sol_params)
            scene.step()

    # Lift:
    path2=paths["lift"]
    for i, qpos in enumerate(path2):
        qpos[-2:]=0.005
        qpos=to_device_tensor(qpos)
        if approach:
            ur5e.control_dofs_position(qpos[:-2], motors_dof)
            ur5e.control_dofs_force(closing_force, fingers_dof)
        else:
            ur5e.control_dofs_position(qpos, dofs_idx_local=dofs_idx)
            ur5e.set_dofs_position(qpos[-2:],fingers_dof)
        if liq:
            particles2 = np.squeeze(liquid.get_particles())
            for idx, particle in enumerate(particles2):
                    if particle[2] < h_min and idx not in excluded: # da cambiare con un collision detection
                        score-=5/len(particles2) # to be tuned
                        excluded.append(idx)
        scene.step()
        # percent = (i + 1) / len(path2)
        # bar = ('#' * int(percent * 20)).ljust(20)
        # logger.info(f"[{bar}] {percent*100:.1f}% lifting")
        progress_bar(i,len(path2), "Lifting")
    
    # Trasporto:
    path3=paths["transport"]
    quat_prev=None
    for i, wp in enumerate(path3): 
        wp=to_device_tensor(wp)
        if liq:
            if antisloshing:
                pos_wp, quat_wp = ur5e.forward_kinematics(wp)
                pos_wp=pos_wp[7] # ['world', 'shoulder_link', 'upper_arm_link', 'forearm_link', 'wrist_1_link', 'wrist_2_link', 'wrist_3_link', 'tool0', 'hand_e_link', 'hande_left_finger', 'hande_right_finger']
                quat_wp=quat_wp[7]
                particles = np.squeeze(liquid.get_particles())
                quat_np = to_numpy_cpu(quat_wp)
                quat_new = liq_compensate(particles, quat_np)
                #print(f"old: {quat_wp}, new: {quat_new}")

                # filtro cambio e riconversione
                if quat_prev is None or np.linalg.norm(quat_new - quat_prev) < 0.1:
                    quat_prev = quat_new
                quat_wp = to_device_tensor(quat_prev)
                try:
                    qpos = ur5e.inverse_kinematics(
                        link=ur5e.get_link("tool0"),
                        pos=pos_wp,
                        quat=quat_wp
                    )
                    qpos[-2:]=0.005
                except Exception as e:
                    raise RuntimeError(f"errore nella IK liq ang")
            else:
                qpos=wp
            if approach: 
                ur5e.control_dofs_position(qpos[:-2], motors_dof)
                ur5e.control_dofs_force(closing_force, fingers_dof)
            else:
                ur5e.control_dofs_position(qpos, dofs_idx_local=dofs_idx)
                ur5e.set_dofs_position(qpos[-2:],fingers_dof)
        else:
            if approach:
                ur5e.control_dofs_position(wp[:-2], motors_dof)
                ur5e.control_dofs_force(closing_force, fingers_dof)
            else:
                ur5e.control_dofs_position(wp, dofs_idx_local=dofs_idx)
                ur5e.set_dofs_position(qpos[-2:],fingers_dof)
        if liq:
            particles3 = np.squeeze(liquid.get_particles())
            for particle in particles3:
                    if particle[2] < h_min and idx not in excluded: # da cambiare con un collision detection
                        score-=5/len(particles3) # to be tuned
                        excluded.append(idx)
        scene.step()
        # percent = (i + 1) / len(path3)
        # bar = ('#' * int(percent * 20)).ljust(20)
        # logger.info(f"[{bar}] {percent*100:.1f}% transport")
        progress_bar(i,len(path3), "Transport")

    # Posizionamento pre pouring:
    path4=paths["pre_pour"]
    for i, qpos in enumerate(path4):
        qpos[-2:]=0.005
        if approach:
            ur5e.control_dofs_position(qpos[:-2], motors_dof)
            ur5e.control_dofs_force(closing_force, fingers_dof)
        else:
            ur5e.control_dofs_position(qpos, dofs_idx_local=dofs_idx)
            ur5e.set_dofs_position(qpos[-2:],fingers_dof)
        if liq:
            particles4 = np.squeeze(liquid.get_particles())
            for particle in particles4:
                    if particle[2] < h_min: # da cambiare con un collision detection
                        score-=5/len(particles4) # to be tuned
                        excluded.append(idx)
        scene.step()
        # percent = (i + 1) / len(path4)
        # bar = ('#' * int(percent * 20)).ljust(20)
        # logger.info(f"[{bar}] {percent*100:.1f}% lowering")
        progress_bar(i,len(path4),"Lowering")

    if approach:
        for _ in range(10):
            ur5e.control_dofs_position(qpos[:-2], motors_dof)
            ur5e.control_dofs_force(closing_force, fingers_dof)
            scene.step()

    # Pouring:
    path5=paths["pour"]   
    for i, qpos in enumerate(path5):
        qpos[-2:]=0.005
        if approach:
            ur5e.control_dofs_position(qpos[:-2], motors_dof)
            ur5e.control_dofs_force(closing_force, fingers_dof)
        else:
            ur5e.control_dofs_position(qpos, dofs_idx_local=dofs_idx)
            ur5e.set_dofs_position(qpos[-2:],fingers_dof)
        if liq:
            particles5 = np.squeeze(liquid.get_particles())
            for particle in particles5:
                    if particle[2] < h_min and idx not in excluded: # da cambiare con un collision detection
                        score-=5/len(particles5) # to be tuned
                        excluded.append(idx)
        scene.step()
        # percent = (i + 1) / len(path5)
        # bar = ('#' * int(percent * 20)).ljust(20)
        # logger.info(f"[{bar}] {percent*100:.1f}% pouring")
        progress_bar(i,len(path5), "Pouring")

    # Unpouring:
    path6=paths["unpour"]   
    for i, qpos in enumerate(path6):
        qpos[-2:]=0.005
        if approach:
            ur5e.control_dofs_position(qpos[:-2], motors_dof)
            ur5e.control_dofs_force(closing_force, fingers_dof)
        else:
            ur5e.control_dofs_position(qpos, dofs_idx_local=dofs_idx)
            ur5e.set_dofs_position(qpos[-2:],fingers_dof) 
        if liq:
            particles6 = np.squeeze(liquid.get_particles())
            for particle in particles6:
                    if particle[2] < h_min and idx not in excluded: # da cambiare con un collision detection
                        score-=5/len(particles5) # to be tuned
                        excluded.append(idx)
        scene.step()
        # percent = (i + 1) / len(path6)
        # bar = ('#' * int(percent * 20)).ljust(20)
        # logger.info(f"[{bar}] {percent*100:.1f}% unpouring")
        progress_bar(i,len(path6), "Unpouring")

    # Release:
    path7=paths["release"]
    for i, qpos in enumerate(path7):
        qpos[-2:]=0.005
        if approach:
            ur5e.control_dofs_position(qpos[:-2], motors_dof)
            ur5e.control_dofs_force(closing_force, fingers_dof)
        else:
            ur5e.control_dofs_position(qpos, dofs_idx_local=dofs_idx)
            ur5e.set_dofs_position(qpos[-2:],fingers_dof)
        if liq:
            particles7 = np.squeeze(liquid.get_particles())
            for particle in particles7:
                    if particle[2] < h_min and idx not in excluded: # da cambiare con un collision detection
                        score-=5/len(particles6) # to be tuned
                        excluded.append(idx)
        scene.step()

    # Valuta successo
    # if liq:
    #     particles = np.squeeze(liquid.get_particles())
    #     contpos = np.array(parameters['pos_cont_goal'])
    #     err = parameters['err_target']
    #     target_vol=parameters['vol_target']

    #     # da modificare ass: la media delle particelle prob non coinc con centro del target -> misurare volume effettivo (con bounding box del becher2)
    #     ck1 = abs(np.mean(particles[:, 0])-contpos[0])< err # err x
    #     ck2 = abs(np.mean(particles[:, 1])-contpos[1])< err # err y
    #     ck3 = abs(np.mean(particles[:, 2])-contpos[2])< err # err z
    #     if ck1 and ck2 and ck3:
    #         score+=4/len(particles) # to be tuned

    #     mask = (
    #         (np.abs(particles[:, 0] - contpos[0]) < err) &
    #         (np.abs(particles[:, 1] - contpos[1]) < err) &
    #         (np.abs(particles[:, 2] - contpos[2]) < err)
    #     )
    #     num_particles_in_target = np.sum(mask)
    #     vol=num_particles_in_target*liquid.particle_size
    #     if abs(vol-target_vol)<err:
    #         score+=1 # to be tuned
    
    # if liq:
    #     particles = np.squeeze(liquid.get_particles())
    #     contpos = np.array(parameters['pos_cont_goal'])
    #     pos_err = parameters['err_target']
    #     target_vol = parameters['vol_target']
    #     vol_tol = 0.1 * target_vol  # 10%

    #     mask = (
    #         (np.abs(particles[:, 0] - contpos[0]) < pos_err) &
    #         (np.abs(particles[:, 1] - contpos[1]) < pos_err) &
    #         (np.abs(particles[:, 2] - contpos[2]) < pos_err)
    #     )
    #     num_in = np.sum(mask)
    #     ratio = num_in / len(particles)
    #     vol = num_in * liquid.particle_size

    #     score += 4 * ratio
    #     if abs(vol - target_vol) < vol_tol:
    #         score += 1

    # tf=scene.get_state().scene.t
    # Dt=tf-t0
    # t_ref = 10 # la sim dovrebbe durare circa 10s
    # score-=1e-2*Dt/t_ref

    if liq:
        score = compute_reward(liquid, becher2, parameters, t0, scene)
    else:
        tf=scene.get_state().scene.t
        Dt=tf-t0
        t_ref = 10 # la sim dovrebbe durare circa 10s
        score+=1
        score-=1e-2*Dt/t_ref
    
    print(f"Simulation completed")
    return score

def is_success(score, threshold=0.5):
    return score > threshold

def fake_sim(ur5e, paths, scene, path_debug, approach=False):
    """
    Note that this sim is only for visualization purposes (i.e. we do not call
    scene.step(), but only update the state and the visualizer) 
    """
    scene.reset(init_scene)
    
    # Ottieni indici locali dei giunti
    dofs_idx = []
    for joint in ur5e.joints:
        if joint.name not in ["joint_world","flange-tool0","robotiq_hande_base_joint"]:
            dofs_idx.extend(joint.dofs_idx_local)
    
    if approach:
        # Init to grasp:
        path1=paths["init_to_grasp"]
        for qpos in path1:
            ur5e.set_dofs_position(qpos)
            scene.visualizer.update(force=True, auto=True)
        
    # Grasping
    # qpos=path1[-1]
    # motors_dof = dofs_idx[:-2]
    # fingers_dof = dofs_idx[-2:]
    # for i in range(100):
    #     ur5e.set_dofs_position(qpos[:-2], motors_dof)
    #     ur5e.control_dofs_force(np.array([-0.5*i, 0.5*i]), fingers_dof)
    #     scene.visualizer.update(force=True, auto=True)

    # Lift:
    path2=paths["lift"]
    for qpos in path2:
        ur5e.set_dofs_position(qpos, dofs_idx_local=dofs_idx) # bisognerebbe aggiungere qui il delay di controllo (delay_control)
        #ur5e.control_dofs_force(np.array([-0.5, 0.5]), fingers_dof)
        scene.visualizer.update(force=True, auto=True)

    # Trasporto:
    path3=paths["transport"]
    for qpos in path3: 
        ur5e.set_dofs_position(qpos, dofs_idx_local=dofs_idx) # bisognerebbe aggiungere qui il delay di controllo (delay_control)
        #ur5e.control_dofs_force(np.array([-0.5, 0.5]), fingers_dof)
        scene.visualizer.update(force=True, auto=True)

    # Posizionamento pre pouring:
    path4=paths["pre_pour"]
    for qpos in path4:
        ur5e.set_dofs_position(qpos, dofs_idx_local=dofs_idx) # bisognerebbe aggiungere qui il delay di controllo (delay_control)
        #ur5e.control_dofs_force(np.array([-0.5, 0.5]), fingers_dof)
        scene.visualizer.update(force=True, auto=True)
    # Pouring:
    path5=paths["pour"]   
    for qpos in path5:
        ur5e.set_dofs_position(qpos, dofs_idx_local=dofs_idx) # bisognerebbe aggiungere qui il delay di controllo (delay_control)
        #ur5e.control_dofs_force(np.array([-0.5, 0.5]), fingers_dof)
        scene.visualizer.update(force=True, auto=True)
    # Pouring:
    path6=paths["unpour"]  
    for qpos in path6:
        ur5e.set_dofs_position(qpos, dofs_idx_local=dofs_idx) # bisognerebbe aggiungere qui il delay di controllo (delay_control)
        #ur5e.control_dofs_force(np.array([-0.5, 0.5]), fingers_dof)
        scene.visualizer.update(force=True, auto=True)
    # Release:
    path7=paths["release"]
    for qpos in path7:
        ur5e.set_dofs_position(qpos, dofs_idx_local=dofs_idx) # bisognerebbe aggiungere qui il delay di controllo (delay_control)
        #ur5e.control_dofs_force(np.array([-0.5, 0.5]), fingers_dof)
        scene.visualizer.update(force=True, auto=True)
    
    scene.clear_debug_object(path_debug)

    print(f"Fake simulation completed")

def update_w(theta, y, w_mean, w_cov):
    X = np.vstack([np.ones(len(theta)), theta]).T # modello logistico lineare (lin logit)
    w = w_mean.copy()
    # Laplace approx of posterior using Newton 
    for _ in range(5):
        p = expit(X @ w) # funzione sigmoide di scipy ottimizzata
        W = np.diag(p*(1-p))
        H = X.T @ W @ X + np.linalg.inv(w_cov)
        g = X.T @ (y - p) - np.linalg.inv(w_cov) @ (w - w_mean)
        try:
            w = w + np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
    w_cov_post = np.linalg.inv(H)
    return w, w_cov_post

def sample_x_TS(w_mean, w_cov, x_min, x_max, M, n_grid=50):
    thetas = np.linspace(x_min, x_max, n_grid)
    w_samples = np.random.multivariate_normal(w_mean, w_cov, size=M)
    x_nexts = []
    for ws in w_samples:
        scores = expit(ws[0] + ws[1]*thetas)
        x_nexts.append(thetas[np.argmax(scores)])
    return np.array(x_nexts)

def best_theta_greedy(w_mean, x_min, x_max, n_grid=200):
    grid = np.linspace(x_min, x_max, n_grid)
    p = expit(w_mean[0] + w_mean[1]*grid)
    return float(grid[np.argmax(p)])

def best_theta_bayes(w_mean, w_cov, x_min, x_max, M=200, n_grid=200):
    grid = np.linspace(x_min, x_max, n_grid)
    acc = np.zeros_like(grid)
    ws = np.random.multivariate_normal(w_mean, w_cov, size=M)
    for w in ws:
        acc += expit(w[0] + w[1]*grid)
    return float(grid[np.argmax(acc / M)])

class PathPlannerService(Node):
    def __init__(self):
        super().__init__('path_planner_service')
        self.srv = self.create_service(Simplan, 'plan_path', self.plan_path_callback)
        self.get_logger().info("Path planner service ready")

    def _make_parameters_range(self, values: dict, tolerances: dict) -> dict:
        """
        Crea un dizionario con i range a partire da valori e tolleranze.

        :param values: dict con valori nominali
        :param tolerances: dict con tolleranze (tol_minus, tol_plus)
                        - se singolo valore -> tol_minus = tol_plus
                        - se lista -> stessa struttura del valore
        :return: dict con ranges
        """
        result = {}

        for key, val in values.items():
            if key not in tolerances:
                continue  # ignora se non definita tolleranza

            tol = tolerances[key]

            # Caso tol rel
            if isinstance(val, (int, float)) and isinstance(tol, tuple) and len(tol) == 3 and tol[0] == "rel":
                minus_frac, plus_frac = tol[1], tol[2]
                result[key] = [val * (1 - minus_frac), val * (1 + plus_frac)]
                continue

            # Caso scalare (float/int)
            if isinstance(val, (int, float)):
                if tol is None:
                    result[key] = [val, val]
                else:
                    tmin, tmax = tol if isinstance(tol, tuple) else (tol, tol)
                    result[key] = [val - tmin, val + tmax]

            # Caso lista/vettore
            elif isinstance(val, (list, tuple)):
                ranges = []
                for i, v in enumerate(val):
                    if tol is None:
                        ranges.append([v, v])
                    elif isinstance(tol, list):  
                        # lista di tuple per ogni componente
                        tmin, tmax = tol[i]
                        ranges.append([v - tmin, v + tmax])
                    else:
                        # stessa tolleranza per tutti
                        tmin, tmax = tol if isinstance(tol, tuple) else (tol, tol)
                        ranges.append([v - tmin, v + tmax])
                result[key] = ranges

            else:
                raise TypeError(f"Tipo non supportato per {key}: {type(val)}")

        return result
    
    def to_builtin(self, obj):
        # numpy scalari -> tipo Python
        if isinstance(obj, np.generic):
            return obj.item()

        # numpy array -> lista Python ricorsiva
        if isinstance(obj, np.ndarray):
            return [self.to_builtin(x) for x in obj.tolist()]

        # lista -> lista pulita
        if isinstance(obj, list):
            return [self.to_builtin(x) for x in obj]

        # tupla -> lista (YAML gestisce le liste meglio, ed è ok perdere l'immutabilità)
        if isinstance(obj, tuple):
            return [self.to_builtin(x) for x in obj]

        # dict -> dict pulito
        if isinstance(obj, dict):
            return {k: self.to_builtin(v) for k, v in obj.items()}

        # tipi base (int, float, str, bool, None) restano così
        return obj

    def plan_path_callback_old(self, request, response):

        N = 4                    # Numero di modelli simulati (iniziale)
        M = 2                    # Numero di traiettorie
        delta = 1/N              # Threshold di successo
        view=False
        liq=False
        record=False
        debug=False        
    
        if request.no_params == True:
            PARAMS_FILE = "/tmp/parameters.yaml"

            if not os.path.exists(PARAMS_FILE):
                response.success = False
                return response
            else:
                with open(PARAMS_FILE, "r") as f:
                    data = yaml.safe_load(f)
                if "parameters" not in data:
                    raise RuntimeError("File init_parameters.yaml non contiene chiave 'parameters'")
                parameters_set=data["parameters"]
                if request.target_vol is not None:
                    for p in parameters_set:
                        p["vol_target"]=request.target_vol
                for p in parameters_set:
                    p["err_target"]=5e-6
                    # TODO sistemare aggiornamento params
        else:
            req_parameters = {
                "pos_init_cont": list(request.pos_init_cont),
                "pos_cont_goal": list(request.pos_cont_goal),
                "pos_init_ee": list(request.pos_init_ee),
                "pos_grip_ee":list(request.pos_grip_ee),
                "offset": list(request.offset),
                "dCoR": [0.0, -0.015, 0.04],
                "vol_init": request.init_vol, #2e-5, +-MAE
                "densità": 998.0,
                "viscosità": 0.001,
                "tens_sup": 0.072,
                "vol_target": request.target_vol, #0.75e-5,
                "err_target": 5e-6,
                "theta_f": request.theta_f, #+-15°
                "num_wp": int(request.num_wp),
            }
            tolerances = {
                "pos_init_cont": [
                    (0.01, 0.01),  # x: ±1.0 cm
                    (0.0, 0.0),    # y: ±0.0 cm
                    (0.0, 0.0),    # z: ±0.0 cm
                ],
                "pos_cont_goal": [
                    (0.015, 0.015),  # x: ±1.5 cm
                    (0.015, 0.015),  # y: ±1.5 cm
                    (0.0, 0.0),      # z: ±0.0 cm
                ],
                "pos_init_ee": [
                    (0.00, 0.00),    # x: fisso
                    (0.00, 0.00),    # y: fisso
                    (0.00, 0.00),    # z: fisso
                    (0.00, 0.00),    # w: fisso
                    (0.00, 0.00),    # x: fisso
                    (0.00, 0.00),    # y: fisso
                    (0.00, 0.00),    # z: fisso
                ],
                "pos_grip_ee": [
                    (0.00, 0.00),    # x: fisso
                    (0.00, 0.00),    # y: fisso
                    (0.00, 0.00),    # z: fisso
                    (0.00, 0.00),    # w: fisso
                    (0.00, 0.00),    # x: fisso
                    (0.00, 0.00),    # y: fisso
                    (0.00, 0.00),    # z: fisso
                ],
                "offset": [
                    (0.01, 0.01),  # x: ±1 cm (0.15)
                    (0.0, 0.0),    # y: ±0 cm (0.0) offset bloccato (le pinze riportano al centro quando chiuse) 
                    (0.0, 0.0),    #  ±0 cm (0.04) offset bloccato (le pinze riportano al centro quando chiuse)
                ],
                "dCoR": [
                    (0.001, 0.001),  # componente 1: ±1 mm
                    (0.01, 0.01),    # componente 2: ±1 cm
                    (0.01, 0.01),    # componente 3: ±1 cm
                ],
                "viscosità": (0.00025, 0.00025),  # ±25% intorno a 0.001 Pa·s
                "densità": (3.0, 3.0),          # ±3 kg/m^3
                "tens_sup": (0.002, 0.001),     # -0.002 / +0.001 N/m (gamme tipiche 0.070–0.073)
                "vol_init": ( 1.5e-5, 1.5e-5),  # ±1e-5 m^3 (15ml)
                "vol_target": (0.0, 0.0),       # no tol, è scelta
                "err_target": (0.0, 0.0),       # vincolo rigido
                "theta_f": (10.0, 10.0),        # ±10°
                "num_wp": ("rel", 0.2, 0.2),    # ±20%
            }
            
            parameters_range=self._make_parameters_range(req_parameters,tolerances)
            parameters_set=[]
            for _ in range(N):
                parameters_set.append(generate_parameters(parameters_range)) 

            tolerances_save=self.to_builtin(tolerances.copy())

        
            try:
                with open("/tmp/tolerances.yaml", "w") as f:
                    yaml.safe_dump({"tolerances":tolerances_save}, f, sort_keys=False)
            except Exception as e:
                self.get_logger().error(f"Errore salvataggio YAML: {e}")
                response.success = False
                return response

        iter_file = "/tmp/iter.yaml"
        if not os.path.exists(iter_file):
            k=0
        else:
            with open(iter_file, "r") as f:
                    k = yaml.safe_load(f)

        if k % 2==0:
            file = "/tmp/TStheta.yaml"
            if not os.path.exists(file):
                x_hist = []
                current_x =  90
                y_hist = []
                w_mean = np.zeros(2)
                w_cov = np.eye(2)*10.0 
            else:
                with open(file, "r") as f:
                    data = yaml.safe_load(f)
                x_hist = data["history"] or []
                current_x = data["new_x"] or 90
                y_hist = data["success"] or [] # lista di 1=success,0=failure
                w_mean = data["w_mean"] or np.zeros(2) # Prior inizializzato come N(0,metà intervallo)
                w_cov = data["w_cov"] or np.eye(2)*10.0 # Prior inizializzato come N(0,metà intervallo)
                w_mean = np.array(w_mean)
                w_cov  = np.array(w_cov)
        else:
            file = "/tmp/TSnum_wp.yaml"
            if not os.path.exists(file):
                x_hist = []
                current_x =  350
                y_hist = []
                w_mean = np.zeros(2)
                w_cov = np.eye(2)*50.0 
            else:
                with open(file, "r") as f:
                    data = yaml.safe_load(f)
                x_hist = data["history"] or []
                current_x = data["new_x"] or 350
                y_hist = data["success"] or [] # lista di 1=success,0=failure
                w_mean = data["w_mean"] or np.zeros(2) # Prior inizializzato come N(0,metà intervallo)
                w_cov = data["w_cov"] or np.eye(2)*50.0 # Prior inizializzato come N(0,metà intervallo)
                w_mean = np.array(w_mean)
                w_cov  = np.array(w_cov)
                
        #init_sim()
        candidate_paths = []

        for i in range(len(parameters_set)):
            parameters = parameters_set[i] # ottiene l'n-esimo dizionario di parametri
            print(f"Parameters of iteration {i}: {parameters}")
            #scene, ur5e, becher, becher2, liquid, dt = generate_sim(parameters,view,liq,debug,record) # genera l'ambiente di simulazione
            dt=0.01
            for j in range(M):
                theta_f =  np.deg2rad(parameters["theta_f"]) #np.pi * 0.48
                num_wp = int(parameters["num_wp"]) #int(10/dt)
                paths = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
                # paths = plan_path(
                #     ur5e, 
                #     theta_f,
                #     parameters,
                #     timeout=5.0, 
                #     smooth_path=True, 
                #     num_waypoints=num_wp, 
                #     ignore_collision=False, 
                #     planner= "RRTStar", # "RRT", "RRTConnect", "RRTstar", "InformedRRTStar"
                #     debug=debug,
                # )
                # path_debug = scene.draw_debug_path(torch.from_numpy(paths["all"]), ur5e)
                # fake_sim(ur5e, paths, scene, path_debug)
                candidate_paths.append(paths)

        # Trova best path e salva params
        best_path = None
        best_success_rate=0

        if not os.path.exists("/tmp/threshold.yaml"):
            threshold=0.1
        else:
            with open("/tmp/threshold.yaml", "r") as f:
                threshold = yaml.safe_load(f)
                if threshold is None:
                    threshold=0.1

        for i,path in enumerate(candidate_paths):
            success=0
            successes = np.zeros(len(parameters_set))
            scores = np.zeros(len(parameters_set))
            for j,parameters in enumerate(parameters_set):
                # score = simulate_action(ur5e, parameters, paths, scene, becher, becher2, liquid, liq)
                score = compute_fake_reward(parameters)
                scores[j]=score
                if is_success(score,threshold):
                    success+=1
                    successes[j]=1
            success_rate=success/len(parameters_set)

            if success_rate >= best_success_rate:
                best_path = path
                best_successes = successes.copy()
                best_scores=scores.copy()
                best_success_rate = success_rate
        
        if best_success_rate < delta:
            self.get_logger().info("Nessuna traiettoria soddisfa il delta succ")
            response.success=False
            return response
        print("Esiste traj che soddisfa req succ")

        exec_path=best_path #["all"]
        n_points = len(exec_path)
        time = np.linspace(0, (n_points - 1) * dt, n_points).tolist()
        #best_path["time"] = time

        new_threshold = min(max(np.mean(best_scores), threshold+0.0001),3.98)

        # Converti in formato compatibile con .yaml e salva
        best_path=self.to_builtin(best_path)
        parameters=self.to_builtin(parameters_set)
        successes=self.to_builtin(best_successes)
        scores=self.to_builtin(best_scores)
        new_threshold=self.to_builtin(new_threshold)

        try:
            with open("/tmp/best_path.yaml", "w") as f:
                yaml.safe_dump({"best_path": best_path}, f, sort_keys=False)
            with open("/tmp/parameters.yaml", "w") as f:
                yaml.safe_dump({"parameters": parameters}, f, sort_keys=False)
            with open("/tmp/scores.yaml", "w") as f:
                yaml.safe_dump({"scores": successes}, f, sort_keys=False)
            with open("/tmp/scores_history.yaml", "a") as f:
                yaml.dump({"scores": scores}, f, explicit_start=True, sort_keys=False)
            with open("/tmp/threshold.yaml", "w") as f:
                yaml.safe_dump(new_threshold, f, sort_keys=False)   

        except Exception as e:
            self.get_logger().error(f"Errore salvataggio YAML: {e}")
            response.success = False
            return response
        
        response.success = True
        flat_best_path = exec_path #[x for wp in exec_path for x in wp]
        response.best_path = flat_best_path
        response.time = time
        return response

    def plan_path_callback(self, request, response):

        liq=True
        record=False
        debug=False   
        view=False

        if view:
            N = 1                    # Numero di modelli simulati (iniziale)
            M = 1                    # Numero di traiettorie
            delta = 1/N              # Threshold di successo
        else:
            N = 15                    # Numero di modelli simulati (iniziale)
            M = 3                     # Numero di traiettorie
            delta = 1/N               # Threshold di successo
    
        if request.no_params == True:
            PARAMS_FILE = "/tmp/parameters.yaml"

            if not os.path.exists(PARAMS_FILE):
                response.success = False
                return response
            else:
                with open(PARAMS_FILE, "r") as f:
                    data = yaml.safe_load(f)
                if "parameters" not in data:
                    raise RuntimeError("File init_parameters.yaml non contiene chiave 'parameters'")
                parameters_set=data["parameters"]
                if request.target_vol is not None:
                    for p in parameters_set:
                        p["vol_target"]=request.target_vol
                for p in parameters_set:
                    p["err_target"]=5e-6
                    # TODO sistemare aggiornamento params
        else:
            req_parameters = {
                "pos_init_cont": list(request.pos_init_cont),
                "pos_cont_goal": list(request.pos_cont_goal),
                "pos_init_ee": list(request.pos_init_ee),
                "pos_grip_ee":list(request.pos_grip_ee),
                "offset": list(request.offset),
                "dCoR": [0.0, -0.015, 0.04],
                "vol_init": request.init_vol, #2e-5, +-MAE
                "densità": 998.0,
                "viscosità": 0.001,
                "tens_sup": 0.072,
                "vol_target": request.target_vol, #0.75e-5,
                "err_target": 5e-6,
                "theta_f": request.theta_f, #+-15°
                "num_wp": int(request.num_wp),
            }
            tolerances = {
                "pos_init_cont": [
                    (0.01, 0.01),  # x: ±1.0 cm
                    (0.0, 0.0),    # y: ±0.0 cm
                    (0.0, 0.0),    # z: ±0.0 cm
                ],
                "pos_cont_goal": [
                    (0.015, 0.015),  # x: ±1.5 cm
                    (0.015, 0.015),  # y: ±1.5 cm
                    (0.0, 0.0),      # z: ±0.0 cm
                ],
                "pos_init_ee": [
                    (0.00, 0.00),    # x: fisso
                    (0.00, 0.00),    # y: fisso
                    (0.00, 0.00),    # z: fisso
                    (0.00, 0.00),    # w: fisso
                    (0.00, 0.00),    # x: fisso
                    (0.00, 0.00),    # y: fisso
                    (0.00, 0.00),    # z: fisso
                ],
                "pos_grip_ee": [
                    (0.00, 0.00),    # x: fisso
                    (0.00, 0.00),    # y: fisso
                    (0.00, 0.00),    # z: fisso
                    (0.00, 0.00),    # w: fisso
                    (0.00, 0.00),    # x: fisso
                    (0.00, 0.00),    # y: fisso
                    (0.00, 0.00),    # z: fisso
                ],
                "offset": [
                    (0.01, 0.01),  # x: ±1 cm (0.15)
                    (0.0, 0.0),    # y: ±0 cm (0.0) offset bloccato (le pinze riportano al centro quando chiuse) 
                    (0.0, 0.0),    #  ±0 cm (0.04) offset bloccato (le pinze riportano al centro quando chiuse)
                ],
                "dCoR": [
                    (0.001, 0.001),  # componente 1: ±1 mm
                    (0.01, 0.01),    # componente 2: ±1 cm
                    (0.01, 0.01),    # componente 3: ±1 cm
                ],
                "viscosità": (0.00025, 0.00025),  # ±25% intorno a 0.001 Pa·s
                "densità": (3.0, 3.0),          # ±3 kg/m^3
                "tens_sup": (0.002, 0.001),     # -0.002 / +0.001 N/m (gamme tipiche 0.070–0.073)
                "vol_init": ( 1.5e-5, 1.5e-5),  # ±1e-5 m^3 (15ml)
                "vol_target": (0.0, 0.0),       # no tol, è scelta
                "err_target": (0.0, 0.0),       # vincolo rigido
                "theta_f": (10.0, 10.0),        # ±10°
                "num_wp": ("rel", 0.2, 0.2),    # ±20%
            }
            
            parameters_range=self._make_parameters_range(req_parameters,tolerances)
            parameters_set=[]
            for _ in range(N):
                parameters_set.append(generate_parameters(parameters_range)) 

            tolerances_save=self.to_builtin(tolerances.copy())
            try:
                with open("/tmp/tolerances.yaml", "w") as f:
                    yaml.safe_dump({"tolerances":tolerances_save}, f, sort_keys=False)
            except Exception as e:
                self.get_logger().error(f"Errore salvataggio YAML: {e}")
                response.success = False
                return response

        iter_file = "/tmp/iter.yaml"
        if not os.path.exists(iter_file):
            k=0
        else:
            with open(iter_file, "r") as f:
                k = yaml.safe_load(f)

        file_theta = "/tmp/TStheta.yaml"
        if not os.path.exists(file_theta):
            x_hist_theta = []
            current_x_theta =  (80,90,100)
            y_hist_theta = []
            w_mean_theta = np.zeros(2)
            w_cov_theta = np.eye(2)*10.0 
        else:
            with open(file_theta, "r") as f:
                data = yaml.safe_load(f)
            x_hist_theta = data["x_hist"] or []
            current_x_theta = tuple(data["new_x"]) or (80,90,100)
            y_hist_theta = data["y_hist"] or [] # lista di 1=success,0=failure
            w_mean_theta = data["w_mean"] or np.zeros(2) # Prior inizializzato come N(0,metà intervallo)
            w_cov_theta = data["w_cov"] or np.eye(2)*10.0 # Prior inizializzato come N(0,metà intervallo)
            w_mean_theta = np.array(w_mean_theta)
            w_cov_theta  = np.array(w_cov_theta)

        file_num_wp = "/tmp/TSnum_wp.yaml"
        if not os.path.exists(file_num_wp):
            x_hist_num_wp = []
            current_x_num_wp =  (300, 350, 400)
            y_hist_num_wp = []
            w_mean_num_wp = np.zeros(2)
            w_cov_num_wp = np.eye(2)*50.0 
        else:
            with open(file_num_wp, "r") as f:
                data = yaml.safe_load(f)
            x_hist_num_wp = data["x_hist"] or []
            current_x_num_wp = tuple(data["new_x"]) or (300, 350, 400)
            y_hist_num_wp = data["y_hist"] or [] # lista di 1=success,0=failure
            w_mean_num_wp = data["w_mean"] or np.zeros(2) # Prior inizializzato come N(0,metà intervallo)
            w_cov_num_wp = data["w_cov"] or np.eye(2)*50.0 # Prior inizializzato come N(0,metà intervallo)
            w_mean_num_wp = np.array(w_mean_num_wp)
            w_cov_num_wp  = np.array(w_cov_num_wp)


        #init_sim()
        candidate_paths = []
        num_wp = int(best_theta_bayes(w_mean_num_wp, w_cov_num_wp, x_min=300, x_max=400))
        theta_f = np.deg2rad(best_theta_bayes(w_mean_theta, w_cov_theta, x_min=80, x_max=100))

        for i in range(len(parameters_set)):
            parameters = parameters_set[i] # ottiene l'n-esimo dizionario di parametri
            print(f"Parameters of iteration {i}: {parameters}")
            #scene, ur5e, becher, becher2, liquid, dt = generate_sim(parameters,view,liq,debug,record) # genera l'ambiente di simulazione
            dt=0.01
            for j in range(M):
                
                if k % 2 == 0:
                    theta_f = np.deg2rad(current_x_theta[j]) #np.deg2rad(parameters["theta_f"]) # 80 - 100 passo 2 (10)
                else:
                    num_wp = int(current_x_num_wp[j]) #int(parameters["num_wp"]) # 300 - 400 passo 10 (20)

                paths = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
                # paths = plan_path(
                #     ur5e, 
                #     theta_f,
                #     parameters,
                #     timeout=5.0, 
                #     smooth_path=True, 
                #     num_waypoints=num_wp, 
                #     ignore_collision=False, 
                #     planner= "RRTStar", # "RRT", "RRTConnect", "RRTstar", "InformedRRTStar"
                #     debug=debug,
                # )
                # path_debug = scene.draw_debug_path(torch.from_numpy(paths["all"]), ur5e)
                # fake_sim(ur5e, paths, scene, path_debug)
                candidate_paths.append(paths)

        # Trova best path e salva params
        best_path = None
        best_success_rate=0

        if not os.path.exists("/tmp/threshold.yaml"):
            threshold=0.1
        else:
            with open("/tmp/threshold.yaml", "r") as f:
                threshold = yaml.safe_load(f)
                if threshold is None:
                    threshold=0.1

        for i,path in enumerate(candidate_paths):
            success=0
            successes = np.zeros(len(parameters_set))
            scores = np.zeros(len(parameters_set))
            for j,parameters in enumerate(parameters_set):
                #scene, ur5e, becher, becher2, liquid, dt = generate_sim(parameters,view,liq,debug,record)
                #score = simulate_action(ur5e, parameters, path, scene, becher, becher2, liquid, liq, dt)
                
                if k % 2 == 0:
                    theta_f=np.deg2rad(current_x_theta[i % M])
                else:
                    num_wp=int(current_x_num_wp[i % M])
                score = compute_fake_reward(parameters, theta_f, num_wp)

                scores[j]=score
                if is_success(score,threshold):
                    success+=1
                    successes[j]=1
            success_rate=success/len(parameters_set)
            # def di miglior path
            if success_rate >= best_success_rate:
                best_path = path
                best_successes = successes.copy()
                best_scores=scores.copy()
                best_success_rate = success_rate
            # se il path è buono su threshold traj allora è buono davvero
            success_path=1 if success_rate > 0.7 else 0
            
            # Aggiornamento TS 
            if k % 2 == 0:
                # Append liste
                y_hist_theta.append(success_path) 
                x_hist_theta.append(current_x_theta[i % M])
                # update posterior
                w_mean_theta_new, w_cov_theta_new = update_w(np.array(x_hist_theta), np.array(y_hist_theta), w_mean_theta, w_cov_theta)
                # new sample
                x_next_theta = sample_x_TS(w_mean_theta_new, w_cov_theta_new, x_min=80, x_max=100, M=M)
                
                state = {
                    "x_hist": self.to_builtin(x_hist_theta),
                    "y_hist": self.to_builtin(y_hist_theta),
                    "w_mean": self.to_builtin(w_mean_theta_new),
                    "w_cov":  self.to_builtin(w_cov_theta_new),
                    "new_x": self.to_builtin(x_next_theta),
                }
                with open(file_theta, "w") as f:
                    yaml.safe_dump(state, f, sort_keys=False)
            else:
                # Append liste
                y_hist_num_wp.append(success_path) 
                x_hist_num_wp.append(current_x_num_wp[i % M])
                # update posterior
                w_mean_num_wp_new, w_cov_num_wp_new = update_w(np.array(x_hist_num_wp), np.array(y_hist_num_wp), w_mean_num_wp, w_cov_num_wp)
                # new sample
                x_next_num_wp = sample_x_TS(w_mean_num_wp_new, w_cov_num_wp_new, x_min=300, x_max=400, M=M)
                state = {
                    "x_hist": self.to_builtin(x_hist_num_wp),
                    "y_hist": self.to_builtin(y_hist_num_wp),
                    "w_mean": self.to_builtin(w_mean_num_wp_new),
                    "w_cov":  self.to_builtin(w_cov_num_wp_new),
                    "new_x": self.to_builtin(x_next_num_wp),
                }
                with open(file_num_wp, "w") as f:
                    yaml.safe_dump(state, f, sort_keys=False)
                
            k+=1
            with open("/tmp/iter.yaml", "w") as f:
                yaml.safe_dump(k, f)
                
        if best_success_rate < delta:
            self.get_logger().info("Nessuna traiettoria soddisfa il delta succ")
            response.success=False
            return response
        print("Esiste traj che soddisfa req succ")

        exec_path=best_path #["all"]
        n_points = len(exec_path)
        time = np.linspace(0, (n_points - 1) * dt, n_points).tolist()
        #best_path["time"] = time

        new_threshold = min(max(np.mean(best_scores), threshold+0.0001),5.98)

        # Converti in formato compatibile con .yaml e salva
        best_path=self.to_builtin(best_path)
        parameters=self.to_builtin(parameters_set)
        successes=self.to_builtin(best_successes)
        scores=self.to_builtin(best_scores)
        new_threshold=self.to_builtin(new_threshold)

        try:
            with open("/tmp/best_path.yaml", "w") as f:
                yaml.safe_dump({"best_path": best_path}, f, sort_keys=False)
            with open("/tmp/parameters.yaml", "w") as f:
                yaml.safe_dump({"parameters": parameters}, f, sort_keys=False)
            with open("/tmp/scores.yaml", "w") as f:
                yaml.safe_dump({"scores": successes}, f, sort_keys=False)
            with open("/tmp/scores_history.yaml", "a") as f:
                yaml.dump({"scores": scores}, f, explicit_start=True, sort_keys=False)
            with open("/tmp/threshold.yaml", "w") as f:
                yaml.safe_dump(new_threshold, f, sort_keys=False)   

        except Exception as e:
            self.get_logger().error(f"Errore salvataggio YAML: {e}")
            response.success = False
            return response
        
        response.success = True
        flat_best_path = exec_path # [x for wp in exec_path for x in wp]
        response.best_path = flat_best_path
        response.time = time
        return response
      
       
def main(args=None):
    rclpy.init(args=args)
    node = PathPlannerService()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
