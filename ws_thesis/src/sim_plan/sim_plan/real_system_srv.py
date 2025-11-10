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
    err_max_theta=20

    num_wp_opt=320
    err_num_wp=np.linalg.norm(num_wp-num_wp_opt)
    err_max_num_wp=100

    w_pos, w_cor = 3, 1
    w_theta, w_num_wp = 0.5,0.5

    reward1 = w_pos * max(0,1-pos_err/pos_tol)
    reward2 = w_cor * max(0,1-cor_err/cor_tol)
    reward3 = w_theta * max (0,1-err_theta/err_max_theta)
    reward4 = w_num_wp * max (0,1-err_num_wp/err_max_num_wp)

    print(f"reward cont pos err: {reward1}")
    print(f"reward pos CoR: {reward2}")
    print(f"reward theta: {reward3}")
    print(f"reward num_wp: {reward4}")

    reward = reward1 + reward2 + reward3 + reward4
    w_tot = w_pos + w_cor + w_theta + w_num_wp
    reward/=w_tot
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

class RealSystemService(Node):
    def __init__(self):
        super().__init__('real_system_service')
        self.srv = self.create_service(Simplan, 'real_system', self.real_system_callback)
        self.get_logger().info("Real system service ready")

    def real_system_callback(self, request, response):

        liq=True
        record=False
        debug=False   
        view=False

        parameters = {
                "pos_init_cont": (0.8021465039268282, 0.2847160024669491, 0.9564999991059303),
                "pos_cont_goal": (0.837326238153434, 0.9863513333825724, 0.9607816689610481),
                "pos_init_ee":  (0.12394027698787038, 0.2665910764094347, 1.1638763741892955, -0.5837257860699608, 0.5925713099055904, -0.388471134978874, 0.3965017359886389),
                "pos_grip_ee": (0.6509734785173125, 0.2847438888358813, 0.9764986855761588, -0.5000033337808922, 0.49998929956451965, -0.4999940109139501, 0.5000133554007884),
                "offset": (2.7886369672436295e-05, 0.1511730254095156, -0.01999868647022847),
                "dCoR": [0.0, -0.015, 0.04],
                "vol_init": 40.0, #2e-5, +-MAE
                "densità": 998.0,
                "viscosità": 0.001,
                "tens_sup": 0.072,
                "vol_target": 20.0, #0.75e-5,
                "err_target": 5e-6,
                "theta_f": 87, #+-15°
                "num_wp": 320,
            }
        # Carica parametri planning (non sono quelli ottimali ma è corretto così):
        FILE_CURRENT_PLAN_PARAMS="/tmp/current_plan_params.yaml"
        if not os.path.exists(FILE_CURRENT_PLAN_PARAMS):
                self.get_logger().error("current plan params don't exist")
        else:
            with open(FILE_CURRENT_PLAN_PARAMS, "r") as f: # aperto dopo sim_plan --> un solo valore
                    data_plan = yaml.safe_load(f)
            theta_f = data_plan["current_theta"]
            num_wp = data_plan["current_num_wp"]

        if not os.path.exists("/tmp/threshold.yaml"):
            threshold=0.1
        else:
            with open("/tmp/threshold.yaml", "r") as f:
                threshold = yaml.safe_load(f)
                if threshold is None:
                    threshold=0.1
        
        if not os.path.exists("/tmp/best_path.yaml"):
            self.get_logger().error("best path doesn't exist")
        else:
            with open("/tmp/best_path.yaml", "r") as f:
                path=yaml.safe_load(f)
                path=path #["all"] # TODO riaggiungilo in test
        #################################################################################à
        # Simulazione
        #init_sim()
        #scene, ur5e, becher, becher2, liquid, dt = generate_sim(parameters,view,liq,debug,record) # genera l'ambiente di simulazione
           
        theta_f = np.deg2rad(theta_f)
        num_wp = int(num_wp)
        
        #score = simulate_action(ur5e, parameters, path, scene, becher, becher2, liquid, liq, dt)
        score = compute_fake_reward(parameters, theta_f, num_wp)
        success = is_success(score,threshold)

        response.success=True if success==1 else False
        return response   
       
def main(args=None):
    rclpy.init(args=args)
    node = RealSystemService()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()