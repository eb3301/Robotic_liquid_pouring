# import os
# os.environ['PYOPENGL_PLATFORM'] = 'glx'
# os.environ['MUJOCO_GL'] = 'glx'
# os.environ['PYOPENGL_PLATFORM'] = 'egl'
# os.environ['MUJOCO_GL'] = 'egl'
import os
clear = lambda: os.system('clear')
clear()
import genesis as gs
import numpy as np
import trimesh
import random 
import torch
from ompl import util as ou
from ompl import base as ob
from ompl import geometric as og
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

 
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

def set_liquid_pose(becher, liquid, scene): # da sistemare
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
    definisce il range entro cui campionare i parametri per la simulazione fisica
    """

    parameters_range={
    "viscosità":[0.0008,0.0012], # range viscosità liquido
    "densità":[995.0,1001.0], # range densità liquido
    "tens_sup":[0.070,0.073], # range di tensione superficiale 
    "vol_init":[1e-5,5e-5],  # da ROS valore medio
    "vol_target":[0.5e-5,1e-5], # da def
    "pos_init_cont":[
        [0.8,0.9],
        [0.15,0.25],
        [0.92,0.92]
    ], # da ROS valore medio
    "pos_init_ee":[
        [0.4,0.5],
        [0.10,0.3],
        [0.97, 1.1],
        [0.5,0.5],
        [0.5,0.5],
        [0.5,0.5],
        [0.5,0.5],
    ], # da ROS, no errore (usata solo se approach = True)
    "pos_cont_goal":[
        [0.8,0.9],
        [0.65,0.75],
        [0.92,0.92]
    ], # da ROS valore medio
    "offset":[
        [0.0,0.0], # 0.0 (non serve modificarlo perché comunque andrà a 0)
        [-0.05,-0.03], # -0.04
        [0.12,0.13], # 0.13
    ],
    "dCoR":[
        [-0.001,0.001], # 0.0
        [-0.02,0.00], #-0.01
        [0.03,0.05], #+0.04 
    ],
    "err_target":[5e-6,5e-6],
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
        logger_verbose_time = 'warning',
    )

def generate_sim(parameters, view=False, liq=True, debug=False, video=False, approach=False):    
    ########################## create a scene ##########################
    DIR="/home/barutta/Robotic_liquid_pouring"
    dt=1e-2
    global scene
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=dt,
            substeps= 10000*dt,  # Increased substeps for better stability
            gravity=(0, 0, -9.81),
        ),
        rigid_options=gs.options.RigidOptions(
        enable_collision=True,
        enable_self_collision=True,
        enable_adjacent_collision=False,
        constraint_timeconst=0.0001,
        max_dynamic_constraints=10,
        ),
        sph_options=gs.options.SPHOptions(
            # position of the bounding box for the liquid
            lower_bound   = (-1.5, -1.5, 0.0), 
            upper_bound   = (1.5, 1.5, 2),
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
    mat_rigid = gs.materials.Rigid(coup_friction=0.1,
                                   coup_softness=0.0001,
                                   coup_restitution=0.001,
                                   sdf_cell_size=0.0001,
                                   sdf_min_res=32,
                                   sdf_max_res=512)
    
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

    if approach:
        contpos= (parameters['pos_init_cont'][0],parameters['pos_init_cont'][1],parameters['pos_init_cont'][2]) # (0.85,0.2, 0.92) # Initial position
        container_scale = 0.015
        container_mesh_path = DIR + '/becher/becher.obj'

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
        contpos= (parameters['offset'][0],parameters['offset'][1],parameters['offset'][2]) #np.array([0.0,-0.04,0.13]) # Offset di presa tool0 --> becher
        container_scale = 0.015
        container_mesh_path = DIR + '/becher/becher1.obj'

        becher = scene.add_entity(
            gs.morphs.Mesh(
                file=container_mesh_path,
                fixed=True,
                pos=contpos,
                euler=(0, 0, 0),
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

    contpos2= (parameters['pos_cont_goal'][0],parameters['pos_cont_goal'][1],parameters['pos_cont_goal'][2])
    container_scale2 = 0.013
    container_mesh_path2 = DIR + '/becher/becher.obj'

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
    liquid_radius = min(container_size[0], container_size[1])/2*0.5
    init_volume = parameters['vol_init']
    liquid_height = init_volume/(np.pi*liquid_radius**2)
    #liquid_height = container_size[2]*container_scale*np.sqrt(2)*0.5
    #print(liquid_radius, liquid_height)
    # Position liquid relative to container center
    liqpos = (parameters['pos_init_cont'][0],parameters['pos_init_cont'][1],parameters['pos_init_cont'][2]+container_size[2]+liquid_height/2) 

    if liq:
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
        kp = np.array([4500, 4500, 4500, 3500, 3500, 3500, 20, 20]),
        dofs_idx_local = dofs_idx,
    )
    # Set dofs kv: (Increase velocity gains for better damping)
    ur5e.set_dofs_kv(
        kv = np.array([450,450,450,350,350,350,2,2]),
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
        init_quat=np.array([parameters['pos_init_ee'][3], parameters['pos_init_ee'][4], parameters['pos_init_ee'][5], parameters['pos_init_ee'][6]])
    else:     
        end_effector = ur5e.get_link("tool0")
        x_shift=0.13
        z_min=0.967
        quat_orizz = np.array([0.5,0.5,0.5,0.5])
        init_pos = np.array([parameters['pos_init_cont'][0],parameters['pos_init_cont'][1],parameters['pos_init_cont'][2]])
        init_pos[0]-=x_shift 
        init_pos[2]+=container_size[2]-0.01
        init_pos[2]=max(init_pos[2],z_min)
        init_quat = quat_orizz
        
    # Use inverse kinematics to get joint angles
    init_qpos = ur5e.inverse_kinematics(
                link=end_effector,
                pos=init_pos,
                quat=init_quat,
        )
    if approach:
        init_qpos[-2:]=-0.02
    else:
        init_qpos[-2:]=0.005

    ur5e.set_dofs_position(init_qpos)
    scene.visualizer.update(force=True, auto=True)

    # Reach steady state of the liquid
    if liq:
        for i in range(100):
            ur5e.control_dofs_position(
                position=init_qpos,
                dofs_idx_local=dofs_idx,
            )
            scene.step()
            # cam.render()
        print("Scene ready to use (steady state reached)")

    global init_scene
    init_scene = scene.get_state()

    return scene, ur5e, becher, becher2, liquid, dt

def reset_sim(scene, ur5e, becher, becher2, liquid, parameters):
    # Set initial robot position
    dofs_idx = []
    for joint in ur5e.joints:
        if joint.name not in ["joint_world","flange-tool0","robotiq_hande_base_joint"]:
            dofs_idx.extend(joint.dofs_idx_local)
    end_effector = ur5e.get_link("tool0")
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

    set_liquid_pose(becher, liquid, scene)

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

def liq_ang(particles, top_percent=10):
    """
    Dato l'array di particelle, stima la normale alla superficie 
    libera del liquido, poi calcola la rotazione dell'ee per 
    mantenere la superficie libera parallela al fondo del contenitore.
    """
    z_vals = particles[:, 2]
    threshold = np.percentile(z_vals, 100 - top_percent) # solo part vicine a sup
    surface_particles = particles[z_vals >= threshold]
    centroid = surface_particles.mean(axis=0)
    X = surface_particles - centroid
    _, _, vh = np.linalg.svd(X) # PCA/SVD per stimare il piano
    normal = vh[-1] # normale al piano è ult vect sing

    if normal[2] < 0: # per avere direzione verso l'alto coerente con asse z
        normal = -normal
    n = normal / np.linalg.norm(normal)
    
    roll = np.arctan2(n[1], n[2])
    # Costruisci rotazione solo attorno a X
    r_roll = R.from_euler('x', -roll)  # negativo se vogliamo compensare
    quat = r_roll.as_quat()  # [x, y, z, w]
    quat_wxyz = np.roll(quat, 1) 

    # z_axis_tool = estimate_liquid_normal(particles)
    # world_x = np.array([1.0, 0.0, 0.0])
    # if np.allclose(np.abs(np.dot(world_x, z_axis_tool)), 1.0, atol=1e-3):
    #     world_x = np.array([0.0, 1.0, 0.0])

    # y_axis_tool = np.cross(z_axis_tool, world_x)
    # y_axis_tool /= np.linalg.norm(y_axis_tool)
    # x_axis_tool = np.cross(y_axis_tool, z_axis_tool)
    # x_axis_tool /= np.linalg.norm(x_axis_tool)

    # R_target = np.vstack([x_axis_tool, y_axis_tool, z_axis_tool]).T # matrice di rotazione
    # r_target = R.from_matrix(R_target)
    # quat_target = r_target.as_quat() # [x, y, z, w]
    # quat_wxyz = np.roll(quat_target, 1)  # porta l'ultimo elemento in prima posizione

    return quat_wxyz

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
    ):
    old=False
    path=np.empty((0, 8))
    print(f"planning started")
    # trasforma tutti i path in array numpy
    #################################
    x_shift=0.13
    z_min=0.967
    quat_orizz = np.array([0.5,0.5,0.5,0.5])
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
        )
        if not valid:  # se invalido
            raise RuntimeError(f"path da posizione iniziale a posizione grasping è invalido")
        path1=path1.cpu().numpy()
        path1=np.concatenate((path01,path1))
        path = np.concatenate((path, path1))
    else:
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
        collisions1 = ur5e.detect_collision()
        if debug: print(f"Collisioni 1: {collisions1}")
   
    ################################# 
    # q2 (sollevam)
    pos2 = pos1
    pos2[2]+=0.20
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
    collisions2 = ur5e.detect_collision()
    if debug: print(f"Collisioni 2: {collisions2}")

    # Sollevamento (1->2): movimento verticale lungo Z
    path2, valid = ur5e.plan_path(
        #ee_link_name="tool0",
        qpos_goal=q2,
        qpos_start=q1,
        timeout=timeout,
        num_waypoints=int(num_waypoints/2),
        smooth_path=smooth_path,
        ignore_collision=ignore_collision,
        planner=planner,
        return_valid_mask=return_valid_mask,
    )
    if not valid:  
        raise RuntimeError(f"path di sollevamento è invalido")
    path2=path2.cpu().numpy()
    path = np.concatenate((path, path2))
   
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
    )
    if not valid:  
        raise RuntimeError(f"path di trasporto è invalido")
    path = np.concatenate((path, path3))
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
    
    pos4[2]=max(pos4[2],z_min)
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
    collisions4 = ur5e.detect_collision()
    if debug: print(f"Collisioni 4: {collisions4}")

    # Posizione pre versamento (3->4): movimento verticale lungo Z
    path4, valid = ur5e.plan_path(
        #ee_link_name="tool0",
        qpos_goal=q4,
        qpos_start=q3,
        timeout=timeout,
        num_waypoints=int(num_waypoints/2),
        smooth_path=smooth_path,
        ignore_collision=ignore_collision,
        planner=planner,
        return_valid_mask=return_valid_mask,
    )
    if not valid:  
        raise RuntimeError(f"path da fine trasporto a preversamento è invalido")
    path4=path4.cpu().numpy()
    path = np.concatenate((path, path4))   

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
        collisions5 = ur5e.detect_collision()
        if debug: print(f"Collisioni 5: {collisions5}")
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
        collisions6 = ur5e.detect_collision()
        if debug:
            print(f"Collisioni 6: {collisions6}")

        path = np.concatenate((path, path6))

    else:
        ######################################
        # Versamento (4->5)
        CoR3D = np.array([
            parameters['pos_cont_goal'][0] + parameters['dCoR'][0], # 0.0
            parameters['pos_cont_goal'][1] + parameters['dCoR'][1], # - 0.01 
            parameters['pos_cont_goal'][2] + parameters['dCoR'][2], # + 0.04
        ])
        p_tcp0 = pos4.copy()
        scene.draw_debug_sphere(CoR3D, radius=0.005, color=(1.0, 0.0, 0.0, 1.0))
        axis_world = np.array([1.0, 0.0, 0.0]) # asse rot x

        # 3) Orientazione iniziale del tool in pre-pour (q4/pos4/quat4)
        R0 = R.from_quat(quat4) # matrice rot init
        l = R0.inv().apply(CoR3D - p_tcp0) # offset tool0 --> CoR3D

        path5 = []
        n_steps = int(num_waypoints/2.5)
        for theta in np.linspace(0, theta_f, n_steps):
            R_theta = R.from_rotvec(theta * axis_world) * R0 # matrice rotazione lungo x
            quat5 = R_theta.as_quat()
            delta_pos=R_theta.apply(l)
            p_tcp = CoR3D - delta_pos
            p_tcp[2] = max(p_tcp[2], z_min) 
            lip_height = parameters['pos_cont_goal'][2] + container2_size[2]+0.05
            p_tcp[2] = max(p_tcp[2], lip_height)


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
        collisions5 = ur5e.detect_collision()
        if debug: print(f"Collisioni 5: {collisions5}")
        path = np.concatenate((path, path5))

        ###########################################
        # Ritorno dal versamento (5->6)
        path6 = []
        for theta in np.linspace(theta_f, 0.0, n_steps):
            R_theta = R.from_rotvec(theta * axis_world) * R0
            quat6 = R_theta.as_quat()

            p_tcp = CoR3D - R_theta.apply(l)
            p_tcp[2] = max(p_tcp[2], z_min)
            lip_height = parameters['pos_cont_goal'][2] + container2_size[2]+0.05
            p_tcp[2] = max(p_tcp[2], lip_height)

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
        collisions6 = ur5e.detect_collision()
        if debug: print(f"Collisioni 6: {collisions6}")

        path = np.concatenate((path, path6))

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
    collisions7 = ur5e.detect_collision()
    if debug: print(f"Collisioni 7: {collisions7}")
    # Termina rilasciando contenitore sul tavolo
    path7, valid = ur5e.plan_path(
        #ee_link_name="tool0",
        qpos_goal=q7,
        qpos_start=q6,
        timeout=timeout,
        num_waypoints=int(num_waypoints/10),
        smooth_path=smooth_path,
        ignore_collision=ignore_collision,
        planner=planner,
        return_valid_mask=return_valid_mask,
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
    
def plot_path(paths,ur5e):
    path=paths["all"]
    pos=[]
    for idx,wp in enumerate(path):
        wp_tensor = torch.tensor(wp, dtype=torch.float32)
        pos_wp, _ = ur5e.forward_kinematics(wp_tensor)
        pos.append(pos_wp[-1])
    pos = np.array(pos) 
    x=pos[:,0]
    y=pos[:,1]
    z=pos[:,2]
    # Crea figura 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plotta il path
    ax.plot(x, y, z, marker='o')

    # Label assi
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show(block=True) 
    quit() 

def plot_paths(paths, ur5e):
    # Colori differenti (se i path sono più di 7 si ricicla la palette)
    colors = plt.cm.tab10.colors  
    
    # Crea figura 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    all_pt=[]
    for i, (name, path) in enumerate(paths.items()):
        if name != "all":
            pos = []
            for wp in path:
                wp_tensor = torch.as_tensor(wp, dtype=torch.float32).clone()

                pos_wp, _ = ur5e.forward_kinematics(wp_tensor)
                pos.append(pos_wp[-1])  # prendo la posizione dell’EE
            pos = np.array(pos)
            all_pt.append(pos)
            x, y, z = pos[:,0], pos[:,1], pos[:,2]
            ax.plot(x, y, z, marker='o', color=colors[i % len(colors)], label=name)

    # Concateno tutti i punti per determinare i limiti globali
    all_pt = np.vstack(all_pt)
    x_min, y_min, z_min = all_pt.min(axis=0)
    x_max, y_max, z_max = all_pt.max(axis=0)

    # Calcolo il range massimo
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2.0
    mid_x = (x_max + x_min) / 2.0
    mid_y = (y_max + y_min) / 2.0
    mid_z = (z_max + z_min) / 2.0

    # Imposto stessi limiti per tutti gli assi
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Label assi
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    plt.show(block=True)
    #quit()

def simulate_action(ur5e, parameters, paths, scene, becher, becher2, liquid, liq, approach=False): 
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
    for qpos in path2:
        qpos[-2:]=0.005
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
    
    # Trasporto:
    path3=paths["transport"]
    for wp in path3: 
        if liq:
            pos_wp, _ = ur5e.forward_kinematics(wp)
            particles = np.squeeze(liquid.get_particles())
            quat_wp = liq_ang(particles)
            try:
                qpos = ur5e.inverse_kinematics(
                    link=ur5e.get_link("tool0"),
                    pos=pos_wp[-1],
                    quat=quat_wp
                )
                qpos[-2:]=0.005
            except Exception as e:
                raise RuntimeError(f"errore nella IK liq ang")
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

    # Posizionamento pre pouring:
    path4=paths["pre_pour"]
    for qpos in path4:
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
    if approach:
        for _ in range(10):
            ur5e.control_dofs_position(qpos[:-2], motors_dof)
            ur5e.control_dofs_force(closing_force, fingers_dof)
            scene.step()

    # Pouring:
    path5=paths["pour"]   
    for qpos in path5:
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
    # Unpouring:
    path6=paths["unpour"]   
    for qpos in path6:
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
    # Release:
    path7=paths["release"]
    for qpos in path7:
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
    if liq:
        particles = np.squeeze(liquid.get_particles())
        contpos = np.array(parameters['pos_cont_goal'])
        err = parameters['err_target']
        target_vol=parameters['vol_target']

        # da modificare ass: la media delle particelle prob non coinc con centro del target -> misurare volume effettivo (con bounding box del becher2)
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

    tf=scene.get_state().scene.t
    Dt=tf-t0
    score-=1e-2*Dt
    print(f"Simulation completed")
    return score

def is_success(score, threshold=0.5):
    return score > threshold

def update_parameters(param, scale=0.1): # Aggiunta rumore gaussiano ai parametri
    print("Updating parameters")
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

########################## main ##########################
def main():
    N = 1#20                    # Numero di modelli simulati (iniziale)
    M = 1#5                     # Numero di traiettorie
    delta = 0.7*M             # Threshold di successo 
    MAX_ITERS = 1#10            # Numero massimo di iterazioni
    view=False
    liq=False
    record=False
    debug=False

    parameters_range=obtain_range() # Ottieni range di parametri
    param_real = generate_parameters(parameters_range)  # Simulazione della realtà

    init_sim()

    parameters_set=[]
    # ottiene una lista di dizionari di parametri in modo randomico sui range iniziali dati
    for _ in range(N):
        parameters_set.append(generate_parameters(parameters_range)) 

    for t in range(MAX_ITERS):
        print(f"\n[ITER {t+1}] Belief size: {len(parameters_set)}")

        candidate_paths = [] # da decidere se rimuovere i path vecchi dopo aggiornamento belief set o no !!!
        for i in range(len(parameters_set)):
            parameters = parameters_set[i] # ottiene l'n-esimo dizionario di parametri
            scene, ur5e, becher, becher2, liquid, dt = generate_sim(parameters,view,liq,debug,record) # genera l'ambiente di simulazione
            
            for j in range(M):
                # parametri da cambiare in questo for per creare traiettorie differenti
                # trovare un valore iniziale attorno al quale muoversi (da test)
                theta_f=np.pi*0.48
                # l'obiettivo è massimizzare la velocità
                k=1 # fattore correttivo velocità robot (da far variare durante la simulazione)
                num_wp=int(10/dt)*k # num di wp tale da compiere traiettoria princ in 10s* 

                paths = plan_path(
                    ur5e, 
                    theta_f,
                    parameters,
                    timeout=5.0, 
                    smooth_path=True, 
                    num_waypoints=num_wp, 
                    ignore_collision=False, 
                    planner= "RRTStar", # "RRT", "RRTConnect", "RRTstar", "InformedRRTStar"
                    debug=debug,
                )
                candidate_paths.append(paths)

                if debug:
                    if view:
                        plot_paths(paths,ur5e)
                    path_debug = scene.draw_debug_path(torch.from_numpy(paths["all"]), ur5e)
                    fake_sim(ur5e, paths, scene, path_debug)
                    

        # Valuta ogni traiettoria su ogni set di param
        best_path = None
        best_score = -1e30
        for paths in candidate_paths:
            score = sum(simulate_action(ur5e, parameters, paths, scene, becher, becher2, liquid, liq) for parameters in parameters_set)
            if score > best_score:
                best_score = score
                best_path = paths
        if liq:
            if best_score < delta:
                print("Nessuna traiettoria soddisfa il delta succ")
                break
            else:
                print("Esiste traj che soddisfa req succ")

        # Esegui su realtà (simulata con param_real)
        real_score = simulate_action(ur5e, param_real, best_path, scene, becher, becher2, liquid, liq)
        real_result = is_success(real_score)
        #print(f"Risultato nel mondo reale (parameters*): {real_score}")

        # Aggiorna belief set
        param_new = [p for p in parameters_set if is_success(simulate_action(ur5e, p, best_path, scene, becher, becher2, liquid, liq)) == real_result]

        if len(param_new) == 0:
                print("Tutte le ipotesi eliminate.")
                break
        
        # Resample
        new_samples = [update_parameters(p) for p in param_new] 
        parameters_set = param_new + new_samples

        max_models = 30
        if len(parameters_set) > max_models:
            parameters_set = random.sample(parameters_set, max_models)

if __name__ == "__main__":
    main()