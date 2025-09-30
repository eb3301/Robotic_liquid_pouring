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
        [4500, 4500],
        [4500, 4500],
        [4500, 4500],
        [4500, 4500],
        [4500, 4500],
        [4500, 4500],
        [4500, 4500],
        [4500, 4500],
    ],
    "kv":[
        [500, 500],
        [500, 500],
        [500, 500],
        [300, 300],
        [300, 300],
        [300, 300],
        [100, 100],
        [100, 100],
    ],
    "max_F":[
        [50,50],
        [50,50],
        [50,50],
        [50,50],
        [50,50],
        [50,50],
        [50,50],
        [50,50],
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
        logger_verbose_time = 'warning',
    )

def generate_sim(parameters, view=False, liq=True, debug=False, video=False):    
    ########################## create a scene ##########################
    DIR="/home/edo/thesis"
    dt=1e-2
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=dt,
            substeps=10000*dt,  # Increased substeps for better stability
            gravity=(0, 0, -9.81),
        ),
        rigid_options=gs.options.RigidOptions(
        enable_collision=True,
        enable_self_collision=False,
        enable_adjacent_collision=False,
        # Prevent the rigid contact solver from being too stiff otherwise this would cause large impulse, especially
        # because the simulation timestep must be very small to ensure numerical stability of rigid body dynamics.
        constraint_timeconst=0.02,
        ),
        sph_options=gs.options.SPHOptions(
            # position of the bounding box for the liquid
            lower_bound   = (-1.5, -1.5, 0.0), 
            upper_bound   = (1.5, 1.5, 2),
            particle_size = 0.01,  
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
    plane = scene.add_entity(gs.morphs.Plane())

    ur5e=scene.add_entity(gs.morphs.URDF(
            file = DIR + '/ur5e_urdf/urdf/ur5e_complete.urdf',
            fixed=True,
            pos   = (0, 0, 0),
            euler = (0, 0, 0), # we follow scipy's extrinsic x-y-z rotation convention, in degrees,
            # quat  = (1.0, 0.0, 0.0, 0.0), # we use w-x-y-z convention for quaternions,
            # contype=0b001,
            # conaffinity=0b001,
            scale = 1.0,
            links_to_keep=['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint', 'hand_e_link','hande_left_finger_joint', 'hande_right_finger_joint','tool0'],
        ),
        material=gs.materials.Rigid(),
        # vis_mode = "collision",
        # visualize_contact=True,
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
        print(f"joint names: {jnt_names}, joint indexes: {dofs_idx}")
        print(f"link names {link_names}, link indexes: {link_idx}")

    contpos= (parameters['pos_init_cont'][0],parameters['pos_init_cont'][1],parameters['pos_init_cont'][2]) # (0.85,0.2, 0.92) # Initial position
    container_scale = 0.013
    container_mesh_path = DIR + '/becher/becher.obj'

    becher = scene.add_entity(
        gs.morphs.Mesh(
            file=container_mesh_path,
            fixed=False,
            pos=contpos,
            euler=(90, 0, 180),
            scale=container_scale,
            decimate=False,
            convexify=False,
            decompose_object_error_threshold=1e-3,
            # decompose_nonconvex=True,
            # contype=0b011,
            # conaffinity=0b011,
            coacd_options=gs.options.CoacdOptions(),
            merge_submeshes_for_collision=True,
        ),
        material=gs.materials.Rigid(needs_coup=True),
        # vis_mode = "collision",
        # visualize_contact=True,
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
            convexify=False,
            decompose_object_error_threshold=1e-3,
            #decompose_nonconvex=True,
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
    liqpos = (contpos[0], contpos[1], contpos[2] + liquid_height)

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
    #     ur5e.set_friction_ratio(parameters['friction'][i],link_idx)
    
    ########################## main ##########################

    # start camera recording. Once this is started, all the rgb images rendered will be recorded internally
    if video==True:
        cam.start_recording()

    # Set initial robot position
    end_effector = ur5e.get_link("tool0")
    init_pos=np.array([parameters['pos_init_ee'][0], parameters['pos_init_ee'][1],parameters['pos_init_ee'][2]])
    init_quat=np.array([parameters['pos_init_ee'][3], parameters['pos_init_ee'][4], parameters['pos_init_ee'][5], parameters['pos_init_ee'][6]])
        
    # Use inverse kinematics to get joint angles
    init_qpos = ur5e.inverse_kinematics(
                link=end_effector,
                pos=init_pos,
                quat=init_quat,
        )
    ur5e.set_dofs_position(init_qpos)

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


def tensor_to_cpu(x):
    if isinstance(x, torch.Tensor):
        x = x.cpu()
    return x

def tensor_to_array(x):
    return np.array(tensor_to_cpu(x))




########################## main ##########################
def main():
    view=True
    liq=False
    record=False
    debug=True

    parameters_range=obtain_range() # Ottieni range di parametri
    param = generate_parameters(parameters_range)  # Simulazione della realtà
    init_sim()
    scene, ur5e, becher, becher2, liquid, dt = generate_sim(param,view,liq,debug,record)
    
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
    motors_dof = dofs_idx[:-2]
    fingers_dof = dofs_idx[-2:]
    print(f"motor dofs {motors_dof}, fing dofs {fingers_dof}")
    q0 = ur5e.get_qpos()

    for i in range(100): 
        ur5e.control_dofs_position(q0, dofs_idx_local=dofs_idx)
        scene.step() 
        print(i)

    for i in range(100):
        ur5e.control_dofs_position(q0[:-2], motors_dof)
        ur5e.control_dofs_force(np.array([-0.5, -0.5]), fingers_dof)
        print(i)
    for i in range(100): scene.step() # per chiudere correttamente  
    
    for i in range(100):
        ur5e.control_dofs_position(q0[:-2], motors_dof)
        ur5e.control_dofs_force(np.array([5, 5]), fingers_dof)
        print(i)
    for i in range(100): scene.step() # per chiudere correttamente
   
if __name__ == "__main__":
    main()