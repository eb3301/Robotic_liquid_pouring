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
from drims2_motion_server.motion_client import MotionClient
from geometry_msgs.msg import PoseStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState

def to_numpy_cpu(x):
    """Converte tensor → numpy solo se è su GPU."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x, dtype=np.float32)

def q2moveit(q):
    q=to_numpy_cpu(q)
    q=q[:6]
    q = [float(x) for x in q]
    q[0]-=np.pi
    return q

def generate_sim(parameters, view=False, liq=False, debug=False, video=False, approach=False):    
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
            # progress_bar(i,n, "steady state")
                
            # cam.render()
        #print("Scene ready to use (steady state reached)")
        logger.info("Scene ready to use (steady state reached)")

    global init_scene
    init_scene = scene.get_state()

    return scene, ur5e, becher, becher2, liquid, dt

def to_builtin(obj):
        # numpy scalari -> tipo Python
        if isinstance(obj, np.generic):
            return obj.item()

        # numpy array -> lista Python ricorsiva
        if isinstance(obj, np.ndarray):
            return [to_builtin(x) for x in obj.tolist()]

        # lista -> lista pulita
        if isinstance(obj, list):
            return [to_builtin(x) for x in obj]

        # tupla -> lista (YAML gestisce le liste meglio, ed è ok perdere l'immutabilità)
        if isinstance(obj, tuple):
            return [to_builtin(x) for x in obj]

        # dict -> dict pulito
        if isinstance(obj, dict):
            return {k: to_builtin(v) for k, v in obj.items()}

        # tipi base (int, float, str, bool, None) restano così
        return obj

def plan_pouring(parameters, theta_f, num_waypoints, ur5e):

    x_shift=0.15
    z_min=0.967
    lip_height = parameters['pos_cont_goal'][2] + container2_size[2]+0.07
    quat_orizz = np.array([0.5,-0.5,0.5,-0.5])
    path=np.empty((0, 6))

    pos4 = np.array([parameters['pos_cont_goal'][0],parameters['pos_cont_goal'][1],parameters['pos_cont_goal'][2]])
    pos4[0]-=x_shift
    pos4[1] -= (container2_size[0]/2+0.03)
    pos4[2] += container2_size[2]
    pos4[2]=max(pos4[2],z_min,lip_height)
    quat4 = quat_orizz
    # Versamento (4->5)
    CoR3D = np.array([
        parameters['pos_cont_goal'][0] + parameters['dCoR'][0], # 0.0
        parameters['pos_cont_goal'][1] - 0.005 + parameters['dCoR'][1], # - 0.01 
        parameters['pos_cont_goal'][2] + parameters['dCoR'][2], # + 0.04
    ])
    p_tcp0 = pos4.copy()
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
        q5=q2moveit(q5)

        path5.append(q5)
    path5 = np.asarray(path5, dtype=float)
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
        q6=q2moveit(q6)

        path6.append(q6)
    path6 = np.asarray(path6, dtype=float)
    q6=path6[-1]
    q6 = [float(x) for x in q6]
    path = np.concatenate((path, path6))

    return path

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

def main():
    parameters = {
                "pos_init_cont": [0.8021465039268282, 0.2847160024669491, 0.9564999991059303],
                "pos_cont_goal": [0.746, 0.961, 0.960],
                "pos_init_ee":  (0.12394027698787038, 0.2665910764094347, 1.1638763741892955, -0.5837257860699608, 0.5925713099055904, -0.388471134978874, 0.3965017359886389),
                "pos_grip_ee": (0.6509734785173125, 0.2847438888358813, 0.9764986855761588, -0.5000033337808922, 0.49998929956451965, -0.4999940109139501, 0.5000133554007884),
                "offset": (0, 0.15, -0.02),
                "dCoR": [0.0, 0.0, 0.0],
                "vol_init": 40.0, #2e-5, +-MAE
                "densità": 998.0,
                "viscosità": 0.001,
                "tens_sup": 0.072,
                "vol_target": 20.0, #0.75e-5,
                "err_target": 5e-6,
                "theta_f": 87, #+-15°
                "num_wp": 320,
            }
    theta_f=90
    num_waypoints=320
    init_sim()
    _, ur5e, _, _, _, dt = generate_sim(parameters)
    path=plan_pouring(parameters, theta_f, num_waypoints, ur5e)
    
    with open("/tmp/best_path.yaml", "w") as f:
                yaml.safe_dump({"best_path": to_builtin(path)}, f, sort_keys=False)
  

if __name__ == '__main__':
    main()