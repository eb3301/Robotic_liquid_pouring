import os
clear = lambda: os.system('clear')
clear()
import rclpy
from rclpy.node import Node
from rclpy.logging import get_logger
import yaml
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R
from drims2_motion_server.motion_client import MotionClient
from geometry_msgs.msg import PoseStamped

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

def plan_pouring(parameters, theta_f, num_waypoints):
    DIR="/home/edo/thesis"
    container_scale2 = 0.013
    container_mesh_path2 = DIR + '/becher/becher1.obj'
    container2_mesh = trimesh.load(container_mesh_path2)
    container2_bounds = container2_mesh.bounds
    container2_size = (container2_bounds[1] - container2_bounds[0])*container_scale2
    
    rclpy.init()
    motion_client=MotionClient()
    x_shift=0.15
    z_min=0.967
    lip_height = parameters['pos_cont_goal'][2] + container2_size[2]+0.07
    quat_orizz = np.array([0.5,-0.5,0.5,-0.5])
    path=np.empty((0, 6))

    

    pos4 = np.array([parameters['pos_cont_goal'][0],parameters['pos_cont_goal'][1],parameters['pos_cont_goal'][2]])
    pos4[0]-=0.02 #x_shift
    pos4[1] -= (container2_size[0]/2+0.03)
    pos4[2] += container2_size[2]
    pos4[2]=max(pos4[2],z_min,lip_height)
    quat4 = quat_orizz


    pose_msg = PoseStamped()
    pose_msg.header.frame_id = "world" # relative motion wrt tool0 frame
    pose_msg.pose.position.x = pos4[0]
    pose_msg.pose.position.y = pos4[1]
    pose_msg.pose.position.z = pos4[2]
    pose_msg.pose.orientation.w = quat4[0]
    pose_msg.pose.orientation.x = quat4[1]
    pose_msg.pose.orientation.y = quat4[2]
    pose_msg.pose.orientation.z = quat4[3]

    result = motion_client.move_to_pose(pose_msg)
    
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
    n_new = int(n_steps/5)

    theta_f = np.deg2rad(theta_f) if theta_f>2*np.pi else theta_f
    n_ik=3
    q_old=np.deg2rad(np.array([-30, -116, -129, -120, -30, -180]))
    result = motion_client.move_to_joint(q_old)
    for i,theta in enumerate(np.linspace(0, theta_f, n_new)):

        R_theta = R.from_rotvec(theta * axis_world) * R0 # matrice rotazione lungo x
        quat5 = R_theta.as_quat()
        delta_pos=R_theta.apply(l)
        p_tcp = CoR3D - delta_pos
        p_tcp[2] = max(pos4[2],z_min, lip_height) 

        pose_msg = PoseStamped()
        pose_msg.header.frame_id = "world" # relative motion wrt tool0 frame
        pose_msg.pose.position.x = p_tcp[0]
        pose_msg.pose.position.y = p_tcp[1]
        pose_msg.pose.position.z = p_tcp[2]
        pose_msg.pose.orientation.w = quat5[0]
        pose_msg.pose.orientation.x = quat5[1]
        pose_msg.pose.orientation.y = quat5[2]
        pose_msg.pose.orientation.z = quat5[3]

        if n_ik > 0:
            #best_norm = 1e30
            best_err = 1e30
            q_best = None
            for _ in range(n_ik):
                result, tmp = motion_client.solve_ik(pose=pose_msg)
                if tmp is None:
                    continue
                tmp=np.asarray(tmp,dtype=float)
                #norm = np.linalg.norm(tmp - q_old)
                err = np.sum(np.abs(tmp - q_old))

                # if norm < best_norm:
                #     best_norm = norm
                #     q_best = tmp
                if err < best_err:
                    best_err = err
                    q_best = tmp

            if q_best is None or best_err > np.pi/3:
                raise Warning("Ocio batocio te funziona mia la IK")
                continue
            else:
                q5=q_best
                q_old = q_best.copy()
        else:
            result, q5 = motion_client.solve_ik(pose=pose_msg)
        #result = motion_client.move_to_joint(q5)
        # print(f"pose: {p_tcp}")
        # print(f"joints: {q5}")
        q5=np.asarray(q5,dtype=float)
        path5.append(q5)
    path5 = np.asarray(path5, dtype=float)
    path = np.concatenate((path, path5))
    q6_old = q5
    ###########################################
    # Ritorno dal versamento (5->6)
    path6 = []
    for i,theta in enumerate(np.linspace(theta_f, 0.0, n_new)):

        R_theta = R.from_rotvec(theta * axis_world) * R0
        quat6 = R_theta.as_quat()

        p_tcp = CoR3D - R_theta.apply(l)
        p_tcp[2] = max(pos4[2],z_min,lip_height)

        pose_msg = PoseStamped()
        pose_msg.header.frame_id = "world" # relative motion wrt tool0 frame
        pose_msg.pose.position.x = p_tcp[0]
        pose_msg.pose.position.y = p_tcp[1]
        pose_msg.pose.position.z = p_tcp[2]
        pose_msg.pose.orientation.w = quat6[0]
        pose_msg.pose.orientation.x = quat6[1]
        pose_msg.pose.orientation.y = quat6[2]
        pose_msg.pose.orientation.z = quat6[3]
        
        if n_ik > 0:
            #best_norm = 1e30
            best_err = 1e30
            q_best = None
            for _ in range(n_ik):
                result, tmp = motion_client.solve_ik(pose=pose_msg)
                if tmp is None:
                    continue
                tmp=np.asarray(tmp,dtype=float)
                #norm = np.linalg.norm(tmp - q_old)
                err = np.sum(np.abs(tmp - q_old))

                # if norm < best_norm:
                #     best_norm = norm
                #     q_best = tmp
                if err < best_err:
                    best_err = err
                    q_best = tmp

            if q_best is None or best_err > np.pi/3: 
                raise Warning("Ocio batocio te funziona mia la IK")
                continue
            else:
                q6=q_best
                q_old = q_best.copy()
        else:
            result, q6 = motion_client.solve_ik(pose=pose_msg)

        q6=np.asarray(q6,dtype=float)
        path6.append(q6)
    path6 = np.asarray(path6, dtype=float)
    path = np.concatenate((path, path6))

    n = path.shape[0]
    d = path.shape[1]
    N = num_waypoints
    t = np.linspace(0, 1, n)
    T = np.linspace(0, 1, N)

    path_dense = np.zeros((N, d))
    for i in range(d):
        path_dense[:, i] = np.interp(T, t, path[:, i])
    path = path_dense
    path = np.array([np.concatenate((p + np.array([np.pi, 0, 0, 0, 0, 0]), np.zeros(2))) for p in path])

    return path

def main():
    parameters = {
                "pos_init_cont": [0.8021465039268282, 0.2847160024669491, 0.9564999991059303],
                "pos_cont_goal": [0.746, 0.961, 0.960],
                "pos_init_ee":  (0.12394027698787038, 0.2665910764094347, 1.1638763741892955, -0.5837257860699608, 0.5925713099055904, -0.388471134978874, 0.3965017359886389),
                "pos_grip_ee": (0.6509734785173125, 0.2847438888358813, 0.9764986855761588, -0.5000033337808922, 0.49998929956451965, -0.4999940109139501, 0.5000133554007884),
                "offset": (0, 0.15, -0.02),
                "dCoR": [0.0, 0.06, -0.004],
                "vol_init": 40.0, #2e-5, +-MAE
                "densità": 998.0,
                "viscosità": 0.001,
                "tens_sup": 0.072,
                "vol_target": 20.0, #0.75e-5,
                "err_target": 5e-6,
                "theta_f": 87, #+-15°
                "num_wp": 320,
            }
    theta_f=100
    num_waypoints=320
    dt=0.01
    path=plan_pouring(parameters, theta_f, num_waypoints)
    time = np.arange(0.0, len(path)*dt, dt)
    best_path = {
        "all": path,
        "time": time,
    }
    print("FATTO TOMBOLA DIOCAN")
    with open("/tmp/best_path.yaml", "w") as f:
                yaml.safe_dump({"best_path": to_builtin(best_path)}, f, sort_keys=False)
  

if __name__ == '__main__':
    main()