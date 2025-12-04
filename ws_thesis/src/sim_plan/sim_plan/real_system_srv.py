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
from sim_plan.modello_pour_liq import reward_pouring
from sim_plan.modello_sloshing import reward_sloshing

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

def remap_trajectory(trj: JointTrajectory, joint_name_map, dt, scale=1):
    """
    Riordina q e interpola traj per avere un punto ogni dt secondi nello stesso tempo totale
    """
    name_to_idx  = {n:i for i,n in enumerate(trj.joint_names)}
    order        = [name_to_idx[n] for n in joint_name_map]
    
    new=[]
    time=[]
    for p in trj.points:
        q = [p.positions[i] for i in order]
        time_from_start = p.time_from_start
        t = time_from_start.sec + time_from_start.nanosec * 1e-9
        t_scaled = t * scale

        new.append(q)
        time.append(t_scaled)
    
    # Funziona anche così ma è meno ottimizzato
    # new=[]
    # time=[]
    # if getattr(result, "val") == 1:
    #     for pt in trj.points:
    #         q=np.zeros(7)
    #         for i,q_trj in enumerate(pt.positions):
    #             for j,name in enumerate(joint_name_map):
    #                 if name == trj.joint_names[i]:
    #                     q[j]=q_trj
    #         t=p.time_from_start.sec+p.time_from_start.nanosec*1e-9
    #         time.append(t)
    #         new.append(q)

    t = np.asarray(time, dtype=float)
    Q = np.asarray(new, dtype=float)  # shape: (N, n_joints)

    # Interpolazione lineare per ogni giunto
    t_new = np.arange(0.0, t[-1] + dt, dt)
    q_new = np.zeros((len(t_new), Q.shape[1]))

    for j in range(Q.shape[1]):
        q_new[:, j] = np.interp(t_new, t, Q[:, j])

    return q_new

def q2moveit(q):
    q=to_numpy_cpu(q)
    q=q[:6]
    q = [float(x) for x in q]
    q[0]-=np.pi
    return q

def compute_reward_models_rs(parameters, theta_f, num_wp, path):
    
    print(f"Evaluating rewards")
    
    dt = 0.01
    transp_mot = path["transport"]
    N = len(transp_mot)
    time_mot = np.linspace(0.0, (N-1)*dt, N)
    
    trj = {
        "positions": transp_mot,
        "time": time_mot,
    }
    
    w_pour, w_sloshing, w_speed = 1, 0, 0

    reward_pour = w_pour * reward_pouring(num_waypoints=num_wp, theta_f=theta_f, vol_target=parameters["vol_target"],parameters=parameters)
    
    reward_slosh = w_sloshing #* reward_sloshing(trj, parameters["vol_init"])

    num_wp_opt=300
    err_speed=np.linalg.norm(num_wp-num_wp_opt)
    err_max_speed=70
    reward_speed = w_speed * max (0,1-err_speed/err_max_speed) # da cambiare con modello


    print(f"reward pouring: {reward_pour}")
    print(f"reward sloshing: {reward_slosh}")
    print(f"reward speed: {reward_speed}")

    reward = reward_pour + reward_slosh + reward_speed
    w_tot = w_pour + w_sloshing + w_speed
    reward/=w_tot

    print(f"final reward: {reward}")
    return reward

def is_success(score, threshold=0.5):
    return score > threshold

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
                "pos_cont_goal": (0.746, 0.961, 0.960),
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
        # Carica parametri planning (non sono quelli ottimali ma è corretto così):
        FILE_CURRENT_PLAN_PARAMS="/tmp/current_plan_params.yaml"
        if not os.path.exists(FILE_CURRENT_PLAN_PARAMS):
                self.get_logger().error("current plan params don't exist")
        else:
            with open(FILE_CURRENT_PLAN_PARAMS, "r") as f: # aperto dopo sim_plan --> un solo valore
                    data_plan = yaml.safe_load(f)
            theta_f = data_plan["current_theta"]
            num_wp = data_plan["current_num_wp"]

        if not os.path.exists("/tmp/threshold_old.yaml"):
            threshold=0.1
        else:
            with open("/tmp/threshold.yaml", "r") as f:
                threshold = yaml.safe_load(f)
                if threshold is None:
                    threshold=0.1
        threshold=0.7
        
        if not os.path.exists("/tmp/best_path.yaml"):
            self.get_logger().error("best path doesn't exist")
        else:
            with open("/tmp/best_path.yaml", "r") as f:
                path=yaml.safe_load(f)
        
        #################################################################################à
        # Simulazione

        score = compute_reward_models_rs(parameters, theta_f, num_wp, path)
        success = is_success(score,threshold)

        print(f"success: {success}")

        response.success=True if success==1 else False
        return response   
       
def main(args=None):
    rclpy.init(args=args)
    node = RealSystemService()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()