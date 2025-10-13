import time
import rclpy
from rclpy.node import Node
import py_trees
from py_trees.blackboard import Blackboard
import numpy as np
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from moveit.planning import MoveItPy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from control_msgs.action import GripperCommand
from std_msgs.msg import Float32
from builtin_interfaces.msg import Duration
from drims2_motion_server.motion_client import MotionClient
import yaml
import os
import paramiko



class Timeout(py_trees.decorators.Decorator):
    def __init__(self, child, seconds: float, name="Timeout"):
        super().__init__(name=name, child=child)
        self.seconds = seconds
        self.start_t = None
    def initialise(self):
        self.start_t = time.time()
    def update(self):
        child_status = self.decorated.tick_once()
        #status = self.decorated.status
        if time.time() - self.start_t > self.seconds:
            self.decorated.stop(py_trees.common.Status.INVALID)
            return py_trees.common.Status.FAILURE
        return child_status

class Retry(py_trees.decorators.Decorator):
    def __init__(self, child, max_attempts: int, name="Retry"):
        super().__init__(name=name, child=child)
        self.max_attempts = max_attempts
        self.attempts = 0
    def initialise(self):
        self.attempts = 0
    def update(self):
        child_status = self.decorated.tick_once()
        #status = self.decorated.status
        if child_status == py_trees.common.Status.FAILURE:
            self.attempts += 1
            self.node.get_logger.warn(f"Retry {self.attempts+1}")
            if self.attempts < self.max_attempts:
                self.decorated.stop(py_trees.common.Status.INVALID)
                #self.decorated.tick_once()
                return py_trees.common.Status.RUNNING
            else:
                return py_trees.common.Status.FAILURE
        return child_status

# ---------- Leaf base ----------
class RosLeaf(py_trees.behaviour.Behaviour):
    def __init__(self, name:str, node:Node):
        super().__init__(name)
        self.node = node
        self.bb = Blackboard()

# ---------- Movimento ----------
class PrintPose(RosLeaf):
    def __init__(self, node, target_frame="base_link", ee_frame="tip", name="PrintPose"):
        super().__init__(name, node)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.node, spin_thread=True)
        self.target_frame = target_frame
        self.ee_frame = ee_frame

    def initialise(self):
        pass

    def update(self):
        try:
            t = self.tf_buffer.lookup_transform(
                self.target_frame,        
                self.ee_frame,           
                rclpy.time.Time(),
                timeout=Duration(seconds=0.5)
            )
            p = t.transform.translation
            q = t.transform.rotation

            pose = PoseStamped()
            pose.header = t.header
            pose.header.frame_id = self.target_frame
            pose.pose.position.x = p.x; pose.pose.position.y = p.y; pose.pose.position.z = p.z
            pose.pose.orientation = q

            self.node.get_logger().info(
                f"EE in {self.target_frame}: p=[{p.x:.3f}, {p.y:.3f}, {p.z:.3f}] "
                f"q=[{q.x:.3f}, {q.y:.3f}, {q.z:.3f}, {q.w:.3f}]"
            )
            # stampa una volta → SUCCESS. Per stampa continua, ritorna RUNNING.
            return py_trees.common.Status.SUCCESS
        except Exception as e:
            self.feedback_message = f"TF lookup failed: {e}"
            return py_trees.common.Status.FAILURE

class MoveToPose(RosLeaf):
    def __init__(self, node, pose_list=None, pose_bb=None, name="MoveToPose"):
        super().__init__(name, node)
        self.pose_list = pose_list
        self.pose_bb = pose_bb
        self.motion_client = MotionClient()
        

    def initialise(self):
        self.node.get_logger().info("MoveToPose started")

    def update(self):
        # se pose_list è chiave del blackboard, leggi una sola volta
        if isinstance(self.pose_list, str):
            val = self.bb.get(self.pose_list)
            if val is not None:
                self.pose_list = val
            else:
                self.feedback_message = f"Pose '{self.pose_list}' non ancora disponibile"
                return py_trees.common.Status.RUNNING

        # controlla validità
        if self.pose_list is None or (len(self.pose_list) != 7 and len(self.pose_list) != 6):
            self.feedback_message = "Pose non valida"
            return py_trees.common.Status.FAILURE

        if len(self.pose_list) == 7:
        # crea messaggio pose
            pose_msg = PoseStamped()
            pose_msg.header.frame_id = "base_link"
            pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z = map(float,self.pose_list[:3])
            pose_msg.pose.orientation.x, pose_msg.pose.orientation.y, pose_msg.pose.orientation.z, pose_msg.pose.orientation.w = map(float,self.pose_list[3:])

            try:
                result = self.motion_client.move_to_pose(pose_msg, cartesian_motion=True) 
                if getattr(result, "val", 1) == 1:
                    if self.pose_bb is not None:
                        self.bb.set(self.pose_bb, self.pose_list)
                    return py_trees.common.Status.SUCCESS
                else:
                    self.feedback_message = "Movimento fallito"
                    return py_trees.common.Status.FAILURE
            except Exception as e:
                self.feedback_message = f"Errore move_to_pose: {e}"
                return py_trees.common.Status.FAILURE
        else:
            try:
                result = self.motion_client.move_to_joint(self.pose_list) 
                if getattr(result, "val", 1) == 1:
                    if self.pose_bb is not None:
                        self.bb.set(self.pose_bb, self.pose_list)
                    return py_trees.common.Status.SUCCESS
                else:
                    self.feedback_message = "Movimento fallito"
                    return py_trees.common.Status.FAILURE
            except Exception as e:
                self.feedback_message = f"Errore move_to_pose: {e}"
                return py_trees.common.Status.FAILURE

# ---------- Percezione ----------
class CallVisionService(RosLeaf):
    def __init__(self, node, estimate_volume: bool, 
                 out_centroid_key="pos_cont_goal",
                 out_pos_key="pos_init_cont",
                 out_vol_key="init_vol",
                 name="CallVisionService"):
        super().__init__(name, node)
        self.estimate_volume = estimate_volume
        self.out_centroid_key = out_centroid_key
        self.out_pos_key = out_pos_key
        self.out_vol_key = out_vol_key

        # Client ROS2
        from interfaces.srv import Perception
        self.client = self.node.create_client(Perception, 'estimate_perception')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info("Servizio estimate_perception non disponibile, retry...")

        self._future = None
        self._sent = False

    def initialise(self):
        self._future = None
        self._sent = False
        self.node.get_logger().info("Perception started")

    def update(self):
        from interfaces.srv import Perception

        # 1. Invia richiesta solo una volta
        if not self._sent:
            req = Perception.Request()
            req.estimate_volume = self.estimate_volume
            self._future = self.client.call_async(req)
            self._sent = True
            return py_trees.common.Status.RUNNING

        # 2. Attende completamento future
        if self._future is None or not self._future.done():
            return py_trees.common.Status.RUNNING

        # 3. Future completato → gestisci risultato
        try:
            resp = self._future.result()
            if resp is None or not resp.success:
                self.feedback_message = "Vision service fallito"
                msg = getattr(resp, "message", "no message")
                self.node.get_logger().warn(f"Vision service fallito: {msg}")
                return py_trees.common.Status.FAILURE

            # Salva risultati nel blackboard
            self.bb.set(self.out_centroid_key, list(resp.centroid))
            if self.estimate_volume:
                self.bb.set(self.out_pos_key, list(resp.centroid))
                self.bb.set(self.out_vol_key, resp.volume)

            self.node.get_logger().info(f"Vision completed: {resp.centroid}")
            return py_trees.common.Status.SUCCESS

        except Exception as e:
            self.feedback_message = str(e)
            self.node.get_logger().error(f"Errore VisionService: {e}")
            return py_trees.common.Status.FAILURE

# ---------- Logica/Utility ----------
class ComputeOffset(RosLeaf):
    def __init__(self, node, ee_pose_key, cont_pose_key, out_key="offset", name="ComputeOffset"):
        super().__init__(name, node)
        self.ee_pose_key = ee_pose_key
        self.cont_pose_key = cont_pose_key
        self.out_key = out_key
    def update(self):
        ee = self.bb.get(self.ee_pose_key)     # 7D (pos+quat)
        cont = self.bb.get(self.cont_pose_key) # 3D
        if ee is None or cont is None:
            return py_trees.common.Status.FAILURE
        offset = [ee[0]-cont[0], ee[1]-cont[1], ee[2]-cont[2]]
        self.bb.set(self.out_key, offset)
        return py_trees.common.Status.SUCCESS

    def _goal_response_cb(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.feedback_message = "Goal rifiutato al gripper"
            self._result_future = None
            return
        self._result_future = goal_handle.get_result_async()

class CloseGripper(RosLeaf):
    def __init__(self, node, name="CloseGripper"):
        super().__init__(name, node)
        self.client = ActionClient(
            self.node,
            GripperCommand,
            "/gripper_action_controller/gripper_cmd"
        )
        self._sent = False
        self._result_future = None

    def initialise(self):
        self._sent = False
        self._result_future = None
        self.node.get_logger().info(f"Closing gripper")

    def _goal_response_cb(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.feedback_message = "Goal rifiutato dal gripper"
            self._result_future = None
            return
        self._result_future = goal_handle.get_result_async()

    def update(self):
        if not self._sent:
            if not self.client.wait_for_server(timeout_sec=1.0):
                self.feedback_message = "Server gripper non disponibile"
                return py_trees.common.Status.FAILURE

            goal = GripperCommand.Goal()
            goal.command.position = 0.0    # chiuso
            goal.command.max_effort = 0.0  # come da comando corretto

            self._goal_future = self.client.send_goal_async(goal)
            self._goal_future.add_done_callback(self._goal_response_cb)
            self._sent = True
            return py_trees.common.Status.RUNNING

        if self._result_future and self._result_future.done():
            result = self._result_future.result().result
            if getattr(result, "reached_goal", True):
                return py_trees.common.Status.SUCCESS
            else:
                self.feedback_message = "Gripper non ha raggiunto il goal"
                return py_trees.common.Status.FAILURE

        return py_trees.common.Status.RUNNING
    
class OpenGripper(RosLeaf):
    def __init__(self, node, name="OpenGripper"):
        super().__init__(name, node)
        self.client = ActionClient(
            self.node,
            GripperCommand,
            "/gripper_action_controller/gripper_cmd"
        )
        self._sent = False
        self._result_future = None

    def initialise(self):
        self._sent = False
        self._result_future = None
        self.node.get_logger().info(f"Opening gripper")

    def _goal_response_cb(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.feedback_message = "Goal rifiutato dal gripper"
            self._result_future = None
            return
        self._result_future = goal_handle.get_result_async()

    def update(self):
        if not self._sent:
            if not self.client.wait_for_server(timeout_sec=1.0):
                self.feedback_message = "Server gripper non disponibile"
                return py_trees.common.Status.FAILURE

            goal = GripperCommand.Goal()
            goal.command.position = 0.025    # aperto
            goal.command.max_effort = 0.1  # come da comando corretto

            self._goal_future = self.client.send_goal_async(goal)
            self._goal_future.add_done_callback(self._goal_response_cb)
            self._sent = True
            return py_trees.common.Status.RUNNING

        if self._result_future and self._result_future.done():
            result = self._result_future.result().result
            if getattr(result, "reached_goal", True):
                return py_trees.common.Status.SUCCESS
            else:
                self.feedback_message = "Gripper non ha raggiunto il goal"
                return py_trees.common.Status.FAILURE

        return py_trees.common.Status.RUNNING
    
class SetPlanParams(RosLeaf):
    def __init__(self, node, theta_f, num_wp, target_vol, name="SetPlanParams"):
        super().__init__(name, node)
        self.theta_f = theta_f; self.num_wp = num_wp; self.target_vol = target_vol
    def update(self):
        self.bb.set("theta_f", self.theta_f)
        self.bb.set("num_wp", self.num_wp)
        self.bb.set("target_vol", self.target_vol)

        # Debug purposes:
        self.bb.set("pos_init_cont", [0.0, 0.0, 0.0]),
        self.bb.set("pos_init_ee",[0.0]*7),
        self.bb.set("pos_cont_goal", [0.0, 0.0, 0.0]),
        self.bb.set("offset", [0.0, 0.0, 0.0]),
        self.bb.set("init_vol", 0.0),
        self.bb.set("densità", 998.0),
        self.bb.set("viscosità", 0.001),
        self.bb.set( "tens_sup", 0.072),
        self.bb.set("err_target", 5e-6),
                
        try:
            init_parameters = {
                "pos_init_cont": list(self.bb.get("pos_init_cont") or [0.0, 0.0, 0.0]),
                "pos_init_ee": list(self.bb.get("pos_init_ee") or [0.0]*7),
                "pos_cont_goal": list(self.bb.get("pos_cont_goal") or [0.0, 0.0, 0.0]),
                "offset": list(self.bb.get("offset") or [0.0, 0.0, 0.0]),
                "vol_init": float(self.bb.get("init_vol") or 0.0),
                "densità": 998.0, # not used in serv but same val
                "viscosità": 0.001, # not used in serv but same val
                "tens_sup": 0.072, # not used in serv but same val
                "vol_target": float(self.bb.get("target_vol") or 0.0), #0.75e-5,
                "err_target": 5e-6, # not used in serv but same val
                "theta_f": float(self.bb.get("theta_f") or 90.0),
                "num_wp": int(self.bb.get("num_wp") or 1000),
            }
            self.bb.set("init_parameters", init_parameters)
            with open("/tmp/init_parameters.yaml", "w") as f:
                yaml.safe_dump({"parameters": init_parameters}, f, sort_keys=False)
            self.node.get_logger().info("File initial parameters created")
            return py_trees.common.Status.SUCCESS
        except Exception as e:
            self.node.get_logger().error(f"File creation failed: {str(e)}")
            return py_trees.common.Status.FAILURE
        
class SendYamlToVM(RosLeaf):
    def __init__(self, node, name="SendYamlToVM"):
        super().__init__(name, node)

    def update(self):
        local_path = "/tmp/init_parameters.yaml"
        remote_path = "/tmp/init_parameters.yaml"

        host = "100.93.166.22"
        user = "barutta"
        key_file = "/home/edo/.ssh/id_barutta"

        # Controllo chiave
        if not os.path.exists(key_file):
            self.logger.error(f"Chiave SSH non trovata: {key_file}")
            return py_trees.common.Status.FAILURE

        try:
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(host, username=user, key_filename=key_file)

            sftp = client.open_sftp()
            sftp.put(local_path, remote_path)
            sftp.close()
            client.close()

            self.node.get_logger().info("File inviato con successo")
            return py_trees.common.Status.SUCCESS

        except Exception as e:
            self.node.get_logger().error(f"File transfer failed: {str(e)}")
            return py_trees.common.Status.FAILURE

class WaitForBestPath(RosLeaf):
    def __init__(self, node, file_path="/tmp/best_path.yaml", check_interval=5.0, name="WaitForBestPath"):
        super().__init__(name, node)
        self.file_path = file_path
        self.check_interval = check_interval
        self._last_check = None
        

    def initialise(self):
        self._last_check = time.time()
        self.node.get_logger().info(f"Waiting for path file")

    def update(self):
        now = time.time()
        # Controlla solo ogni check_interval
        if self._last_check is None or (now - self._last_check) >= self.check_interval:
            self._last_check = now
            if os.path.exists(self.file_path):
                try:
                    with open(self.file_path, "r") as f:
                        data = yaml.safe_load(f)
                    # estrazione campi
                    time_arr = data.get("time")
                    path = data.get("best_path")
                    if time_arr is None or path is None:
                        self.node.get_logger().info("File trovato ma campi mancanti")
                        return py_trees.common.Status.FAILURE
                    # scrive su blackboard
                    self.bb.set("time", list(time_arr))
                    self.bb.set("best_path", [list(p) for p in path])
                    return py_trees.common.Status.SUCCESS
                except Exception as e:
                    self.node.get_logger().info(f"Errore parsing yaml: {str(e)}")
                    return py_trees.common.Status.FAILURE

        return py_trees.common.Status.RUNNING

# To be tested
class ExecutePathPublisher(RosLeaf):
    def __init__(self, node, name="ExecutePathPublisher", tol=0.01, grace_t=1.0):
        super().__init__(name, node)
        self.pub = self.node.create_publisher(
            JointTrajectory,
            "/scaled_joint_trajectory_controller/joint_trajectory",
            10
        )
        # Joint traj message (pochi punti --> interpolaz interna control)
        # /stream ai giunti diretto
        self.sub = self.node.create_subscription(
            JointState,
            "/joint_states",
            self._joint_state_cb,
            10,
        )
        self._sent = False
        self._traj_duration = 0.0
        self._last_joint_state = None
        self.tol = tol  # tolleranza in rad
        self.grace_t = grace_t

    def _joint_state_cb(self, msg):
        self._last_joint_state = msg

    def initialise(self):
        self._sent = False
        self._traj_duration = 0.0
        self.bb.set("traj_start_time", self.node.get_clock().now().nanoseconds / 1e9)

    def update(self):
        time_arr = self.bb.get("time")
        path = self.bb.get("best_path")

        if time_arr is None or path is None:
            self.feedback_message = "Traiettoria non disponibile"
            return py_trees.common.Status.FAILURE

        if not self._sent:
            traj = JointTrajectory()
            traj.joint_names = [
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint"
            ]

            for t, q in zip(time_arr, path):
                pt = JointTrajectoryPoint()
                pt.positions = q[:6]
                pt.time_from_start = Duration(sec=int(t), nanosec=int((t % 1.0) * 1e9))
                #pt.time_from_start = rclpy.duration.Duration(seconds=float(t)).to_msg()
                traj.points.append(pt)

            self.pub.publish(traj)
            self._sent = True
            self._traj_duration = float(time_arr[-1])
            self.bb.set("goal_joints", path[-1][:6])  # salva goal finale
            return py_trees.common.Status.RUNNING

        elapsed = (self.node.get_clock().now().nanoseconds / 1e9) - self.bb.get("traj_start_time", 0.0)
        if elapsed >= self._traj_duration + self.grace_t: # SÌ MA DAMMI IL TEMPOOO
            # Check anche sulla posizione attuale dei giunti
            if self._check_joints_close():
                return py_trees.common.Status.SUCCESS
            else:
                self.feedback_message = "Joint finali fuori tolleranza"
                return py_trees.common.Status.FAILURE

        return py_trees.common.Status.RUNNING

    def _check_joints_close(self):
        """Verifica che i giunti attuali siano vicini al goal"""
        goal = self.bb.get("goal_joints")
        if self._last_joint_state is None or goal is None:
            return False

        name_to_idx = {n: i for i, n in enumerate(self._last_joint_state.name)}
        current_pos = []
        for j in [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint"
        ]:
            if j not in name_to_idx:
                return False
            current_pos.append(self._last_joint_state.position[name_to_idx[j]])

        current_pos = np.array(current_pos)
        goal = np.array(goal)
        err = np.linalg.norm(goal - current_pos, ord=np.inf)
        return err < self.tol


#==============================================================================================================
# COSTRUZIONE ALBERO E AVVIO:

def create_tree(node: Node):
    joint_v1= [0.7227166962826039,-1.746930173633286, -2.2322865329350017, -2.046302405379515, 0.738723064687373, 2.948454781562834]
    joint_v2 = [-3.129748565398613, -2.1683139224910026, -2.134126744414425, -3.519583411401461, -2.9772426124069256, -1.5698350868947821]

    open=OpenGripper(node)
    move_t1 = MoveToPose(node, pose_list=joint_v1,pose_bb=None)
    vision_1 = CallVisionService(node, estimate_volume=False, out_centroid_key="pos_cont_goal")

    move_t2 = MoveToPose(node, pose_list=joint_v2,pose_bb="pos_init_ee")
    vision_2 = CallVisionService(node, estimate_volume=True, out_centroid_key="pos_init_cont", out_vol_key="init_vol")

    pose_c=[0.331, 0.571, 0.066, -0.018, 0.721, 0.692, -0.027] # x,y,z,x,y,z,w
    move_c = MoveToPose(node, pose_list=pose_c, pose_bb="pos_appr_ee")
    #move_c = MoveToPose(node, pose_list="pos_init_cont", pose_bb="pos_appr_ee")

    off = ComputeOffset(node, "pos_appr_ee", "pos_init_cont")
    grip = CloseGripper(node)
    par_util = py_trees.composites.Parallel(
        "UtilitiesParallel",
        policy=py_trees.common.ParallelPolicy.SuccessOnAll()
    )
    par_util.add_children([off, grip])
    params  = SetPlanParams(node, theta_f=90, num_wp=1000, target_vol=20.0)

    send = SendYamlToVM(node)
    wait_path = WaitForBestPath(node)
    # execp   = Retry(Timeout(ExecutePathPublisher(node), 60.0), 1) # ExecutePathPublisher o ExecutePathAction
   
    pose=PrintPose(node)
    seq = py_trees.composites.Sequence("FullCycle",memory=True)
    seq.add_children([
        #pose,
        open,
        move_t1, vision_1,
        move_t2, vision_2, 
        move_c, par_util, params,
        send,
        wait_path,
        #execp,
        ])
    
  
    return seq  

def main():
    rclpy.init()
    node = Node("bt_orchestrator")
    tree = py_trees.trees.BehaviourTree(create_tree(node))
    # Tick ~10 Hz (a piacere)
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
            tree.tick()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
