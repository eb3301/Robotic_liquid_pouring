import time
import rclpy
from rclpy.node import Node
import py_trees
from py_trees.blackboard import Blackboard
import numpy as np
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from moveit.planning import MoveItPy
from interfaces.srv import Simplan, UpdateBelief
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from control_msgs.action import GripperCommand
from std_msgs.msg import Float32
from builtin_interfaces.msg import Duration
#from drims2_motion_server.motion_client import MotionClient
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
class MoveToPose(RosLeaf):
    def __init__(self, node, pose_list=None, pose_from_bb=None, name="MoveToPose"):
        """
        :param pose_list: lista [x, y, z, qx, qy, qz, qw]
        :param pose_from_bb: chiave sul blackboard da cui leggere la pose
        """
        super().__init__(name, node)
        self.pose_list = pose_list
        self.pose_from_bb = pose_from_bb
        self._sent = False
        self._executing = False
        self._traj = None
        self._last_joint_state = None

        # Subscriber a /joint_states
        self.sub = self.node.create_subscription(
            JointState,
            "/joint_states",
            self._joint_state_cb,
            10,
        )

        # Inizializza MoveIt Client
        #motion_client = MotionClient()
        self.moveit = MoveItPy(node_name="moveit")
        self.planner = self.moveit.get_planning_component("ur_manipulator") 
        self.node.get_logger().info("Move to pose started")

    def _joint_state_cb(self, msg: JointState):
        self._last_joint_state = msg

    def initialise(self):
        self._sent = False
        self._executing = False
        self._traj = None
        self.bb.set("final_traj_joints", None)
        self.bb.set("final_traj_names", None)

    def update(self):
        if not self._sent:
            # Scegli la pose
            if self.pose_from_bb:
                pose_list = self.bb.get(self.pose_from_bb)
            else:
                pose_list = self.pose_list

            if pose_list is None or len(pose_list) != 7:
                self.feedback_message = "Pose non valida"
                return py_trees.common.Status.FAILURE

            # Converti in geometry_msgs/Pose
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = pose_list[:3]
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = pose_list[3:]

            # Pianificazione
            self.planner.set_start_state_to_current_state()
            self.planner.set_pose_target(pose)
            plan_result = self.planner.plan()

            if not plan_result or plan_result.joint_trajectory is None:
                self.feedback_message = "Pianificazione fallita"
                return py_trees.common.Status.FAILURE

            # Esecuzione
            self._traj = plan_result.joint_trajectory
            if self._traj.points:
                self.bb.set("final_traj_joints", list(self._traj.points[-1].positions))
                self.bb.set("final_traj_names", list(self._traj.joint_names))
            self.planner.execute(self._traj)
            self._sent = True
            self._executing = True
            return py_trees.common.Status.RUNNING

        # Controllo stato
        if self._executing:
            if self._check_motion_done():
                self._executing = False
                # Salva la pose corrente dell’end-effector se serve
                if self.pose_list is not None:
                    ee_pose = self._get_ee_pose()
                    self.bb.set("pos_init_ee", ee_pose)
                return py_trees.common.Status.SUCCESS
            else:
                return py_trees.common.Status.RUNNING

        return py_trees.common.Status.SUCCESS

    def _check_motion_done(self, tol=0.05):
        if self._traj is None or self._last_joint_state is None:
            return False

        goal_pos = np.array(self._traj.points[-1].positions)
        current_pos = []

        name_to_idx = {n: i for i, n in enumerate(self._last_joint_state.name)}
        for j in self._traj.joint_names:
            if j not in name_to_idx:
                return False
            idx = name_to_idx[j]
            current_pos.append(self._last_joint_state.position[idx])

        current_pos = np.array(current_pos)
        err = np.linalg.norm(goal_pos - current_pos, ord=np.inf)
        return err < tol

    def _get_ee_pose(self):
        pose_msg = self.planner.get_current_pose()
        return [
            pose_msg.position.x,
            pose_msg.position.y,
            pose_msg.position.z,
            pose_msg.orientation.x,
            pose_msg.orientation.y,
            pose_msg.orientation.z,
            pose_msg.orientation.w,
        ]


# Cambia con lib prof
from rclpy.action import ActionClient
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, PositionConstraint, OrientationConstraint
from geometry_msgs.msg import PoseStamped
from shape_msgs.msg import SolidPrimitive
import numpy as np
import py_trees
from sensor_msgs.msg import JointState

# To be tested:
class MoveToPose1(RosLeaf):
    def __init__(self, node, pose_list=None, pose_from_bb=None, name="MoveToPose1"):
        """
        :param pose_list: lista [x, y, z, qx, qy, qz, qw]
        :param pose_from_bb: chiave sul blackboard da cui leggere la pose
        """
        super().__init__(name, node)
        self.pose_list = pose_list
        self.pose_from_bb = pose_from_bb
        self.client = ActionClient(node, MoveGroup, '/move_action')

        self._sent = False
        self._result_future = None
        self._last_joint_state = None

        # subscriber a joint_states per controllare arrivo
        self.sub = self.node.create_subscription(
            JointState,
            "/joint_states",
            self._joint_state_cb,
            10,
        )

    def _joint_state_cb(self, msg: JointState):
        self._last_joint_state = msg

    def initialise(self):
        self._sent = False
        self._result_future = None
        self.bb.set("final_traj_joints", None)
        self.bb.set("final_traj_names", None)

    def update(self):
        # Se goal già inviato → controlla risultato
        if self._sent:
            if self._result_future and self._result_future.done():
                result = self._result_future.result().result
                if result.error_code == 1:  # SUCCESS
                    # salva posizione EE (se disponibile via tf/joint_states)
                    ee_pose = self._get_ee_pose_from_state()
                    if ee_pose:
                        self.bb.set("pos_init_ee", ee_pose)
                    return py_trees.common.Status.SUCCESS
                else:
                    self.feedback_message = f"Esecuzione fallita: code={result.error_code}"
                    return py_trees.common.Status.FAILURE

            if self._result_future is None:
                self.feedback_message = "Goal non accettato"
                return py_trees.common.Status.FAILURE

            return py_trees.common.Status.RUNNING

        # Primo invio goal
        pose_list = self.bb.get(self.pose_from_bb) if self.pose_from_bb else self.pose_list
        if pose_list is None or len(pose_list) != 7:
            self.feedback_message = "Pose non valida"
            return py_trees.common.Status.FAILURE

        pose = PoseStamped()
        pose.header.frame_id = "base_link"
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = pose_list[:3]
        pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w = pose_list[3:]

        # Costruisci constraints
        pos_c = PositionConstraint()
        pos_c.header.frame_id = "base_link"
        pos_c.link_name = "tool0"
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [0.001, 0.001, 0.001]  # tolleranza 1mm
        pos_c.constraint_region.primitives.append(box)
        pos_c.constraint_region.primitive_poses.append(pose.pose)

        ori_c = OrientationConstraint()
        ori_c.header.frame_id = "base_link"
        ori_c.link_name = "tool0"
        ori_c.orientation = pose.pose.orientation
        ori_c.absolute_x_axis_tolerance = 0.01
        ori_c.absolute_y_axis_tolerance = 0.01
        ori_c.absolute_z_axis_tolerance = 0.01
        ori_c.weight = 1.0

        constraints = Constraints()
        constraints.position_constraints.append(pos_c)
        constraints.orientation_constraints.append(ori_c)

        goal = MoveGroup.Goal()
        goal.request.group_name = "ur_manipulator"
        goal.request.goal_constraints.append(constraints)

        if not self.client.wait_for_server(timeout_sec=1.0):
            self.feedback_message = "MoveGroup server non disponibile"
            return py_trees.common.Status.FAILURE

        send_future = self.client.send_goal_async(goal)
        send_future.add_done_callback(self._goal_response_cb)

        self._sent = True
        return py_trees.common.Status.RUNNING

    def _goal_response_cb(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.feedback_message = "Goal rifiutato"
            self._result_future = None
            return
        self._result_future = goal_handle.get_result_async()

    def _get_ee_pose_from_state(self):
        """
        Ritorna la posa stimata di tool0 leggendo tf o joint_states.
        Qui come placeholder ritorna None, puoi sostituire con tf2 per ottenere la vera pose.
        """
        return None

class WaitRobotArrived(RosLeaf):
    def __init__(self, node, target_key="final_traj_joints", timeout_s=20, tol=0.01, name="WaitRobotArrived"):
        super().__init__(name, node)
        self.target_key = target_key    
        self.timeout_s = timeout_s
        self.tol = tol
        self._last_joint_state = None

        self.sub = self.node.create_subscription(
            JointState,
            "/joint_states",
            self._joint_state_cb,
            10,
        )

    def _joint_state_cb(self, msg: JointState):
        self._last_joint_state = msg

    def initialise(self):
        self.t0 = self.node.get_clock().now()

    def update(self):
        if self._last_joint_state is None:
            return py_trees.common.Status.RUNNING

        target = self.bb.get(self.target_key)
        names  = self.bb.get("final_traj_names")
        if target is None or names is None:
            self.feedback_message = "Target non disponibile"
            return py_trees.common.Status.FAILURE

        name_to_idx = {n: i for i, n in enumerate(self._last_joint_state.name)}
        current_pos = []
        for jn in names:
            if jn not in name_to_idx:
                self.feedback_message = f"Giunto {jn} non presente in /joint_states"
                return py_trees.common.Status.FAILURE
            current_pos.append(self._last_joint_state.position[name_to_idx[jn]])

        if len(current_pos) != len(target):
            self.feedback_message = "Mismatch numero giunti"
            return py_trees.common.Status.FAILURE

        current_pos = np.array(current_pos)
        goal_pos = np.array(target)
        err = np.linalg.norm(goal_pos - current_pos, ord=np.inf)

        if err < self.tol:
            return py_trees.common.Status.SUCCESS

        elapsed = (self.node.get_clock().now() - self.t0).nanoseconds / 1e9
        if elapsed > self.timeout_s:
            self.feedback_message = "Timeout"
            return py_trees.common.Status.FAILURE

        return py_trees.common.Status.RUNNING

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

        # Crea il client
        from interfaces.srv import Perception
        self.client = self.node.create_client(Perception, 'estimate_perception')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info("Servizio estimate_perception non disponibile, retry...")
        

    def update(self):
        #self.node.get_logger().info("vision tick")
        try:
            from interfaces.srv import Perception
            req = Perception.Request()
            req.estimate_volume = self.estimate_volume

            future = self.client.call_async(req)
            rclpy.spin_until_future_complete(self.node, future)
            resp = future.result()

            if resp is None or not resp.success:
                self.feedback_message = "Vision service fallito"
                self.node.get_logger().warn(f"{resp.message}")
                return py_trees.common.Status.FAILURE

            # Salva nel blackboard
            self.bb.set(self.out_centroid_key, list(resp.centroid))
            self.node.get_logger().info(self.out_centroid_key)
            if self.estimate_volume:
                self.bb.set(self.out_pos_key, list(resp.centroid))  # qui ipotizziamo centroid ≈ pos_init_cont
                self.bb.set(self.out_vol_key, resp.volume)

            self.node.get_logger().info(f"Risposta ricevuta {resp.centroid}")
            return py_trees.common.Status.SUCCESS

        except Exception as e:
            self.feedback_message = str(e)
            self.node.get_logger().info("Failed request")
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

# Decidere quale close gripper usare in base a come è configurato il nodo +test
class CloseGripper(RosLeaf):
    def __init__(self, node, name="CloseGripper"):
        super().__init__(name, node)
        self.pub = self.node.create_publisher(
            JointTrajectory,
            "/robotiq_hande_controller/joint_trajectory",  # controlla nome esatto
            10
        )
        self._sent = False

    def initialise(self):
        self._sent = False

    def update(self):
        if not self._sent:
            traj = JointTrajectory()
            traj.joint_names = ["hande_left_finger_joint", "hande_right_finger_joint"]

            pt = JointTrajectoryPoint()
            pt.positions = [0.0, 0.0]   # chiuso; per aprire ~0.025
            pt.time_from_start = Duration(sec=1)

            traj.points.append(pt)
            self.pub.publish(traj)

            self._sent = True
            return py_trees.common.Status.RUNNING

        return py_trees.common.Status.SUCCESS

class CloseGripper1(RosLeaf):
    def __init__(self, node, name="CloseGripper1"):
        super().__init__(name, node)
        self.client = ActionClient(
            self.node,
            GripperCommand,
            "/robotiq_hande_controller/gripper_cmd"
        )
        self._sent = False
        self._result_future = None

    def initialise(self):
        self._sent = False
        self._result_future = None

    def update(self):
        if not self._sent:
            if not self.client.wait_for_server(timeout_sec=1.0):
                self.feedback_message = "Gripper server non disponibile"
                return py_trees.common.Status.FAILURE

            goal = GripperCommand.Goal()
            goal.command.position = 0.0   # chiuso
            goal.command.max_effort = 40.0

            self._goal_future = self.client.send_goal_async(goal)
            self._goal_future.add_done_callback(self._goal_response_cb)
            self._sent = True
            return py_trees.common.Status.RUNNING

        if self._result_future and self._result_future.done():
            result = self._result_future.result().result
            if result.reached_goal:
                return py_trees.common.Status.SUCCESS
            else:
                self.feedback_message = "Gripper non ha raggiunto il goal"
                return py_trees.common.Status.FAILURE

        return py_trees.common.Status.RUNNING

    def _goal_response_cb(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.feedback_message = "Goal rifiutato al gripper"
            self._result_future = None
            return
        self._result_future = goal_handle.get_result_async()

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
                "densità": 998.0,
                "viscosità": 0.001,
                "tens_sup": 0.072,
                "vol_target": float(self.bb.get("target_vol") or 0.0), #0.75e-5,
                "err_target": 5e-6,
                "theta_f": float(self.bb.get("theta_f") or 90.0),
                "num_wp": int(self.bb.get("num_wp") or 1000),
            }
            self.bb.set("init_parameters", init_parameters)
            with open("/tmp/init_parameters.yaml", "w") as f:
                yaml.safe_dump({"parameters": init_parameters}, f, sort_keys=False)
            self.node.get_logger().info("File initial parameters creates")
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

        host = "100.110.226.44"
        user = "edo"
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

class ExecutePathAction(RosLeaf):
    def __init__(self, node, name="ExecutePathAction", grace_t=1.0):
        super().__init__(name, node)
        self.client = ActionClient(
            self.node,
            FollowJointTrajectory,
            "/scaled_joint_trajectory_controller/follow_joint_trajectory"
        )
        self._goal_future = None
        self._result_future = None
        self._sent = False
        self.grace_t = grace_t

    def initialise(self):
        self._goal_future = None
        self._result_future = None
        self._sent = False

    def update(self):
        time_arr = self.bb.get("time")
        path = self.bb.get("best_path")

        if time_arr is None or path is None:
            self.feedback_message = "Traiettoria non disponibile"
            return py_trees.common.Status.FAILURE

        if not self._sent:
            if not self.client.wait_for_server(timeout_sec=1.0):
                self.feedback_message = "Action server non disponibile"
                return py_trees.common.Status.FAILURE

            goal_msg = FollowJointTrajectory.Goal()
            goal_msg.trajectory.joint_names = [
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
                pt.time_from_start = Duration(
                    sec=int(t),
                    nanosec=int((t % 1.0) * 1e9)
                )
                goal_msg.trajectory.points.append(pt)

            self._goal_future = self.client.send_goal_async(goal_msg)
            self._goal_future.add_done_callback(self._goal_response_callback)
            self._sent = True
            self.bb.set("traj_duration", float(time_arr[-1]))
            self.bb.set("traj_start_time", self.node.get_clock().now().nanoseconds / 1e9)
            return py_trees.common.Status.RUNNING

        if self._result_future is not None:
            if self._result_future.done():
                result = self._result_future.result().result
                if result.error_code == 0:  # SUCCESSFUL
                    return py_trees.common.Status.SUCCESS
                else:
                    self.feedback_message = f"Esecuzione fallita: {result.error_string}"
                    return py_trees.common.Status.FAILURE
            else:
                # opzionale: timeout rispetto a traj_duration
                elapsed = (self.node.get_clock().now().nanoseconds / 1e9) - self.bb.get("traj_start_time", 0.0)
                duration = self.bb.get("traj_duration", 0.0)
                if elapsed >= duration + self.grace_t:
                    self.feedback_message = "Timeout esecuzione path"
                    return py_trees.common.Status.FAILURE
                return py_trees.common.Status.RUNNING

        return py_trees.common.Status.RUNNING

    def _goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.node.get_logger().error("Goal rifiutato dal controller")
            self._result_future = None
            # forza fallimento
            self.feedback_message = "Goal rifiutato"
            return
        self._result_future = goal_handle.get_result_async()


#==============================================================================================================
# COSTRUZIONE ALBERO E AVVIO:

def create_tree(node: Node):
    # definizione target:
    poses = {
        "target_1": [0.4, 0.2, 0.3, 0, 0, 0, 1],
        "target_2": [0.5, -0.2, 0.35, 0, 0, 0, 1],
    }

    move_t1 = MoveToPose(node, pose_list=poses["target_1"])
    # wait_t1 = Timeout(WaitRobotArrived(node, target_key="final_traj_joints", timeout_s=20), 25.0)
    # vision_1 = Retry(Timeout(CallVisionService(node, estimate_volume=False, out_centroid_key="pos_cont_goal"), 15.0), 2)

    # move_t2 = Retry(Timeout(MoveToPose(node, pose_list=poses["target_2"]), 40.0), 2)
    # wait_t2 = Timeout(WaitRobotArrived(node, target_key="final_traj_joints", timeout_s=20), 25.0)
    # vision_2 = Retry(Timeout(CallVisionService(node, estimate_volume=True, out_centroid_key="pos_init_cont", out_vol_key="init_vol"), 20.0), 2)

    # move_c  = Retry(Timeout(MoveToPose(node, pose_from_bb="pos_init_cont"), 40.0), 2)
    # wait_c = Timeout(WaitRobotArrived(node, target_key="final_traj_joints", timeout_s=20), 25.0)

    # off     = ComputeOffset(node, "pos_init_ee", "pos_init_cont")
    # grip    = Retry(Timeout(CloseGripper(node), 5.0), 2) # CloseGripper o CloseGripper1
    # par_util = py_trees.composites.Parallel(
    #     "UtilitiesParallel",
    #     policy=py_trees.common.ParallelPolicy.SuccessOnAll()
    # )
    # par_util.add_children([off, grip])
    params  = SetPlanParams(node, theta_f=0.6, num_wp=50, target_vol=100.0)

    send = SendYamlToVM(node)
    wait_path = WaitForBestPath(node)
    # execp   = Retry(Timeout(ExecutePathPublisher(node), 60.0), 1) # ExecutePathPublisher o ExecutePathAction
   
    # seq = py_trees.composites.Sequence("FullCycle",memory=False)
    # seq.add_children([
    #     move_t1, wait_t1, vision_1,
    #     move_t2, wait_t2, vision_2, 
    #     move_c, wait_c, 
    #     par_util, params,
    #     send,
    #     wait_path,
    #     execp,
    #     ])
    
   
    test = py_trees.composites.Sequence("FullCycle", memory=True)
    test.add_children([
            move_t1
        ])
    return test

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
