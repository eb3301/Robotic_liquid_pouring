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

        # Inizializza MoveIt
        self.moveit = MoveItPy(node=self.node)
        self.planner = self.moveit.get_planning_component("ur_manipulator")  # cambia con il tuo planning group

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
        try:
            from interfaces.srv import Perception
            req = Perception.Request()
            req.estimate_volume = self.estimate_volume

            future = self.client.call_async(req)
            rclpy.spin_until_future_complete(self.node, future)
            resp = future.result()
            if resp is None or not resp.success:
                self.feedback_message = "Vision service fallito"
                return py_trees.common.Status.FAILURE

            # Salva nel blackboard
            self.bb.set(self.out_centroid_key, list(resp.centroid))

            if self.estimate_volume:
                self.bb.set(self.out_pos_key, list(resp.centroid))  # qui ipotizziamo centroid ≈ pos_init_cont
                self.bb.set(self.out_vol_key, resp.volume)

            return py_trees.common.Status.SUCCESS

        except Exception as e:
            self.feedback_message = str(e)
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

# Decidere quale close gripper usare in base a come è configurato il nodo
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
        return py_trees.common.Status.SUCCESS

class CallPlannerSrv(RosLeaf):
    def __init__(self, node, name="CallPlannerSrv"):
        super().__init__(name, node)
        # Crea un client per il servizio definito in path_planner_service
        self.client = self.node.create_client(Simplan, 'plan_path')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info('Servizio plan_path non disponibile, retry...')

    def update(self):
        # Prepara la request prendendo i dati dal blackboard
        req = Simplan.Request()
        req.pos_init_cont = self.bb.get("pos_init_cont") or [0.0, 0.0, 0.0]
        req.pos_init_ee   = self.bb.get("pos_init_ee")   or [0.0]*7
        req.pos_cont_goal = self.bb.get("pos_cont_goal") or [0.0, 0.0, 0.0]
        req.offset        = self.bb.get("offset")        or [0.0, 0.0, 0.0]
        req.theta_f       = float(self.bb.get("theta_f") or 0.0)
        req.num_wp        = int(self.bb.get("num_wp") or 0)
        req.init_vol      = float(self.bb.get("init_vol") or 0.0)
        req.target_vol    = float(self.bb.get("target_vol") or 0.0)

        # Chiamata asincrona → blocchiamo fino a risultato
        future = self.client.call_async(req)
        rclpy.spin_until_future_complete(self.node, future)
        if future.result() is None:
            self.feedback_message = "Chiamata al planner fallita"
            return py_trees.common.Status.FAILURE

        resp = future.result()
        # Salva nel blackboard
        self.bb.set("time", list(resp.time))
        self.bb.set("best_path", [list(p) for p in resp.best_path])

        return py_trees.common.Status.SUCCESS

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

# ros2 topic pub /reward std_msgs/msg/Float32 "{data: reward value}"
class WaitForReward(RosLeaf):
    def __init__(self, node, topic="/reward", timeout=None, name="WaitForReward"):
        super().__init__(name, node)
        self.topic = topic
        self.timeout = timeout
        self._sub = None
        self.got = False
        self.t0 = None
        self._sub = self.node.create_subscription(
            Float32,     
            self.topic,
            self._callback,
            10
        )

    def initialise(self):
        self.got = False
        self.t0 = self.node.get_clock().now()

    def _callback(self, msg):
        # Quando arriva un messaggio → aggiorna blackboard
        self.bb.set("reward", msg.data)
        self.got = True

    def update(self):
        if self.got:
            return py_trees.common.Status.SUCCESS

        if self.timeout is not None:
            elapsed = (self.node.get_clock().now() - self.t0).nanoseconds / 1e9
            if elapsed > self.timeout:
                self.feedback_message = "Timeout scaduto"
                return py_trees.common.Status.FAILURE

        return py_trees.common.Status.RUNNING

class UpdateSimParamsFromReward(RosLeaf):
    def __init__(self, node, name="UpdateSimParamsFromReward"):
        super().__init__(name, node)
        self.client = self.node.create_client(UpdateBelief, 'update_belief')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info('Servizio update_belief non disponibile, retry...')

        self._sent = False
        self._result_future = None
    
    def initialise(self):
        self._sent = False
        self._result_future = None

    def update(self):
        if not self._sent:
            reward = self.bb.get("reward")
            if reward is None:
                self.feedback_message = "Reward non disponibile"
                return py_trees.common.Status.FAILURE

            req = UpdateBelief.Request()
            req.real_score = float(reward)

            self._result_future = self.client.call_async(req)
            self._sent = True
            return py_trees.common.Status.RUNNING
        
        if self._result_future.done():
            resp = self._result_future.result()
            if resp is None:
                self.feedback_message = "Chiamata al servizio fallita"
                return py_trees.common.Status.FAILURE
            if resp.success:
                return py_trees.common.Status.SUCCESS
            else:
                self.feedback_message = "Update belief non riuscito"
                return py_trees.common.Status.FAILURE

        return py_trees.common.Status.RUNNING

#==============================================================================================================
# COSTRUZIONE ALBERO E AVVIO:

def create_tree(node: Node):
    # definizione target:
    poses = {
        "target_1": [0.4, 0.2, 0.3, 0, 0, 0, 1],
        "target_2": [0.5, -0.2, 0.35, 0, 0, 0, 1],
    }

    seq = py_trees.composites.Sequence("FullCycle")

    move_t1 = Retry(Timeout(MoveToPose(node, pose_list=poses["target_1"]), 30.0), 2)
    wait_t1 = Timeout(WaitRobotArrived(node, target_key="final_traj_joints", timeout_s=20), 25.0)
    vision_1 = Retry(Timeout(CallVisionService(node, estimate_volume=False, out_centroid_key="pos_cont_goal"), 15.0), 2)

    move_t2 = Retry(Timeout(MoveToPose(node, pose_list=poses["target_2"]), 40.0), 2)
    wait_t2 = Timeout(WaitRobotArrived(node, target_key="final_traj_joints", timeout_s=20), 25.0)
    vision_2 = Retry(Timeout(CallVisionService(node, estimate_volume=True, out_centroid_key="pos_init_cont", out_vol_key="init_vol"), 20.0), 2)

    move_c  = Retry(Timeout(MoveToPose(node, pose_from_bb="pos_init_cont"), 40.0), 2)
    wait_c = Timeout(WaitRobotArrived(node, target_key="final_traj_joints", timeout_s=20), 25.0)

    off     = ComputeOffset(node, "pos_init_ee", "pos_init_cont")
    grip    = Retry(Timeout(CloseGripper(node), 5.0), 2) # CloseGripper o CloseGripper1
    params  = SetPlanParams(node, theta_f=0.6, num_wp=50, target_vol=100.0)
    par_util = py_trees.composites.Parallel(
        "UtilitiesParallel",
        policy=py_trees.common.ParallelPolicy.SuccessOnAll()
    )
    par_util.add_children([off, grip, params])

    plan    = Retry(Timeout(CallPlannerSrv(node), 10.0), 2)
    execp   = Retry(Timeout(ExecutePathPublisher(node), 60.0), 1) # ExecutePathPublisher o ExecutePathAction
    reward  = WaitForReward(node, topic="/reward", timeout=None)
    update  = Retry(Timeout(UpdateSimParamsFromReward(node), 10.0), 2)

    seq.add_children([
        move_t1, wait_t1, vision_1,
        move_t2, wait_t2, vision_2, 
        move_c, wait_c, 
        par_util,
        plan, execp,
        reward, update
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
