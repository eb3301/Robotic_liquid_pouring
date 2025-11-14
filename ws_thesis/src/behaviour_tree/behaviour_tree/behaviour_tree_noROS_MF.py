import time
import rclpy
from rclpy.node import Node
import py_trees
from py_trees.blackboard import Blackboard
import numpy as np
from geometry_msgs.msg import PoseStamped, PointStamped
from sensor_msgs.msg import JointState
from moveit.planning import MoveItPy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from control_msgs.action import GripperCommand
from std_msgs.msg import Float32
from builtin_interfaces.msg import Duration as MsgDuration
from drims2_motion_server.motion_client import MotionClient
import yaml
import os
import paramiko
from tf2_ros import Buffer, TransformListener
from rclpy.duration import Duration
from rclpy.time import Time
from interfaces.srv import Perception
import tf2_geometry_msgs 
import threading
from rclpy.executors import MultiThreadedExecutor


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

# ---------- Motion ----------
class PrintPose(RosLeaf):
    def __init__(self, node, tf_buffer, target_frame="base_link", ee_frame="tool0", name="PrintPose"):
        super().__init__(name, node)
        self.tf_buffer = tf_buffer
        self.target_frame = target_frame
        self.ee_frame = ee_frame

    def initialise(self):
        self.node.get_logger().info("Print pose started")

    def update(self):
        
        time=Time() 
        # Wait for the transform asynchronously
        # tf_future = self.tf_buffer.wait_for_transform_async(
        # target_frame=self.target_frame,
        # source_frame=self.ee_frame,
        # time=time
        # )
        # rclpy.spin_until_future_complete(self.node, tf_future, timeout_sec=1)
        if not self.tf_buffer.can_transform(
            self.target_frame,
            self.ee_frame,
            Time(),
            timeout=Duration(seconds=0.5)
        ):
            self.feedback_message = "TF non pronta"
            return py_trees.common.Status.RUNNING

        # Lookup tansform
        try:
            t = self.tf_buffer.lookup_transform(self.target_frame,
                                            self.ee_frame,
                                            time)
            p = t.transform.translation
            q = t.transform.rotation
            self.node.get_logger().info(
                f"EE ({self.ee_frame}) in {self.target_frame}: "
                f"p=[{p.x:.3f}, {p.y:.3f}, {p.z:.3f}] "
                f"q=[{q.x:.3f}, {q.y:.3f}, {q.z:.3f}, {q.w:.3f}]"
            )
            return py_trees.common.Status.SUCCESS
        except Exception as e:
            self.node.get_logger().info(f"No transform found: {str(e)}")
            return py_trees.common.Status.RUNNING

class MoveToPose(RosLeaf):
    def __init__(self, node, tf_buffer, pose_list=None, pose_bb=None, motion_client=None, name="MoveToPose"):
        super().__init__(name, node)
        self.tf_buffer = tf_buffer
        self.pose_list = pose_list
        self.pose_bb = pose_bb
        self.motion_client = motion_client or MotionClient()
        if self.pose_list == "pos_init_cont":
            self.cartesian=True
        else:
            self.cartesian=False

    def initialise(self):
        self.node.get_logger().info(f"MoveToPose started {self.pose_bb}")

    def update(self):
        # Offset per mancanza base in sim:
        offset=0.02 # 2cm
        # se pose_list è chiave del blackboard, leggi una sola volta
        if isinstance(self.pose_list, str):
            val = self.bb.get(self.pose_list)    
            if val is not None and len(val)==3:
                quat=[-np.sqrt(2)/2, 0.0, 0.0, np.sqrt(2)/2]
                self.pose_list = val + quat
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
                result = self.motion_client.move_to_pose(pose_msg, cartesian_motion=self.cartesian) 
                if getattr(result, "val", 0) == 1:
                    if self.pose_bb is not None:
                        bb_grip=self.pose_bb + "_tip"
                        self.bb.set(bb_grip, self.pose_list)

                        target_frame="base_link"
                        ee_frame="tool0"

                        time=Time() 
                        # Wait for the transform asynchronously
                        # tf_future = self.tf_buffer.wait_for_transform_async(
                        # target_frame=target_frame,
                        # source_frame=ee_frame,
                        # time=time
                        # )
                        # rclpy.spin_until_future_complete(self.node, tf_future, timeout_sec=1)
                        if not self.tf_buffer.can_transform(
                            target_frame,
                            ee_frame,
                            Time(),
                            timeout=Duration(seconds=0.5)
                        ):
                            self.feedback_message = "TF non pronta"
                            return py_trees.common.Status.RUNNING

                        # Lookup tansform
                        try:
                            #rclpy.spin_once(self.node, timeout_sec=0.1)
                            t = self.tf_buffer.lookup_transform(target_frame,
                                                            ee_frame,
                                                            time)
                            p = t.transform.translation
                            q = t.transform.rotation
                            pose_val=[p.x, p.y, p.z+offset, q.x, q.y, q.z, q.w]
                            self.bb.set(self.pose_bb, pose_val)
                            self.node.get_logger().info(f"pose {self.pose_bb}: {pose_val}")
                        except Exception as e:
                            self.node.get_logger().warn(f"TF lookup failed: {e}")
                            self.bb.set(self.pose_bb, None)

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
                if getattr(result, "val", 0) == 1:
                    if self.pose_bb is not None:
                        target_frame="base_link"
                        ee_frame="tool0"

                        time=Time() 
                        # Wait for the transform asynchronously
                        # tf_future = self.tf_buffer.wait_for_transform_async(
                        # target_frame=target_frame,
                        # source_frame=ee_frame,
                        # time=time
                        # )
                        # rclpy.spin_until_future_complete(self.node, tf_future, timeout_sec=1)
                        if not self.tf_buffer.can_transform(
                            target_frame,
                            ee_frame,
                            Time(),
                            timeout=Duration(seconds=0.5)
                        ):
                            self.feedback_message = "TF non pronta"
                            return py_trees.common.Status.RUNNING

                        # Lookup tansform
                        try:
                            #rclpy.spin_once(self.node, timeout_sec=0.1)
                            t = self.tf_buffer.lookup_transform(target_frame,
                                                            ee_frame,
                                                            time)
                            p = t.transform.translation
                            q = t.transform.rotation
                            pose_val=[p.x, p.y, p.z+offset, q.x, q.y, q.z, q.w]
                            self.bb.set(self.pose_bb, pose_val)
                            #self.node.get_logger().info(f"pose {self.pose_bb}: {pose_val}")
                        except Exception as e:
                            self.node.get_logger().warn(f"TF lookup failed: {e}")
                            self.bb.set(self.pose_bb, None)
                        
                    return py_trees.common.Status.SUCCESS
                else:
                    self.feedback_message = "Movimento fallito"
                    return py_trees.common.Status.FAILURE
            except Exception as e:
                self.feedback_message = f"Errore move_to_pose: {e}"
                return py_trees.common.Status.FAILURE

# ---------- Perception ----------
class CallVisionService(RosLeaf):
    def __init__(self, node, estimate_volume: bool, 
                 out_centroid_key=None,
                 out_vol_key="init_vol",
                 name="CallVisionService"):
        super().__init__(name, node)
        self.estimate_volume = estimate_volume
        self.out_centroid_key = out_centroid_key
        self.out_vol_key = out_vol_key

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
        

        if not self._sent:
            req = Perception.Request()
            req.estimate_volume = self.estimate_volume
            self._future = self.client.call_async(req)
            self._sent = True
            return py_trees.common.Status.RUNNING

        if self._future is None or not self._future.done():
            return py_trees.common.Status.RUNNING

        try:
            resp = self._future.result()
            if resp is None or not resp.success:
                self.feedback_message = "Vision service fallito"
                msg = getattr(resp, "message", "no message")
                self.node.get_logger().warn(f"Vision service fallito: {msg}")
                return py_trees.common.Status.FAILURE

            centroid=resp.centroid 
            centroid[1]+=0.03
            centroid[2]=max(0.04,centroid[2])

            pre_centroid=centroid.copy()
            pre_centroid[1]-=0.15 # nel frame base_link la y è la direzione x in world
            pre_key=self.out_centroid_key+"_pre"

            # Salva risultati nel blackboard
            self.bb.set(self.out_centroid_key, list(centroid))
            self.bb.set(pre_key, list(pre_centroid))
            if self.estimate_volume:
                self.bb.set(self.out_vol_key, resp.volume)

            self.node.get_logger().info(f"Vision completed: {centroid}")
            return py_trees.common.Status.SUCCESS

        except Exception as e:
            self.feedback_message = str(e)
            self.node.get_logger().error(f"Errore VisionService: {e}")
            return py_trees.common.Status.FAILURE

# ---------- Gripper ----------
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
        offset = [-ee[0]+cont[0], -ee[1]+cont[1], -ee[2]+cont[2]]
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
            goal.command.position = 0.0    
            goal.command.max_effort = 0.0 # min_F = 20N 

            self._goal_future = self.client.send_goal_async(goal)
            self._goal_future.add_done_callback(self._goal_response_cb)
            self._sent = True
            return py_trees.common.Status.RUNNING

        if self._result_future and self._result_future.done():
            result = self._result_future.result().result
            
            if getattr(result, "stalled", True) or getattr(result, "goal_reached", True):
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
            goal.command.position = 0.049  
            goal.command.max_effort = 0.1  

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
    
# ---------- Parameters --------
class SetPlanParams(RosLeaf):
    def __init__(self, node, tf_buffer, theta_f, num_wp, target_vol, name="SetPlanParams"):
        super().__init__(name, node)
        self.tf_buffer = tf_buffer
        self.theta_f = theta_f 
        self.num_wp = num_wp
        self.target_vol = target_vol
    
    def transform_to_world(self, point_list):
        """Converte una posizione [x, y, z] da base_link a world"""
        to_frame_rel = 'world'
        from_frame_rel = 'base_link'
        time=Time() 

        if len(point_list)<=3:
            ps = PointStamped()
            ps.header.frame_id = "base_link"
            ps.header.stamp = self.node.get_clock().now().to_msg()
            ps.point.x, ps.point.y, ps.point.z = map(float, point_list)
 
            # Wait for the transform asynchronously
            # tf_future = self.tf_buffer.wait_for_transform_async(
            # target_frame=to_frame_rel,
            # source_frame=from_frame_rel,
            # time=time
            # )
            # rclpy.spin_until_future_complete(self.node, tf_future, timeout_sec=1)
            if not self.tf_buffer.can_transform(
                to_frame_rel,
                from_frame_rel,
                time,
                timeout=rclpy.duration.Duration(seconds=0.5)
            ):
                self.node.get_logger().warn(f"No transform {from_frame_rel} → {to_frame_rel}")
                return None

            # Lookup tansform
            try:
                t = self.tf_buffer.lookup_transform(to_frame_rel,
                                                from_frame_rel,
                                                time)
                # Do the transform
                transformed_point_msg = tf2_geometry_msgs.do_transform_point(ps, t)
                transformed_point=[float(transformed_point_msg.point.x), float(transformed_point_msg.point.y), float(transformed_point_msg.point.z)]
                return transformed_point
                
            except Exception as e:
                self.node.get_logger().warn(f"No transform found: {str(e)}")
                return point_list
            
        else:
            ps = PoseStamped()
            ps.header.frame_id = "base_link"
            ps.header.stamp = self.node.get_clock().now().to_msg()
            ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = map(float, point_list[:3])
            ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w = map(float, point_list[3:])

            # Wait for the transform asynchronously
            # tf_future = self.tf_buffer.wait_for_transform_async(
            # target_frame=to_frame_rel,
            # source_frame=from_frame_rel,
            # time=time
            # )
            # rclpy.spin_until_future_complete(self.node, tf_future, timeout_sec=1)
            if not self.tf_buffer.can_transform(
                to_frame_rel,
                from_frame_rel,
                time,
                timeout=rclpy.duration.Duration(seconds=0.5)
            ):
                self.node.get_logger().warn(f"No transform {from_frame_rel} → {to_frame_rel}")
                return None

            # Lookup tansform
            try:
                t = self.tf_buffer.lookup_transform(to_frame_rel,
                                                from_frame_rel,
                                                time)
                # Do the transform
                transformed_point_msg = tf2_geometry_msgs.do_transform_pose_stamped(ps, t)
                transformed_point = [
                    float(transformed_point_msg.pose.position.x), float(transformed_point_msg.pose.position.y), float(transformed_point_msg.pose.position.z),
                    float(transformed_point_msg.pose.orientation.x), float(transformed_point_msg.pose.orientation.y),
                    float(transformed_point_msg.pose.orientation.z), float(transformed_point_msg.pose.orientation.w)
                ]
                return transformed_point
            except Exception as e:
                self.node.get_logger().warn(f"No transform found: {str(e)}")  
                return point_list
            
        
    def update(self):
        self.bb.set("theta_f", self.theta_f)
        self.bb.set("num_wp", self.num_wp)
        self.bb.set("target_vol", self.target_vol)

        # Debug purposes:
        # self.bb.set("pos_init_cont", [0.85, 0.2, 0.92]),
        # self.bb.set("pos_init_ee",[0.0]*7),
        # self.bb.set("pos_cont_goal", [0.85, 0.7, 0.92]),
        # self.bb.set("offset", [0.0,-0.04,0.13]),
        # self.bb.set("init_vol", 40.0),

                
        pos_init_cont = self.bb.get("pos_init_cont") or [0.0, 0.0, 0.0]
        pos_init_ee = self.bb.get("pos_init_ee") or [0.0]*7
        pos_cont_goal = self.bb.get("pos_cont_goal") or [0.0, 0.0, 0.0]
        pos_grip_ee = self.bb.get("pos_grip_ee") or [0.0]*7

        #self.node.get_logger().info(f"pos_init_cont {pos_init_cont}")
        #self.node.get_logger().info(f"pos_init_ee {pos_init_ee}")
        #self.node.get_logger().info(f"pos_cont_goal {pos_cont_goal}")
        #self.node.get_logger().info(f"pos_grip_ee {pos_grip_ee}")

        pos_init_cont = self.transform_to_world(pos_init_cont)
        pos_cont_goal = self.transform_to_world(pos_cont_goal)
        pos_init_ee = self.transform_to_world(pos_init_ee)
        pos_grip_ee = self.transform_to_world(pos_grip_ee)

        #self.node.get_logger().info(f"pos_init_cont {pos_init_cont}")
        self.node.get_logger().info(f"pos_init_ee {pos_init_ee}")
        #self.node.get_logger().info(f"pos_cont_goal {pos_cont_goal}")
        self.node.get_logger().info(f"pos_grip_ee {pos_grip_ee}")

        try:
            init_parameters = {
                "pos_init_cont": list(pos_init_cont),
                "pos_cont_goal": list(pos_cont_goal),
                "pos_init_ee": list(pos_init_ee),
                "pos_grip_ee":list(pos_grip_ee),
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

            init_parameters = self.to_builtin(init_parameters)
            self.bb.set("init_parameters", init_parameters)
            with open("/tmp/init_parameters.yaml", "w") as f:
                yaml.safe_dump({"parameters": init_parameters}, f, sort_keys=False)
            self.node.get_logger().info("File initial parameters created")
            return py_trees.common.Status.SUCCESS
        except Exception as e:
            self.node.get_logger().error(f"File creation failed: {str(e)}")
            return py_trees.common.Status.FAILURE
    
    def to_builtin(self, obj):
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, list):
            return [self.to_builtin(x) for x in obj]
        if isinstance(obj, dict):
            return {k: self.to_builtin(v) for k, v in obj.items()}
        return obj
        
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
                    time_arr = data.get("best_path", {}).get("time", [])
                    path = data.get("best_path", {}).get("all", [])
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

class ExecutePathPublisher(RosLeaf):
    joint_name_map = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]

    def __init__(self, node, name="ExecutePathPublisher", tol=0.01, grace_t=10.0, motion_client=None):
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
        self.tol = tol
        self.grace_t = grace_t
        self.motion_client = motion_client or MotionClient()

    def _joint_state_cb(self, msg):
        self._last_joint_state = msg

    def initialise(self):
        self._sent = False
        self._traj_duration = 0.0
        self.bb.set("traj_start_time", self.node.get_clock().now().nanoseconds / 1e9)
        self.time_arr = self.bb.get("time")
        self.path = self.bb.get("best_path")
        for p in self.path:
            p[0]-=np.pi


    def update(self):
        path=self.path
        time_arr=self.time_arr
     
        if time_arr is None or path is None:
            self.feedback_message = "Traiettoria non disponibile"
            self.node.get_logger().warn("Traiettoria non disponibile")
            return py_trees.common.Status.FAILURE

        if not self._sent:
            if self._last_joint_state is None:
                self.feedback_message = "Joint state non ancora ricevuto"
                return py_trees.common.Status.RUNNING

            # Controlla che la posizione iniziale sia coerente con la traiettoria
            init_q = path[0][:6]
            if not self._check_initial_joints_close(init_q):
                self.feedback_message = "Posizione iniziale non coerente con la traiettoria"
                self.node.get_logger().warn(self.feedback_message)
                self.motion_client.move_to_joint(init_q)
                return py_trees.common.Status.RUNNING

            traj = JointTrajectory()
            traj.joint_names = self.joint_name_map

            for t, q in zip(time_arr, path):
                pt = JointTrajectoryPoint()
                pt.positions = q[:6]
                pt.time_from_start = MsgDuration(sec=int(t), nanosec=int((t % 1.0) * 1e9))
                traj.points.append(pt)

            self.pub.publish(traj)
            self._sent = True
            self._traj_duration = float(time_arr[-1])
            self.bb.set("goal_joints", path[-1][:6])
            return py_trees.common.Status.RUNNING
        
        speed_scaling=0.32
        elapsed = ((self.node.get_clock().now().nanoseconds / 1e9) - self.bb.get("traj_start_time"))*speed_scaling
        if elapsed >= self._traj_duration + self.grace_t:
            if self._check_final_joints_close():
                return py_trees.common.Status.SUCCESS
            else:
                self.feedback_message = "Joint finali fuori tolleranza"
                return py_trees.common.Status.FAILURE

        return py_trees.common.Status.RUNNING

    def _check_final_joints_close(self):
        goal = self.bb.get("goal_joints")
        if self._last_joint_state is None or goal is None:
            return False

        name_to_idx = {n: i for i, n in enumerate(self._last_joint_state.name)}
        current_pos = []
        for j in self.joint_name_map:
            if j not in name_to_idx:
                return False
            current_pos.append(self._last_joint_state.position[name_to_idx[j]])

        err = np.linalg.norm(np.array(goal) - np.array(current_pos), ord=np.inf)
        return err < self.tol

    def _check_initial_joints_close(self, init_q):
        if self._last_joint_state is None:
            return False

        name_to_idx = {n: i for i, n in enumerate(self._last_joint_state.name)}
        current_pos = []
        for j in self.joint_name_map:
            if j not in name_to_idx:
                return False
            current_pos.append(self._last_joint_state.position[name_to_idx[j]])

        err = np.linalg.norm(np.array(init_q) - np.array(current_pos), ord=np.inf)
        return err < self.tol

#==============================================================================================================
# COSTRUZIONE ALBERO E AVVIO:

def create_tree(node: Node, tf_buffer, motion_client):

    #genesis joints = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint', 'hande_left_finger_joint', 'hande_right_finger_joint']
    #ros2 joints    = [elbow_joint, robotiq_hande_left_finger_joint, shoulder_lift_joint, shoulder_pan_joint, wrist_1_joint, wrist_2_joint, wrist_3_joint,
    
    #joint_v1= [0.7227166962826039,-1.746930173633286, -2.2322865329350017, -2.046302405379515, 0.738723064687373, 2.948454781562834]
    #joint_v2 = [-3.129748565398613, -2.1683139224910026, -2.134126744414425, -3.519583411401461, -2.9772426124069256, -1.5698350868947821]

    joint_v1=[0.7012355923652649, -1.7084723911681117, -2.219346523284912, -1.8182255230345667, 0.793083667755127, -3.5496469179736536]
    joint_v2=[-3.4113157431231897, -1.5812603435912074, -2.313349723815918, -1.3917177480510254, -3.618211809788839, -2.1802199522601526]
    #joint_v2= [-2.9784508387195032, -2.2492810688414515, -1.4298287630081177, -1.9104792080321253, -3.66062838235964, -2.5633793512927454]
    
    open=OpenGripper(node)
    move_t1 = MoveToPose(node, tf_buffer, pose_list=joint_v1,pose_bb=None, motion_client=motion_client)
    vision_1 = CallVisionService(node, estimate_volume=False, out_centroid_key="pos_cont_goal")

    move_t2 = MoveToPose(node, tf_buffer, pose_list=joint_v2,pose_bb="pos_init_ee", motion_client=motion_client)
    vision_2 = CallVisionService(node, estimate_volume=True, out_centroid_key="pos_init_cont", out_vol_key="init_vol")

    #joint_v3 = [-1.5371907393084925, -1.2124689680388947, -2.2734172344207764, -2.786943098107809, -1.552525822316305, -3.143904987965719]
    joint_v3 = [-2.08985406557192, -1.8842722378172816, -2.6087613105773926, -1.8289934597411097, -2.0932639280902308, -3.162097756062643]
    move_t3 = MoveToPose(node, tf_buffer, pose_list=joint_v3,pose_bb=None, motion_client=motion_client)

    
    move_pre_c = MoveToPose(node, tf_buffer, pose_list="pos_init_cont_pre", pose_bb=None, motion_client=motion_client)
    move_c = MoveToPose(node, tf_buffer, pose_list="pos_init_cont", pose_bb="pos_grip_ee", motion_client=motion_client)

    off = ComputeOffset(node, "pos_grip_ee", "pos_init_cont")
    close = CloseGripper(node)
    par_util = py_trees.composites.Parallel(
        "UtilitiesParallel",
        policy=py_trees.common.ParallelPolicy.SuccessOnAll()
    )
    par_util.add_children([off, close])
    params  = SetPlanParams(node, tf_buffer, theta_f=90, num_wp=350, target_vol=20.0)
    send = SendYamlToVM(node)
    wait_path = WaitForBestPath(node)
    execp   = ExecutePathPublisher(node, motion_client=motion_client)
    pose=PrintPose(node, tf_buffer)

    seq = py_trees.composites.Sequence("FullCycle",memory=True)
    
    seq.add_children([
        open,
        move_t1, vision_1,
        move_t2, vision_2, 
        move_t3, move_pre_c,
        move_c, par_util, params,
        send,
        wait_path,
        execp,
        ])

    #pose_c=[0.231, 0.578, 0.043,-0.762, 0.002, -0.007, 0.647] # x,y,z,x,y,z,w
    #pose_c=[0.261, 0.535, 0.044, -0.730, -0.000, -0.007, 0.683]
    #pose_c=[0.261, 0.535, 0.043, -np.sqrt(2)/2, 0.000, 0.000, np.sqrt(2)/2]

    # joint_c=[0.7406882047653198-np.pi, -2.323422431945801, -1.83205818176269753, -2.117997884750366, -2.4009978771209717, -3.1415913105010986]
    # move_c_test = MoveToPose(node, tf_buffer, pose_list=joint_c, pose_bb="pos_grip_ee", motion_client=motion_client)
    # # 42, -133, -106, -121, -137, -180
    # seq.add_children([
    #     open,
    #     move_t1, 
    #     move_t2,  
    #     move_t3,
    #     move_c_test,
    #     wait_path,
    #     execp,
    #     ])

    pour = py_trees.composites.Sequence("FullCycle",memory=True)
    pour.add_children([WaitForBestPath(node), ExecutePathPublisher(node, motion_client=motion_client)])
    return seq

def main():
    rclpy.init()
    node = Node("bt_orchestrator")

    tf_buffer = Buffer()
    tf_listener = TransformListener(tf_buffer, node)

    motion_client = MotionClient()

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.add_node(motion_client)
    threading.Thread(target=executor.spin, daemon=True).start()

    tree = py_trees.trees.BehaviourTree(create_tree(node, tf_buffer, motion_client))
    try:
        while rclpy.ok():
            tree.tick()
            time.sleep(0.01)           
    finally:
        executor.shutdown()
        motion_client.destroy_node()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()

