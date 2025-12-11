import os
clear = lambda: os.system('clear')
clear()
import os
import time
import rclpy
from rclpy.node import Node
from interfaces.srv import Simplan, UpdateBelief
from std_msgs.msg import Float32
import yaml
import paramiko

OUTPUT_FILE = "/tmp/best_path.yaml"
class CallPlannerSrv(Node):
    def __init__(self):
        super().__init__("call_planner_client")
        self.client_plan = self.create_client(Simplan, "plan_path")
        while not self.client_plan.wait_for_service(timeout_sec=2.0):
            self.get_logger().info("Servizio plan_path non disponibile, retry...")
        
        self.client_upd = self.create_client(UpdateBelief, 'update_belief')
        while not self.client_upd.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Servizio update_belief non disponibile, retry...')

        # Subscriber
        self.subscription = self.create_subscription(
            Float32,
            '/reward',
            self.reward_callback,
            10
        )

        self.future = None
        self.received_reward = None

    def call_service(self):
        req = Simplan.Request()
        req.no_params = True

        future = self.client_plan.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is None:
            raise RuntimeError("Chiamata al planner fallita")
        return future.result()
        
    def send_path(self):
        local_path = "/tmp/best_path.yaml"
        remote_path = "/tmp/best_path.yaml"

        host = "100.110.226.44"
        user = "edo"
        key_file = "/home/barutta/.ssh/id_edo"

        # Controllo chiave
        if not os.path.exists(key_file):
            self.get_logger().error(f"Chiave SSH non trovata: {key_file}")
            
        try:
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(host, username=user, key_filename=key_file)

            sftp = client.open_sftp()
            sftp.put(local_path, remote_path)
            sftp.close()
            client.close()

            self.get_logger().info("File inviato con successo")

        except Exception as e:
            self.get_logger().error(f"File transfer failed: {str(e)}")

    def reward_callback(self, msg):
        self.get_logger().info(f"Reward ricevuto: {msg.data}")
        self.received_reward = msg.data  # sblocca il ciclo principale

    def wait_for_reward(self):
        self.received_reward = None  # reset
        while rclpy.ok() and self.received_reward is None:
            rclpy.spin_once(self, timeout_sec=0.1)
        return float(self.received_reward)
    
    def update_belief_with_reward(self, reward):
        req = UpdateBelief.Request()
        req.real_score = float(reward)

        future = self.client_upd.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        result = future.result()
        if result is None:
            self.get_logger().error("Update belief fallito")
        else:
            self.get_logger().info(f"Update belief success={result.success}")   

def main():
    rclpy.init()
    node = CallPlannerSrv()

    try:
        old_reward=-1
        c=0
        for i in range(100):
            print(f"ITER {i}")

            resp = node.call_service()
            if not resp.success:
                c+=1
                if c>1:
                    break
                reward = -1
                print("Planning fallito")
            else:        
                node.send_path()
                reward = node.wait_for_reward()
                old_reward=reward
                print(f"Reward ottenuto = {reward}")

            node.update_belief_with_reward(reward)

            time.sleep(0.1)

        print("done")

    finally:
        node.destroy_node()
        rclpy.shutdown()

# def main():
#     rclpy.init()
#     node = CallPlannerSrv()
#     try:
#         c=0
#         resp = None
#         for i in range(100):        # 100 iterazioni
#             print(f"iter {i}")
            
#             while resp is None:
#                 resp = node.call_service()
            
#             if not resp.success:
#                 c+=1
#                 if c>1:
#                     break
#                 print("Planning fallito")
#             else:
#                 node.send_path()

#             node.spin_until_result()
#             resp = None
#             time.sleep(0.1)  # opzionale, evita martellamento
            
#         print("done")
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()

if __name__ == "__main__":
    main()

# import rclpy
# from rclpy.node import Node
# from interfaces.srv import Simplan, UpdateBelief

# class CallPlannerSrv(Node):
#     def __init__(self):
#         super().__init__("call_planner_client")

#         self.client_plan = self.create_client(Simplan, "plan_path")
#         while not self.client_plan.wait_for_service(timeout_sec=2.0):
#             self.get_logger().info("Attesa servizio plan_path...")

#         self.client_upd = self.create_client(UpdateBelief, "update_belief")
#         while not self.client_upd.wait_for_service(timeout_sec=2.0):
#             self.get_logger().info("Attesa servizio update_belief...")

#     def call_planner_blocking(self):
#         req = Simplan.Request()
#         req.no_params = True

#         future = self.client_plan.call_async(req)
#         rclpy.spin_until_future_complete(self, future)

#         if future.result() is None:
#             raise RuntimeError("Errore planner")

#         return future.result()

#     def update_belief_blocking(self, score):
#         req = UpdateBelief.Request()
#         req.real_score = float(score)

#         future = self.client_upd.call_async(req)
#         rclpy.spin_until_future_complete(self, future)

#         return future.result()

# def main():
#     rclpy.init()
#     node = CallPlannerSrv()

#     try:
#         for i in range(10000):
#             print(f"\nIterazione {i}")

#             # 1. Chiamo il planner
#             resp = node.call_planner_blocking()

#             # 2. Se success → chiedo reward positivo
#             if resp.success:
#                 reward = float(input("Planner SUCCESS. Inserisci reward: "))

#             # 3. Se non success → chiedo reward alternativo
#             else:
#                 reward = float(input("Planner FAIL. Inserisci reward opposto: "))

#             # 4. Aggiorno belief
#             upd = node.update_belief_blocking(reward)

#             print(f"Aggiornamento belief: success={upd.success}")

#     finally:
#         node.destroy_node()
#         rclpy.shutdown()

# if __name__ == "__main__":
#     main()
