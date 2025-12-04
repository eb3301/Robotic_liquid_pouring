import os
clear = lambda: os.system('clear')
clear()
import os
import time
import rclpy
from rclpy.node import Node
from interfaces.srv import Simplan, UpdateBelief
from std_msgs.msg import Float32

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

    def call_service(self):
        req = Simplan.Request()
        req.no_params = True

        future = self.client_plan.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is None:
            raise RuntimeError("Chiamata al planner fallita")
        return future.result()

    def reward_callback(self, msg):
        self.get_logger().info(f"Ricevuto reward: {msg.data}")

        # Prepara richiesta
        request = UpdateBelief.Request()
        request.real_score = float(msg.data)

        # Chiama servizio
        self.future = self.client_upd.call_async(request)

    def call_update(self):
        #self.get_logger().info(f"Ricevuto reward: {0}")
        user_input = int(input("Pianificazione fallita, esplora mettendo opposto di reward precedente: "))

        # Prepara richiesta
        request = UpdateBelief.Request()
        request.real_score = float(user_input)

        # Chiama servizio
        self.future = self.client_upd.call_async(request)

    def spin_until_result(self):
        while rclpy.ok():
            rclpy.spin_once(self)
            if self.future and self.future.done():
                result = self.future.result()
                if result is None:
                    self.get_logger().error("Chiamata fallita")
                else:
                    self.get_logger().info(f"Service call success={result.success}")
                break

def main():
    rclpy.init()
    node = CallPlannerSrv()
    try:
        for i in range(10000):        # 100 iterazioni
            print(f"iter {i}")

            resp = None
            while resp is None or not resp.success:
                resp = node.call_service()

            # if not resp.success:
            #     resp = None
            #     node.call_update()
            #     node.spin_until_result()
            #     resp = node.call_service()

            node.spin_until_result()
            
            time.sleep(0.1)  # opzionale, evita martellamento
        
        print("done")
    finally:
        node.destroy_node()
        rclpy.shutdown()

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
