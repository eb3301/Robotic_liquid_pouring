import os
clear = lambda: os.system('clear')
clear()
import os
import time
import rclpy
from rclpy.node import Node
from interfaces.srv import Simplan, UpdateBelief


class CallPlannerSrv(Node):
    def __init__(self):
        super().__init__("call_planner_client")
        self.client_plan = self.create_client(Simplan, "plan_path")
        while not self.client_plan.wait_for_service(timeout_sec=2.0):
            self.get_logger().info("Servizio plan_path non disponibile, retry...")
        
        self.client_upd = self.create_client(UpdateBelief, 'update_belief')
        while not self.client_upd.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Servizio update_belief non disponibile, retry...')

        self.future = None

    def call_service(self):
        req = Simplan.Request()
        req.no_params = True # to use the following params

        future = self.client_plan.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is None:
            raise RuntimeError("Chiamata al planner fallita")
        return future.result()
    
    def reward_callback(self):
        # Prepara richiesta
        request = UpdateBelief.Request()
        request.real_score = float(1)

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
        for i in range(100):        # 100 iterazioni
            print(f"iter {i}")

            resp = None
            while resp is None or not resp.success:
                resp = node.call_service()

            node.reward_callback()
            node.spin_until_result()
            
            time.sleep(0.1)  # opzionale, evita martellamento
        
        print("done")
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main1():
    rclpy.init()
    node = CallPlannerSrv()
    try:
        resp = None
        while resp is None or resp.success is False:
            resp = node.call_service()
        node.reward_callback()
        node.spin_until_result()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
