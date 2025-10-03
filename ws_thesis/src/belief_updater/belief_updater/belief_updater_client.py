import os
clear = lambda: os.system('clear')
clear()
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from interfaces.srv import UpdateBelief   


class RewardClient(Node):
    def __init__(self):
        super().__init__('reward_client')

        # Subscriber
        self.subscription = self.create_subscription(
            Float32,
            '/reward',
            self.reward_callback,
            10
        )

        # Service client
        self.client = self.create_client(UpdateBelief, 'update_belief')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Servizio update_belief non disponibile, retry...')

        self.future = None

    def reward_callback(self, msg):
        self.get_logger().info(f"Ricevuto reward: {msg.data}")

        # Prepara richiesta
        request = UpdateBelief.Request()
        request.real_score = float(msg.data)

        # Chiama servizio
        self.future = self.client.call_async(request)

    def spin_until_result(self):
        while rclpy.ok():
            rclpy.spin_once(self)
            if self.future and self.future.done():
                result = self.future.result()
                if result is None:
                    self.get_logger().error("Chiamata fallita")
                else:
                    self.get_logger().info(f"Risultato: success={result.success}")
                break


def main():
    rclpy.init()
    node = RewardClient()
    node.spin_until_result()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
