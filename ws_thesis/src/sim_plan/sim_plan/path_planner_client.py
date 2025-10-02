import os
import time
import yaml
import rclpy
from rclpy.node import Node
from interfaces.srv import Simplan
import paramiko

PARAMS_FILE = "/tmp/init_parameters.yaml"
OUTPUT_FILE = "/tmp/best_path.yaml"

class CallPlannerSrv(Node):
    def __init__(self):
        super().__init__("call_planner_client")
        self.client = self.create_client(Simplan, "plan_path")
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Servizio plan_path non disponibile, retry...")

    def wait_for_init_file(self):
        self.get_logger().info(f"Aspetto file {PARAMS_FILE} ...")
        while not os.path.exists(PARAMS_FILE):
            time.sleep(1.0)
        self.get_logger().info(f"Trovato file {PARAMS_FILE}")

    def load_parameters(self):
        with open(PARAMS_FILE, "r") as f:
            data = yaml.safe_load(f)
        if "parameters" not in data:
            raise RuntimeError("File init_parameters.yaml non contiene chiave 'parameters'")
        return data["parameters"]

    def call_service(self, params: dict):
        req = Simplan.Request()
        req.pos_init_cont = params.get("pos_init_cont", [0.0, 0.0, 0.0])
        req.pos_init_ee   = params.get("pos_init_ee", [0.0]*7)
        req.pos_cont_goal = params.get("pos_cont_goal", [0.0, 0.0, 0.0])
        req.offset        = params.get("offset", [0.0, 0.0, 0.0])
        req.theta_f       = float(params.get("theta_f", 0.0))
        req.num_wp        = int(params.get("num_wp", 0))
        req.init_vol      = float(params.get("vol_init", 0.0))
        req.target_vol    = float(params.get("vol_target", 0.0))

        future = self.client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is None:
            raise RuntimeError("Chiamata al planner fallita")
        return future.result()

    def save_response(self, resp):
        out = {
            "best_path": resp.best_path,
            "time": resp.time.tolist() if hasattr(resp.time, "tolist") else list(resp.time),
        }
        with open(OUTPUT_FILE, "w") as f:
            yaml.safe_dump(out, f, sort_keys=False)
        self.get_logger().info(f"Risultato salvato in {OUTPUT_FILE}")
    
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

            self.node.get_logger().info("File inviato con successo")

        except Exception as e:
            self.node.get_logger().error(f"File transfer failed: {str(e)}")
        
def main():
    rclpy.init()
    node = CallPlannerSrv()
    try:
        node.wait_for_init_file()
        params = node.load_parameters()
        resp = node.call_service(params)
        node.save_response(resp)
        node.send_path()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
