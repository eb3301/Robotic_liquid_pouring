import rclpy
from rclpy.node import Node
import yaml
import random
from interfaces.srv import UpdateBelief  

PARAMS_FILE = "/tmp/parameters_set.yaml"
SCORES_FILE = "/tmp/score_best_path.yaml"
MAX_MODELS = 30

def is_success(score, threshold=0.5):
    return score > threshold

def update_parameters(param, scale=0.1):
    new_param = {}
    for key, val in param.items():
        if isinstance(val, float):
            import numpy as np
            noise = np.random.normal(0, scale * abs(val))
            new_param[key] = float(val + noise)
        elif isinstance(val, list) and all(isinstance(v, float) for v in val):
            import numpy as np
            new_param[key] = [float(v + np.random.normal(0, scale * abs(v))) for v in val]
        else:
            new_param[key] = val
    return new_param
# ------------------------------------------------------

class BeliefUpdater(Node):
    def __init__(self):
        super().__init__('belief_updater')
        self.srv = self.create_service(UpdateBelief, 'update_belief', self.updater_callback)
        self.get_logger().info("Belief updater service ready")

    def _load_yaml_list(self, path, what):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        if not isinstance(data, list):
            raise ValueError(f"{what} in {path} deve essere una lista")
        return data

    def updater_callback(self, request, response):
        real_result = is_success(request.real_score)
        self.get_logger().info(f"Real score={request.real_score:.3f} -> real_result={real_result}")

        # Carica set di parametri e score
        try:
            parameters_set = self._load_yaml_list(PARAMS_FILE, "parameters_set")
            scores = self._load_yaml_list(SCORES_FILE, "scores")
        except Exception as e:
            self.get_logger().error(f"Errore caricamento YAML: {e}")
            response.success = False
            return response

        if len(scores) != len(parameters_set):
            self.get_logger().warn(f"Dimensioni diverse: scores={len(scores)} vs params={len(parameters_set)}; uso min(n).")
        n = min(len(scores), len(parameters_set))

        # Filtra i parametri coerenti col risultato reale
        param_new = [p for i, p in enumerate(parameters_set[:n]) if is_success(scores[i]) == real_result]
        if len(param_new) == 0:
            self.get_logger().warn("Tutte le ipotesi eliminate.")
            response.success = False
            return response

        # Resampling
        new_samples = [update_parameters(p) for p in param_new]
        updated = param_new + new_samples
        # Limita a MAX_MODELS
        if len(updated) > MAX_MODELS:
            updated = random.sample(updated, MAX_MODELS)

        # Salva su file
        try:
            with open(PARAMS_FILE, 'w') as f:
                yaml.safe_dump(updated, f, sort_keys=False)
        except Exception as e:
            self.get_logger().error(f"Errore salvataggio YAML: {e}")
            response.success = False
            return response

        self.get_logger().info(f"Belief set aggiornato: {len(updated)} modelli")
        response.success = True
        return response

def main(args=None):
    rclpy.init(args=args)
    node = BeliefUpdater()
    rclpy.spin(node)
    rclpy.shutdown()
