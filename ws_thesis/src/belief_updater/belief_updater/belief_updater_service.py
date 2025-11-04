import rclpy
from rclpy.node import Node
import yaml
import random
from interfaces.srv import UpdateBelief  
import numpy as np
import copy

PARAMS_FILE = "/tmp/parameters.yaml"
SCORES_FILE = "/tmp/scores.yaml"
TOLERANCES_FILE = "/tmp/tolerances.yaml"
MAX_MODELS = 30
MIN_MODELS = 20

def is_success(score, threshold=0.5):
    return score > threshold

def update_parameters1(param, scale=0.1):
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

def sample_param(value, tol):
    
    if isinstance(tol, list):
        return tuple(tol)
    
    # tol = (neg,pos)  oppure ("rel",neg%,pos%)
    # caso assoluto
    if isinstance(tol, tuple) and isinstance(tol[0], (int, float)):
        neg, pos = tol
        if neg == 0 and pos == 0:
            return value
        sigma = (neg + pos) / 6.0  # 3σ = tol
        sample = np.random.normal(value, sigma)
        return float(np.clip(sample, value - neg, value + pos))

    # caso relativo
    if isinstance(tol, tuple) and tol[0] == "rel":
        _, neg_r, pos_r = tol
        neg = abs(value) * neg_r
        pos = abs(value) * pos_r
        if neg == 0 and pos == 0:
            return value
        sigma = (neg + pos) / 6.0
        sample = np.random.normal(value, sigma)
        return float(np.clip(sample, value - neg, value + pos))

    raise ValueError(f"Tolleranza non valida: {tol}")
    

def update_parameters(params, tolerances):
    new = copy.deepcopy(params)

    for key, val in params.items():
        if key not in tolerances:
            continue
        
        tol = tolerances[key]

        # vettori
        if isinstance(val, list):
            if not isinstance(tol, list) or len(val) != len(tol):
                raise ValueError(f"Mismatch tolleranze per {key}: {len(val)} vs {len(tol)}")

            out = [sample_param(v, t) for v, t in zip(val, tol)]

            # normalizza quaternion se 7 componenti: [x,y,z, qw,qx,qy,qz]
            if key in ("pos_init_ee", "pos_grip_ee") and len(out) == 7:
                q = np.array(out[3:], dtype=float)
                norm = np.linalg.norm(q)
                if norm > 0:
                    q = q / norm
                out[3:] = q.tolist()

            new[key] = out

        # scalari
        else:
            v = sample_param(val, tol)
            # cast a int per num_wp
            if key == "num_wp":
                v = int(round(v))
            new[key] = v

    return new

def scale_tolerances(tolerances, factor):
    new_tol = {}
    for key, tol in tolerances.items():
        if isinstance(tol, list):
            new_tol[key] = []
            for t in tol:
                if isinstance(t, tuple) and isinstance(t[0], (int,float)):
                    neg, pos = t
                    new_tol[key].append((neg*factor, pos*factor))
                else:
                    new_tol[key].append(t)
        elif isinstance(t, tuple) and isinstance(t[0], (int,float)):
            neg, pos = tol
            new_tol[key] = (neg*factor, pos*factor)
        else:
            new_tol[key] = tol
    return new_tol


# ------------------------------------------------------

class BeliefUpdater(Node):
    def __init__(self):
        super().__init__('belief_updater')
        self.srv = self.create_service(UpdateBelief, 'update_belief', self.updater_callback)
        self.get_logger().info("Belief updater service ready")

    def _load_yaml(self, path):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return data

    def updater_callback(self, request, response):
        real_result = is_success(request.real_score)
        self.get_logger().info(f"Real score={request.real_score:.3f} -> real_result={real_result}")

        # Carica set di parametri e score
        try:
            data_params = self._load_yaml(PARAMS_FILE)
            data_scores = self._load_yaml(SCORES_FILE)
            data_tolerances = self._load_yaml(TOLERANCES_FILE)
            parameters_set = data_params["parameters"]
            scores = data_scores["scores"]
            tolerances = data_tolerances["tolerances"]

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
        updated = list(param_new)
        while len(updated) < MIN_MODELS:
            for p in param_new:
                updated.append(update_parameters(p,tolerances))
                if len(updated) >= MIN_MODELS:
                    break
        if len(updated) > MAX_MODELS:
            updated = random.sample(updated, MAX_MODELS)

        # Salva su file
        try:
            with open(PARAMS_FILE, 'w') as f:
                yaml.safe_dump({"parameters": updated}, f, sort_keys=False)
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
