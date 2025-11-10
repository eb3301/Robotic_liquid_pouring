import rclpy
from rclpy.node import Node
import yaml
import random
from interfaces.srv import UpdateBelief  
import numpy as np
import copy
import os
from scipy.special import expit

PARAMS_FILE = "/tmp/parameters.yaml"
SCORES_FILE = "/tmp/scores.yaml"
TOLERANCES_FILE = "/tmp/tolerances.yaml"
SUCCESS_PATH_FILE = "/tmp/success_path.yaml"
FILE_TS = "/tmp/TS.yaml"
FILE_CURRENT_PLAN_PARAMS="/tmp/current_plan_params.yaml"

MAX_MODELS = 30
MIN_MODELS = 15
PATH_NUM = 3

def is_success(score, threshold=0.5):
    return score > threshold

def sample_param(value, tol):
    
    if isinstance(tol, list):
        tol = tuple(tol)
    
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

def scale_tolerances(tol, factor):
    """
    Scala tolleranze di qualsiasi struttura ricorsiva.
    Gestisce:
      - [neg,pos] -> (neg*factor,pos*factor)
      - ("rel",neg%,pos%) -> invariato
      - scalari -> scalati
      - liste annidate -> ricorsive
    """
    # tuple/liste di 3 elementi: caso relativo ["rel", neg_r, pos_r]
    if (isinstance(tol, (list, tuple))
        and len(tol) == 3
        and tol[0] == "rel"):
        # non scalare percentuali
        return ("rel", tol[1], tol[2])

    # coppia numerica [neg,pos]
    if (isinstance(tol, (list, tuple))
        and len(tol) == 2
        and all(isinstance(x, (int,float)) for x in tol)):
        neg, pos = tol
        return (neg * factor, pos * factor)

    # lista generica -> ricorsione su elementi
    if isinstance(tol, list):
        return [scale_tolerances(t, factor) for t in tol]

    # numero singolo
    if isinstance(tol, (int, float)):
        return tol * factor

    # stringa o altro → lascia invariato
    return tol

def scale_tol(obj, factor):
    """
    Ricorsiva, robusta, ordina i casi in modo logico.
    Scala solo:
      - liste/tuple [neg,pos]
      - scalari numerici
    Non scala:
      - ["rel", a, b]
      - stringhe e dati non numerici
    """

    # numpy scalari -> Python scalari
    if isinstance(obj, np.generic):
        return scale_tol(obj.item(), factor)

    # numpy array -> lista ricorsiva
    if isinstance(obj, np.ndarray):
        return [scale_tol(x, factor) for x in obj.tolist()]

    # caso relativo: ["rel", neg%, pos%]
    if (
        isinstance(obj, (list, tuple))
        and len(obj) == 3
        and isinstance(obj[0], str)
        and obj[0] == "rel"
    ):
        return ["rel", obj[1], obj[2]]  # invariato

    # caso coppia numerica [neg,pos]
    if (
        isinstance(obj, (list, tuple))
        and len(obj) == 2
        and all(isinstance(x, (int, float, np.generic)) for x in obj)
    ):
        return [float(obj[0]) * factor, float(obj[1]) * factor]

    # singolo numero
    if isinstance(obj, (int, float, np.generic)):
        return float(obj) * factor

    # lista -> ricorsione
    if isinstance(obj, list):
        return [scale_tol(x, factor) for x in obj]

    # tupla -> ricorsione + ritorno lista
    if isinstance(obj, tuple):
        return [scale_tol(x, factor) for x in obj]

    # dict -> ricorsione
    if isinstance(obj, dict):
        return {k: scale_tol(v, factor) for k, v in obj.items()}

    # qualsiasi altro tipo -> invariato
    return obj

def update_w(theta, y, w_mean, w_cov):
    X = np.vstack([np.ones(len(theta)), theta]).T # modello logistico lineare (lin logit)
    w = w_mean.copy()
    # Laplace approx of posterior using Newton 
    for _ in range(5):
        p = expit(X @ w) # funzione sigmoide di scipy ottimizzata
        W = np.diag(p*(1-p))
        H = X.T @ W @ X + np.linalg.inv(w_cov)
        g = X.T @ (y - p) - np.linalg.inv(w_cov) @ (w - w_mean)
        try:
            w = w + np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
    w_cov_post = np.linalg.inv(H)
    return w, w_cov_post

def sample_x_TS(w_mean, w_cov, x_min, x_max, M, n_grid=50):
    thetas = np.linspace(x_min, x_max, n_grid)
    w_samples = np.random.multivariate_normal(w_mean, w_cov, size=M)
    x_nexts = []
    for ws in w_samples:
        scores = expit(ws[0] + ws[1]*thetas)
        x_nexts.append(thetas[np.argmax(scores)])
    return np.array(x_nexts)

def best_theta_greedy(w_mean, x_min, x_max, n_grid=200):
    grid = np.linspace(x_min, x_max, n_grid)
    p = expit(w_mean[0] + w_mean[1]*grid)
    return float(grid[np.argmax(p)])

def best_theta_bayes(w_mean, w_cov, x_min, x_max, M=200, n_grid=200):
    grid = np.linspace(x_min, x_max, n_grid)
    acc = np.zeros_like(grid)
    ws = np.random.multivariate_normal(w_mean, w_cov, size=M)
    for w in ws:
        acc += expit(w[0] + w[1]*grid)
    return float(grid[np.argmax(acc / M)])

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

    def to_builtin(self, obj):
        # numpy scalari -> tipo Python
        if isinstance(obj, np.generic):
            return obj.item()

        # numpy array -> lista Python ricorsiva
        if isinstance(obj, np.ndarray):
            return [self.to_builtin(x) for x in obj.tolist()]

        # lista -> lista pulita
        if isinstance(obj, list):
            return [self.to_builtin(x) for x in obj]

        # tupla -> lista (YAML gestisce le liste meglio, ed è ok perdere l'immutabilità)
        if isinstance(obj, tuple):
            return [self.to_builtin(x) for x in obj]

        # dict -> dict pulito
        if isinstance(obj, dict):
            return {k: self.to_builtin(v) for k, v in obj.items()}

        # tipi base (int, float, str, bool, None) restano così
        return obj

    def updater_callback(self, request, response):
        real_result = is_success(request.real_score)
        self.get_logger().info(f"Real score={request.real_score:.3f} -> real_result={real_result}")

        # Simulation Parameters Update:
        # Carica set di parametri e score
        try:
            data_params = self._load_yaml(PARAMS_FILE)
            data_scores = self._load_yaml(SCORES_FILE)
            data_tolerances = self._load_yaml(TOLERANCES_FILE)
            parameters_set = data_params["parameters"]
            scores = data_scores["scores"] # CONTIENE IN REALTÀ SUCCESS (BINARIO 0/1)
            tolerances = data_tolerances["tolerances"]
            k = data_tolerances.get("iteration", 0)
           
            # Fattore shrinking
            # k = iterazione corrente, H = orizzonte previsto
            # f0 = valore iniziale, f_min = minimo da raggiungere dopo H iteraz
            H=2000
            f0, f_min = 1.0, 0.0001
            tau = H / np.log(f0 / f_min)   # es: H=1000 => tau≈334
            factor = max(f_min, f0 * np.exp(-k / tau))
            # Re-heating per avere + esploraz 
            #boost_every, boost = 500, 1.4
            #factor = min(1.0, factor * boost) if k % boost_every == 0 else factor


            # Applica scaling
            tolerances_scaled = scale_tol(tolerances, factor)
            print(tolerances_scaled)

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
                updated.append(update_parameters(p,tolerances_scaled))
                if len(updated) >= MIN_MODELS:
                    break
        if len(updated) > MAX_MODELS:
            updated = random.sample(updated, MAX_MODELS)

        # Salva su file
        try:
            with open(PARAMS_FILE, 'w') as f:
                yaml.safe_dump({"parameters": updated}, f, sort_keys=False)
                data_tolerances["iteration"] = k + 1
            with open(TOLERANCES_FILE, 'w') as f:
                yaml.safe_dump(data_tolerances, f, sort_keys=False)

        except Exception as e:
            self.get_logger().error(f"Errore salvataggio YAML: {e}")
            response.success = False
            return response
        
        ##################################################################################
        ##################################################################################
        ##################################################################################
        
        # Planning Parameters Update:
        try:
            data_success_path = self._load_yaml(SUCCESS_PATH_FILE)
            success_path = data_success_path["success_path"]
        except Exception as e:
            self.get_logger().error(f"Errore caricamento YAML: {e}")
            response.success = False
            return response
        
        # Se il successo del path coincide con quello reale --> aggiorna theta/num_wp
        if success_path == real_result:
            # Carica file necessari per update
            try:
                if not os.path.exists(FILE_TS):
                    k=0

                    x_hist_theta = []
                    y_hist_theta = []
                    w_mean_theta = np.zeros(2)
                    w_cov_theta = np.eye(2)*10.0 

                    x_hist_num_wp = []
                    y_hist_num_wp = []
                    w_mean_num_wp = np.zeros(2)
                    w_cov_num_wp = np.eye(2)*50.0 

                else:
                    with open(FILE_TS, "r") as f:
                        data_TS = yaml.safe_load(f)

                    k = data_TS["k"]

                    x_hist_theta = data_TS["x_hist_theta"] 
                    y_hist_theta = data_TS["y_hist_theta"] # lista di 1=success,0=failure
                    w_mean_theta = data_TS["w_mean_theta"] # Prior inizializzato come N(0,metà intervallo)
                    w_cov_theta = data_TS["w_cov_theta"]   # Prior inizializzato come N(0,metà intervallo)
                    w_mean_theta = np.array(w_mean_theta)
                    w_cov_theta  = np.array(w_cov_theta)

                    x_hist_num_wp = data_TS["x_hist_num_wp"] 
                    y_hist_num_wp = data_TS["y_hist_num_wp"] # lista di 1=success,0=failure
                    w_mean_num_wp = data_TS["w_mean_num_wp"] # Prior inizializzato come N(0,metà intervallo)
                    w_cov_num_wp = data_TS["w_cov_num_wp"]   # Prior inizializzato come N(0,metà intervallo)
                    w_mean_num_wp = np.array(w_mean_num_wp)
                    w_cov_num_wp  = np.array(w_cov_num_wp)

            except Exception as e:
                self.get_logger().error(f"Errore caricamento YAML: {e}")
                response.success = False
                return response
            
            try:
                if not os.path.exists(FILE_CURRENT_PLAN_PARAMS):
                    current_theta=90
                    current_num_wp=350
                else:
                    with open(FILE_CURRENT_PLAN_PARAMS, "r") as f:
                        data_current_plan_params = yaml.safe_load(f) 
                    current_theta = data_current_plan_params["current_theta"]
                    current_num_wp = data_current_plan_params["current_num_wp"]

            except Exception as e:
                self.get_logger().error(f"Errore caricamento YAML: {e}")
                response.success = False
                return response
            
            # Aggiornamento TS 
            if k % 2 == 0: # aggiorna theta
                # Append liste
                y_hist_theta.append(success_path) 
                x_hist_theta.append(current_theta)
                # update posterior
                w_mean_theta, w_cov_theta = update_w(np.array(x_hist_theta), np.array(y_hist_theta), w_mean_theta, w_cov_theta)
                # new sample
                num_wp = int(best_theta_bayes(w_mean_num_wp, w_cov_num_wp, x_min=300, x_max=400))
                x_next_num_wp = np.ones(PATH_NUM)*num_wp
                
                x_next_theta = sample_x_TS(w_mean_theta, w_cov_theta, x_min=80, x_max=100, M=PATH_NUM)
                
            else: # aggiorna num wp
                # Append liste
                y_hist_num_wp.append(success_path) 
                x_hist_num_wp.append(current_num_wp)
                # update posterior
                w_mean_num_wp, w_cov_num_wp = update_w(np.array(x_hist_num_wp), np.array(y_hist_num_wp), w_mean_num_wp, w_cov_num_wp)
                # new sample
                x_next_num_wp = sample_x_TS(w_mean_num_wp, w_cov_num_wp, x_min=300, x_max=400, M=PATH_NUM)
                theta = best_theta_bayes(w_mean_theta, w_cov_theta, x_min=80, x_max=100)
                x_next_theta = np.ones(PATH_NUM)*theta

            k+=1
            
            # Salva nuovi valori
            state_TS = {
                    "k": self.to_builtin(k),

                    "x_hist_theta": self.to_builtin(x_hist_theta),
                    "y_hist_theta": self.to_builtin(y_hist_theta),
                    "w_mean_theta": self.to_builtin(w_mean_theta),
                    "w_cov_theta":  self.to_builtin(w_cov_theta),

                    "x_hist_num_wp": self.to_builtin(x_hist_num_wp),
                    "y_hist_num_wp": self.to_builtin(y_hist_num_wp),
                    "w_mean_num_wp": self.to_builtin(w_mean_num_wp),
                    "w_cov_num_wp":  self.to_builtin(w_cov_num_wp),
                }
            with open(FILE_TS, "w") as f:
                yaml.safe_dump(state_TS, f, sort_keys=False)

            state_current_plan_params = {
                "current_theta": self.to_builtin(x_next_theta),
                "current_num_wp": self.to_builtin(x_next_num_wp),
            }
            with open(FILE_CURRENT_PLAN_PARAMS, "w") as f:
                yaml.safe_dump(state_current_plan_params, f, sort_keys=False)            

        self.get_logger().info(f"Belief set aggiornato: {len(updated)} modelli")
        response.success = True
        return response

def main(args=None):
    rclpy.init(args=args)
    node = BeliefUpdater()
    rclpy.spin(node)
    rclpy.shutdown()
