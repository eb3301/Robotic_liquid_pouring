import rclpy
from rclpy.node import Node
import yaml
import numpy as np

from interfaces.srv import Simplan
from sim_plan.main import init_sim, generate_sim, plan_path, generate_parameters, simulate_action

class PathPlannerService(Node):
    def __init__(self):
        super().__init__('path_planner_service')
        self.srv = self.create_service(Simplan, 'plan_path', self.plan_path_callback)
        self.get_logger().info("Path planner service ready")

    def _make_parameters_range(self, values: dict, tolerances: dict) -> dict:
        """
        Crea un dizionario con i range a partire da valori e tolleranze.

        :param values: dict con valori nominali
        :param tolerances: dict con tolleranze (tol_minus, tol_plus)
                        - se singolo valore -> tol_minus = tol_plus
                        - se lista -> stessa struttura del valore
        :return: dict con ranges
        """
        result = {}

        for key, val in values.items():
            if key not in tolerances:
                continue  # ignora se non definita tolleranza

            tol = tolerances[key]

            # Caso tol rel
            if isinstance(val, (int, float)) and isinstance(tol, tuple) and len(tol) == 3 and tol[0] == "rel":
                minus_frac, plus_frac = tol[1], tol[2]
                result[key] = [val * (1 - minus_frac), val * (1 + plus_frac)]
                continue

            # Caso scalare (float/int)
            if isinstance(val, (int, float)):
                if tol is None:
                    result[key] = [val, val]
                else:
                    tmin, tmax = tol if isinstance(tol, tuple) else (tol, tol)
                    result[key] = [val - tmin, val + tmax]

            # Caso lista/vettore
            elif isinstance(val, (list, tuple)):
                ranges = []
                for i, v in enumerate(val):
                    if tol is None:
                        ranges.append([v, v])
                    elif isinstance(tol, list):  
                        # lista di tuple per ogni componente
                        tmin, tmax = tol[i]
                        ranges.append([v - tmin, v + tmax])
                    else:
                        # stessa tolleranza per tutti
                        tmin, tmax = tol if isinstance(tol, tuple) else (tol, tol)
                        ranges.append([v - tmin, v + tmax])
                result[key] = ranges

            else:
                raise TypeError(f"Tipo non supportato per {key}: {type(val)}")

        return result

    def plan_path_callback(self, request, response):

        N = 20                    # Numero di modelli simulati (iniziale)
        M = 5                     # Numero di traiettorie
        delta = 0.7            # Threshold di successo
        view=True
        liq=True
        record=False
        debug=False

        # Deve andare solo la prima volta la generazione del range, dopodiché check esistenza paraeters_range.yaml e uso quello
        req_parameters = {
            "pos_init_cont": list(request.pos_init_cont),
            "pos_init_ee": list(request.pos_init_ee),
            "pos_cont_goal": list(request.pos_cont_goal),
            "offset": list(request.offset),
            "dCoR": [0.0, -0.01, 0.04],
            "vol_init": request.init_vol, #2e-5, +-MAE
            "densità": 998.0,
            "viscosità": 0.001,
            "tens_sup": 0.072,
            "vol_target": request.target_vol, #0.75e-5,
            "err_target": 5e-6,
            "theta_f": request.theta_f, #+-15°
            "num_wp": request.num_wp,
        }
        tolerances = {
            "pos_init_cont": [
                (0.015, 0.015),  # x: ±1.5 cm
                (0.015, 0.015),  # y: ±1.5 cm
                (0.01, 0.01),    # z: ±1.0 cm
            ],
            "pos_init_ee": [
                (0.005, 0.005),  # x: ±5 mm
                (0.005, 0.005),  # y: ±5 mm
                (0.005, 0.005),  # z: ±5 mm
                (0.00, 0.00),    # w: fisso
                (0.00, 0.00),    # x: fisso
                (0.00, 0.00),    # y: fisso
                (0.00, 0.00),    # z: fisso
            ],
            "pos_cont_goal": [
                (0.015, 0.015),  # x: ±1.5 cm
                (0.015, 0.015),  # y: ±1.5 cm
                (0.01, 0.01),    # z: ±1.0 cm
            ],
            "offset": [
                (0.00, 0.00),  # primo offset bloccato (le pinze riportano al centro quando chiuse)
                (0.01, 0.01),  # secondo: ±1 cm
                (0.01, 0.01),  # terzo: -1 cm / +0 cm (per restare entro [0.12, 0.13])
            ],
            "dCoR": [
                (0.001, 0.001),  # componente 1: ±1 mm
                (0.01, 0.01),    # componente 2: ±1 cm
                (0.01, 0.01),    # componente 3: ±1 cm
            ],
            "viscosità": (0.00025, 0.00025),  # ±25% intorno a 0.001 Pa·s
            "densità": (3.0, 3.0),          # ±3 kg/m^3
            "tens_sup": (0.002, 0.001),     # -0.002 / +0.001 N/m (gamme tipiche 0.070–0.073)
            "vol_init": ( 1.5e-5, 1.5e-5),  # ±1e-5 m^3 (15ml)
            "vol_target": (0.0, 0.0),       # no tol, è scelta
            "err_target": (0.0, 0.0),       # vincolo rigido
            "theta_f": (15.0, 15.0),        # ±15°
            "num_wp": ("rel", 0.5, 0.5),    # ±50%
        }
        parameters_range=self._make_parameters_range(req_parameters,tolerances)

        parameters_set=[]
        for _ in range(N):
            parameters_set.append(generate_parameters(parameters_range)) 

        # Da qui inizia parte ciclica codice:
        candidate_paths = []

        for i in range(len(parameters_set)):
            parameters = parameters_set[i] # ottiene l'n-esimo dizionario di parametri
            scene, ur5e, becher, becher2, liquid, dt = generate_sim(parameters,view,liq,debug,record) # genera l'ambiente di simulazione
            
            for j in range(M):
                theta_f =  parameters["theta_f"] #np.pi * 0.48
                num_wp = parameters["num_wp"] #int(10/dt)

                paths = plan_path(
                    ur5e, 
                    theta_f,
                    parameters,
                    timeout=5.0, 
                    smooth_path=True, 
                    num_waypoints=num_wp, 
                    ignore_collision=False, 
                    planner= "RRTStar", # "RRT", "RRTConnect", "RRTstar", "InformedRRTStar"
                    debug=debug,
                )
                candidate_paths.append(paths)
                    

        # Valuta ogni traiettoria su ogni set di param
        best_path = None
        best_score = -1e30
        best_parameters = None
        score_best_path=[]
        
        if liq:
            for paths in candidate_paths:
                total_score = 0
                local_best_score = -1e30
                local_best_parameters = None
                local_scores = []
                
                for parameters in parameters_set:
                    score = simulate_action(ur5e, parameters, paths, scene, becher, becher2, liquid, liq)
                    total_score += score
                    local_scores.append((parameters, score))
                    if score > local_best_score:
                        local_best_score = score
                        local_best_parameters = parameters

                if total_score > best_score:
                    best_score = total_score
                    best_path = paths
                    best_parameters = local_best_parameters
                    score_best_path = local_scores
            
            best_score/=N
            if best_score < delta:
                self.get_logger().info("Nessuna traiettoria soddisfa il delta succ")
                response.success=False
                return response
            else:
                print("Esiste traj che soddisfa req succ")

        if best_parameters is None or best_path is None: 
            self.get_logger().info("Nessuna traiettoria o no best params")
            response.success=False
            return response

        n_points = len(best_path)
        time = np.linspace(0, (n_points - 1) * dt, n_points)

        try:
            with open("/tmp/best_path.yaml", "w") as f:
                yaml.safe_dump({"best_path": best_path}, f, sort_keys=False)
            with open("/tmp/parameters.yaml", "w") as f:
                yaml.safe_dump({"parameters": best_parameters}, f, sort_keys=False)
            with open("/tmp/tolerances.yaml", "w") as f:
                yaml.safe_dump({"tolerances": tolerances}, f, sort_keys=False)
            with open("/tmp/score_best_path.yaml", "w") as f:
                yaml.safe_dump({"score_best_path": score_best_path}, f, sort_keys=False)
        except Exception as e:
            self.get_logger().error(f"Errore salvataggio YAML: {e}")
            response.success = False
            return response

        best_path = best_path.tolist()
        response.best_path = best_path
        response.time = time
        return response
        
class MinimalPathPlannerService(Node):
    def __init__(self):
        super().__init__('path_planner_service')
        self.srv = self.create_service(Simplan, 'plan_path', self.plan_path_callback)
        self.get_logger().info("Service 'plan_path' ready")

    def plan_path_callback(self, request, response):
        # questi sono valori medi, bisogna trasformarli in range di params
        parameters = {
            "pos_init_cont": list(request.pos_init_cont),
            "pos_init_ee": list(request.pos_init_ee),
            "pos_cont_goal": list(request.pos_cont_goal),
            "offset": list(request.offset),
            "dCoR": [0.0, -0.01, 0.04],
            "vol_init": request.init_vol, #2e-5, +-MAE
            "densità": 998.0,
            "viscosità": 0.001,
            "tens_sup": 0.072,
            "vol_target": request.target_vol, #0.75e-5,
            "err_target": 5e-6,
            "theta_f": request.theta_f, #+-15°
            "num_wp": request.num_wp,
        }

        #ripristinare logica iterazioni su param e traj
        init_sim()
        scene, ur5e, becher, becher2, liquid, dt = generate_sim(parameters, view=False, liq=False, debug=False)

        theta_f =  parameters["theta_f"] #np.pi * 0.48
        num_wp = parameters["num_wp"] #int(10/dt)

        paths = plan_path(
            ur5e,
            theta_f,
            parameters,
            timeout=5.0,
            smooth_path=True,
            num_waypoints=num_wp,
            ignore_collision=False,
            planner="RRTstar",
        )

        
        
        best_path = paths["all"].tolist()
        n_points = len(best_path)
        time = np.linspace(0, (n_points - 1) * dt, n_points)

        # salva in yaml
        with open("/tmp/best_path.yaml", "w") as f:
            yaml.dump({"best_path": best_path}, f)
        
        # salva anche i parametri del best path in un file o passali via ros2

        response.best_path = best_path
        response.time = time
        return response

def main(args=None):
    rclpy.init(args=args)
    node = PathPlannerService()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
