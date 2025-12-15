import os
import yaml
from datetime import datetime

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_yaml(path):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return yaml.safe_load(f)

def save_yaml(path, data):
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)

def save_experiment_data(
    base_dir="/home/barutta/Robotic_liquid_pouring/ws_thesis/experiment_data",
    experiment_name=None,
    init_params=None,
    init_tolerances=None,
    iteration_id=None,
    iteration_parameters=None,
    iteration_scores=None,
    iteration_real_score=None,
    threshold=None,
    ts_file_data=None,
):
    """
    Salva i dati sperimentali in modo persistente.

    Struttura generata:
    /home/user/experiment_data/<experiment_name>/
        init_parameters.yaml
        tolerances_initial.yaml
        iterations_log.yaml        # append di tutte le iterazioni
        TS_state_latest.yaml       # versione aggiornata

    Parametri:
    - init_params: dict, parametri iniziali solo la prima volta
    - init_tolerances: dict, tolleranze iniziali solo la prima volta
    - iteration_id: numero dell'iterazione corrente
    - iteration_parameters: lista/dict parametri dell'iterazione
    - iteration_scores: lista punteggi associati
    - threshold: valore soglia per is_success()
    - ts_file_data: dict YAML del TS aggiornato
    """

    # Nome esperimento
    if experiment_name is None:
        experiment_name = datetime.now().strftime("exp_%Y%m%d_%H%M%S")

    exp_dir = os.path.join(base_dir, experiment_name)
    ensure_dir(exp_dir)

    # ---------- 1) Salva parametri iniziali (solo se non esistono) ----------
    init_params_path = os.path.join(exp_dir, "init_parameters.yaml")
    if init_params is not None and not os.path.exists(init_params_path):
        save_yaml(init_params_path, init_params)

    # ---------- 2) Salva tolleranze iniziali (solo se non esistono) ----------
    init_tol_path = os.path.join(exp_dir, "tolerances_initial.yaml")
    if init_tolerances is not None and not os.path.exists(init_tol_path):
        save_yaml(init_tol_path, init_tolerances)

    # ---------- 3) Append iterazione su iterations_log.yaml ----------
    iter_log_path = os.path.join(exp_dir, "iterations_log.yaml")

    iter_entry = {
        "iteration_id": iteration_id,
        "parameters": iteration_parameters,
        "scores": iteration_scores,
        "real result": iteration_real_score,
        "threshold": threshold
    }

    current_log = load_yaml(iter_log_path)
    if current_log is None:
        current_log = {"iterations": []}

    current_log["iterations"].append(iter_entry)
    save_yaml(iter_log_path, current_log)

    # ---------- 4) Salva copia aggiornata del TS ----------
    if ts_file_data is not None:
        ts_latest_path = os.path.join(exp_dir, "TS_state_latest.yaml")
        save_yaml(ts_latest_path, ts_file_data)

    return exp_dir
