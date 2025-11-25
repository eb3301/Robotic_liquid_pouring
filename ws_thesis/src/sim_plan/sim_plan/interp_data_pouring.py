import re
import numpy as np
from scipy.interpolate import LinearNDInterpolator, Rbf

def load_tests(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    blocks = re.split(r'\n\s*-\s*id:', text)
    tests = []

    for block in blocks[1:]:
        lines = block.strip().split("\n")
        test = {}

        test_id = lines[0].strip()
        test["id"] = int(test_id)

        for ln in lines[1:]:
            key_val = ln.strip().split(":")
            key = key_val[0].strip()
            val = key_val[1].strip()
            try:
                val = float(val)
                if val.is_integer():
                    val = int(val)
            except:
                pass
            test[key] = val

        tests.append(test)

    return tests

def compute_k(tests):
    for t in tests:
        t["k"] = t["v_pour"] / t["v_model"]
    return tests

def build_linear_interpolator(tests):
    X = np.array([(t["theta_f"], t["v_init"]) for t in tests])
    k_vals = np.array([t["k"] for t in tests])
    return LinearNDInterpolator(X, k_vals)

def build_rbf_model(tests, function="multiquadric"):
    X = np.array([(t["theta_f"], t["v_init"]) for t in tests])
    xv = X[:, 0]
    yv = X[:, 1]
    kv = np.array([t["k"] for t in tests])

    rbf = Rbf(xv, yv, kv, function=function)
    return rbf

def get_k(theta_f, v_init, model="linear"):
    DIR = "/home/barutta/Robotic_liquid_pouring/ws_thesis/src/sim_plan/sim_plan/"
    path = DIR + "test_pouring.txt"
    tests = load_tests(path)
    tests = compute_k(tests)
    if model=="linear":
        f_lin = build_linear_interpolator(tests)
        k = f_lin(theta_f, v_init)
    else:
        f_rbf = build_rbf_model(tests)
        k = f_rbf(theta_f, v_init)
    return k

################################################

def main():
    DIR = "/home/barutta/Robotic_liquid_pouring/ws_thesis/src/sim_plan/sim_plan/"
    path = DIR + "test_pouring.txt"
    tests = load_tests(path)
    print(tests)
    tests = compute_k(tests)
    print(tests)

    theta=90
    v_init=40

    # Lineare:
    f_lin = build_linear_interpolator(tests)
    k_lin = f_lin(theta, v_init)
    print(k_lin)

    # RBF nonlineare:
    f_rbf = build_rbf_model(tests)
    k_rbf = f_rbf(theta, v_init)
    print(k_rbf)
    
if __name__ == '__main__':
    main()  