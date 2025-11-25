import re
import numpy as np
from scipy.interpolate import LinearNDInterpolator, Rbf
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

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
    X = np.array([(t["v_init"], t["v_pour"]) for t in tests])
    k_vals = np.array([t["k"] for t in tests])
    return LinearNDInterpolator(X, k_vals)


def build_rbf_model(tests, function="multiquadric"):
    X = np.array([(t["v_init"], t["v_pour"]) for t in tests])
    xv = X[:, 0]
    yv = X[:, 1]
    kv = np.array([t["k"] for t in tests])

    rbf = Rbf(xv, yv, kv, function=function)
    return rbf


def build_polynomial_model(tests, degree=3):
    X = np.array([(t["v_init"], t["v_pour"]) for t in tests])
    kv = np.array([t["k"] for t in tests])

    poly = PolynomialFeatures(degree=degree, include_bias=True)
    Xp = poly.fit_transform(X)

    model = LinearRegression()
    model.fit(Xp, kv)

    def f(v_init, v_pour):
        inp = np.array([[v_init, v_pour]])
        inp_p = poly.transform(inp)
        return model.predict(inp_p)[0]

    return f

def main():
    tests = load_tests("tests.txt")
    tests = compute_k(tests)

    # Lineare:
    f_lin = build_linear_interpolator(tests)
    k_lin = f_lin(100, 40)

    # RBF nonlineare:
    f_rbf = build_rbf_model(tests)
    k_rbf = f_rbf(100, 40)

    # Polinomiale 3° ordine:
    f_poly = build_polynomial_model(tests, degree=3)
    k_poly = f_poly(100, 40)
