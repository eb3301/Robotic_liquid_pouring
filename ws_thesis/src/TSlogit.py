import numpy as np
from scipy.special import expit
import yaml
import os

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

iter_file = "/tmp/iter.yaml"
if not os.path.exists(iter_file):
    k=0
else:
    with open(iter_file, "r") as f:
            k = yaml.safe_load(f)

if k % 2==0:
    file = "/tmp/TStheta.yaml"
    if not os.path.exists(file):
        x_hist = []
        current_x =  90
        y_hist = []
        w_mean = np.zeros(2)
        w_cov = np.eye(2)*10.0 
    else:
        with open(file, "r") as f:
            data = yaml.safe_load(f)
        x_hist = data["history"] or []
        current_x = data["new_x"] or 90
        y_hist = data["success"] or [] # lista di 1=success,0=failure
        w_mean = data["w_mean"] or np.zeros(2) # Prior inizializzato come N(0,metà intervallo)
        w_cov = data["w_cov"] or np.eye(2)*10.0 # Prior inizializzato come N(0,metà intervallo)
        w_mean = np.array(w_mean)
        w_cov  = np.array(w_cov)
else:
    file = "/tmp/TSnum_wp.yaml"
    if not os.path.exists(file):
        x_hist = []
        current_x =  350
        y_hist = []
        w_mean = np.zeros(2)
        w_cov = np.eye(2)*50.0 
    else:
        with open(file, "r") as f:
            data = yaml.safe_load(f)
        x_hist = data["history"] or []
        current_x = data["new_x"] or 350
        y_hist = data["success"] or [] # lista di 1=success,0=failure
        w_mean = data["w_mean"] or np.zeros(2) # Prior inizializzato come N(0,metà intervallo)
        w_cov = data["w_cov"] or np.eye(2)*50.0 # Prior inizializzato come N(0,metà intervallo)
        w_mean = np.array(w_mean)
        w_cov  = np.array(w_cov)

###############################
# dopo ogni simulaz.: (aggiorno una volta sola dopo che ogni traj è stata sim con tutti i parametri, serve nuovo success!!)
##############################

# update liste
y_hist.append(success) 
x_hist.append(float(current_x))

# update posterior
w_mean, w_cov = update_w(np.array(x_hist), np.array(y_hist), w_mean, w_cov)

# new sample
x_next = sample_x_TS(w_mean, w_cov, theta_min=0.5, theta_max=2.0)

# salva 
state = {
    "history": list(x_hist),
    "new_x": list(x_next),
    "success": list(y_hist),
    "w_mean":  w_mean.tolist(),
    "w_cov":   w_cov.tolist()     
}

with open(file, "w") as f:
    yaml.safe_dump(state, f)

