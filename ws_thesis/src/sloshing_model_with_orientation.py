
import numpy as np
import pandas as pd

GRAVITY = 9.81

def _finite_diff_second_derivative(t, x):
    t = np.asarray(t)
    x = np.asarray(x)
    vx = np.gradient(x, t, edge_order=2)
    ax = np.gradient(vx, t, edge_order=2)
    return ax

def _sanitize_inputs(R, H, V0):
    if R <= 0 or H <= 0:
        raise ValueError("R e H devono essere > 0.")
    if V0 < 0:
        raise ValueError("V0 deve essere >= 0.")

def _quat_to_R(qw, qx, qy, qz):
    """
    Restituisce la matrice di rotazione 3x3 R(q) che ruota vettori
    dal frame 'body' al frame 'world' (convenzione NASA scalar-first).
    Per trasformare un vettore del mondo nel body, usa R.T @ v.
    """
    # normalizza
    norm = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    qw, qx, qy, qz = qw/norm, qx/norm, qy/norm, qz/norm
    # elementi
    ww, xx, yy, zz = qw*qw, qx*qx, qy*qy, qz*qz
    wx, wy, wz = qw*qx, qw*qy, qw*qz
    xy, xz, yz = qx*qy, qx*qz, qy*qz
    R = np.array([
        [ww + xx - yy - zz,      2*(xy - wz),        2*(xz + wy)],
        [2*(xy + wz),            ww - xx + yy - zz,  2*(yz - wx)],
        [2*(xz - wy),            2*(yz + wx),        ww - xx - yy + zz]
    ])
    return R

def _angle_from_accel_upright(ax, ay, g=GRAVITY):
    ah = np.sqrt(ax*ax + ay*ay)
    theta = np.arctan2(ah, g)
    return theta, ah

def _angle_from_g_eff_body(g_eff_body):
    """
    g_eff_body: array shape (N,3) nel frame del contenitore.
    Angolo della superficie libera rispetto all'orizzontale del contenitore:
    theta = arctan2( sqrt(gx^2+gy^2), |gz| )
    """
    gx, gy, gz = g_eff_body[:,0], g_eff_body[:,1], g_eff_body[:,2]
    gh = np.sqrt(gx*gx + gy*gy)
    theta = np.arctan2(gh, np.abs(gz))
    return theta, gh

def _instant_overfill_volume(R, H, V, tan_theta, c):
    if tan_theta <= 0:
        return 0.0
    hbar = V / (np.pi * R**2)
    hmax = hbar + R * tan_theta
    if hmax <= H:
        return 0.0
    xc = c / tan_theta
    xc = np.clip(xc, -R, R)
    def I1(x):
        return 0.5*x*np.sqrt(R**2 - x**2) + 0.5*R**2*np.arcsin(x/R)
    term1 = (tan_theta/3.0) * (R**2 - xc**2)**1.5
    term2 = c * ( (np.pi*R**2)/4.0 - I1(xc) )
    V_over = 2.0*(term1 - term2)
    return max(float(V_over), 0.0)

def _weir_outflow(R, H, V, tan_theta, c, Cd=0.6, g=GRAVITY):
    if tan_theta <= 0:
        return 0.0
    hbar = V / (np.pi * R**2)
    hmax = hbar + R * tan_theta
    if hmax <= H:
        return 0.0
    xc = c / tan_theta
    xc = np.clip(xc, -R, R)
    phi_c = np.arccos(xc / R)
    L = 2.0 * R * phi_c
    h_head = 0.5 * (R * tan_theta - c)
    if h_head <= 0:
        return 0.0
    Q = (2.0/3.0) * Cd * L * np.sqrt(2.0*g) * (h_head**1.5)
    return float(Q)

def simulate_sloshing(
    t, x, y, z,
    R, H, V0,
    method="instant", Cd=0.6, g=GRAVITY,
    quats=None
):
    """
    Simula lo sloshing in un cilindro con possibilità di orientamento variabile.
    Se 'quats' è None: contenitore sempre verticale.
    Se 'quats' è Nx4 [qw,qx,qy,qz]: orientamento body->world a ogni istante.
    In tal caso, la gravità efficace nel frame body è: g_eff_body = R(q)^T @ [(0,0,-g) - a_world].

    Ritorna dict con: t, theta, tan_theta, V, V_spilled, ah (o gh), ax,ay,az.
    """
    _sanitize_inputs(R, H, V0)

    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    if t.ndim != 1 or any(arr.shape != t.shape for arr in (x,y,z)):
        raise ValueError("t, x, y, z devono avere stessa lunghezza 1D.")

    ax = _finite_diff_second_derivative(t, x)
    ay = _finite_diff_second_derivative(t, y)
    az = _finite_diff_second_derivative(t, z)

    # gravità efficace nel mondo
    g_world = np.array([0.0, 0.0, -g])
    a_world = np.stack([ax, ay, az], axis=1)
    g_eff_world = g_world - a_world  # N x 3

    if quats is None:
        # caso 'upright'
        theta, gh = _angle_from_accel_upright(ax, ay, g=g)
    else:
        quats = np.asarray(quats, dtype=float)
        if quats.shape != (len(t), 4):
            raise ValueError("quats deve avere shape (N,4) con colonne [qw,qx,qy,qz].")
        # ruota g_eff_world nel body
        g_eff_body = np.empty_like(g_eff_world)
        for i in range(len(t)):
            qw, qx, qy, qz = quats[i]
            R_bw = _quat_to_R(qw, qx, qy, qz)  # body->world
            g_eff_body[i] = R_bw.T @ g_eff_world[i]  # world->body
        theta, gh = _angle_from_g_eff_body(g_eff_body)

    tan_theta = np.tan(theta)

    V = np.empty_like(t, dtype=float)
    V_spilled = np.empty_like(t, dtype=float)
    V_curr = float(V0)
    V[0] = V_curr
    V_spilled[0] = 0.0

    for i in range(1, len(t)):
        dt = t[i] - t[i-1]
        if dt <= 0:
            raise ValueError("t deve essere strettamente crescente.")
        hbar = V_curr / (np.pi * R**2)
        c = H - hbar
        if method == "instant":
            V_over = _instant_overfill_volume(R, H, V_curr, tan_theta[i], c)
            V_curr = max(V_curr - V_over, 0.0)
        elif method == "weir":
            Q = _weir_outflow(R, H, V_curr, tan_theta[i], c, Cd=Cd, g=g)
            V_loss = Q * dt
            V_curr = max(V_curr - V_loss, 0.0)
        else:
            raise ValueError("method deve essere 'instant' oppure 'weir'.")
        V[i] = V_curr
        V_spilled[i] = V0 - V_curr

    # output
    out = {
        "t": t,
        "theta": theta,           # [rad]
        "tan_theta": tan_theta,
        "V": V,
        "V_spilled": V_spilled,
        "ah": gh,                 # modulo componente orizzontale di g_eff (ah o gh)
        "ax": ax, "ay": ay, "az": az,
    }
    if quats is not None:
        out["g_eff_world"] = g_eff_world
    return out

def load_path_csv(path, t_col="t", x_col="x", y_col="y", z_col="z",
                  quat_cols=("qw","qx","qy","qz")):
    """
    Carica un CSV con colonne t,x,y,z e opzionalmente qw,qx,qy,qz.
    Ritorna t,x,y,z e quats (oppure None se non presenti).
    """
    df = pd.read_csv(path)
    for c in (t_col, x_col, y_col, z_col):
        if c not in df.columns:
            raise ValueError(f"Colonna '{c}' mancante nel CSV.")
    t = df[t_col].to_numpy()
    x = df[x_col].to_numpy()
    y = df[y_col].to_numpy()
    z = df[z_col].to_numpy()
    if all(c in df.columns for c in quat_cols):
        quats = df[list(quat_cols)].to_numpy()
    else:
        quats = None
    return t, x, y, z, quats

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Stima spillato per sloshing in cilindro con orientamento variabile da traiettoria e quaternioni.")
    p.add_argument("--csv", required=True, help="Percorso CSV con colonne t,x,y,z e opzionalmente qw,qx,qy,qz")
    p.add_argument("--R", type=float, required=True, help="Raggio del cilindro [m]")
    p.add_argument("--H", type=float, required=True, help="Altezza utile [m]")
    p.add_argument("--V0", type=float, required=True, help="Volume iniziale [m^3]")
    p.add_argument("--method", choices=["instant","weir"], default="weir", help="Modello di spill: instant oppure weir")
    p.add_argument("--Cd", type=float, default=0.6, help="Coeff. di deflusso per 'weir'")
    args = p.parse_args()

    t,x,y,z,quats = load_path_csv(args.csv)
    res = simulate_sloshing(t,x,y,z, R=args.R, H=args.H, V0=args.V0, method=args.method, Cd=args.Cd, quats=quats)
    total_spilled = float(res["V_spilled"][-1])
    print(f"Spillato totale [m^3]: {total_spilled:.6f}")
    out = pd.DataFrame({
        "t": res["t"],
        "theta_rad": res["theta"],
        "tan_theta": res["tan_theta"],
        "V_remaining_m3": res["V"],
        "V_spilled_m3": res["V_spilled"],
        "a_h_like_mps2": res["ah"],
        "ax": res["ax"],
        "ay": res["ay"],
        "az": res["az"],
    })
    out_path = "sloshing_results.csv"
    out.to_csv(out_path, index=False)
    print(f"Serie temporali salvate in {out_path}")
