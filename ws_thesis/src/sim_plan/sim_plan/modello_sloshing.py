import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import jv, jvp
from scipy.interpolate import interp1d
import mujoco
import mujoco.viewer
import os
import yaml
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
# ============================================================
# 1) Funzioni utili
# ============================================================

# Radici ξ_0n della derivata della Bessel J0' (modalità radiali)
# Per comodità le includiamo fino ai primi 5 modi
# (valori tabulati in Bauer 1964)
XI_1N = {
    1: 1.84118,
    2: 5.33144,
    3: 8.53632,
    4: 11.7060,
    5: 14.8636
}

H = 9.5 * 1e-2
R = 3 * 1e-2
g = 9.81
Cd = 0.6
rho = 998
mu = 1e-3  
dt = 0.01

def natural_frequency(R, h, g, xi):
    """ω_n = sqrt( g*xi/R * tanh(xi*h/R) )"""
    return np.sqrt(g * xi / R * np.tanh(xi * h / R))

def damping_ratio(mu, rho, g, R, h):
    """Formula empirica eq. (2)"""
    return 0.92 * np.sqrt( (mu / rho) / np.sqrt(g * R**3) ) * ( 1 + 0.318 / np.sinh(1.84*h/R) * (1 + (1 - h/R) / np.cosh(1.84*h/R)) )

def sloshing_mass(mf, R, h, xi):
    """Eq. (4): m_n"""
    return 2 * mf * R / ( (xi * h) * (xi**2 - 1) ) * np.tanh(xi * h / R)

def ode_func(wn, damp, x0_ddot):
    """
    Restituisce una funzione f(t, y) per solve_ivp:
    y = [x_n, x_n_dot]
    """
    def f(t, y):
        x, xd = y
        xdd = -2*damp*wn*xd - wn**2*x - x0_ddot(t)
        return [xd, xdd]
    return f

def _trajectory_to_arrays(trj):
    t = []
    qs = []
    qds = []
    qdds = []
    for p in trj.points:
        tt = p.time_from_start.sec + p.time_from_start.nanosec * 1e-9
        t.append(tt)
        qs.append(p.positions)
        qds.append(p.velocities)
        qdds.append(p.accelerations)
    return (np.array(t), np.array(qs))

def trajectory_to_arrays(trj):
    t = np.asarray(trj["time"], dtype=float)
    qs = np.asarray(trj["positions"], dtype=float)[:, :6]

    return (t, qs)

def acc_from_q(time, q_trj, model, data, tool_body_id):
    """
    time: array shape (N,) con i timestamp assoluti
    q_trj: array shape (N, 6) con qpos
    """
    #with mujoco.viewer.launch_passive(model, data) as viewer:
    pos_list = []
    for q in q_trj:
        data.qpos[:6] = q
        mujoco.mj_forward(model, data)
        #viewer.sync()
        pos_list.append(np.array(data.xpos[tool_body_id], dtype=float))
        #input("Press Enter to continue...")

    pos = np.vstack(pos_list)   # shape (N, 3)
    N = len(pos)
 
    vel = np.zeros_like(pos)    # velocità cartesiana
    acc = np.zeros_like(pos)    # accelerazione cartesiana

    # Diff centrate per vel e acc (Estremi con differenze in avanti e indietro)
    for i in range(1, N-1):
        dt = time[i+1] - time[i-1]
        vel[i] = (pos[i+1] - pos[i-1]) / dt
 
    vel[0] = (pos[1] - pos[0]) / (time[1] - time[0])
    vel[-1] = (pos[-1] - pos[-2]) / (time[-1] - time[-2])

    for i in range(1, N-1):
        dt = time[i+1] - time[i-1]
        acc[i] = (vel[i+1] - vel[i-1]) / dt

    acc[0] = (vel[1] - vel[0]) / (time[1] - time[0])
    acc[-1] = (vel[-1] - vel[-2]) / (time[-1] - time[-2])

    ax=acc[:,0]
    ay=acc[:,1]
    az=acc[:,2]
    return ax, ay, az

def simulate_linear_sloshing(
        trj,     
        V_init=0,
        n_modes=1,
    ):
    """
    Simula il modello lineare di sloshing con n_modes.

    R: raggio contenitore [m]
    h: livello del liquido [m]
    rho: densità liquido [kg/m3]
    mu: viscosità dinamica [Pa*s]
    g: gravità [m/s2]
    x0_ddot: funzione accelerazione del contenitore: f(t)->a(t)
    t_eval: array dei tempi
    n_modes: numero di modi da includere
    fluid_volume: volume [m3] (se non dato: mf = pi*R^2*h*rho)
    """

    # Carica modello robot da urdf
    model_path = "/home/barutta/projects/src/ur5e_utils_mujoco/ur5e/ur5e.xml"
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    tool_body_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_frame")


    mf = V_init * rho
    h = V_init / (np.pi * R**2)
    
    # name_to_idx = {name: i for i, name in enumerate(trj.joint_names)}
    time, q_trj = trajectory_to_arrays(trj)
    t0 = time[0]
    tf = time[-1]
    time = [t-t0 for t in time]

    a_x, a_y, _ = acc_from_q(time, q_trj, model, data, tool_body_id)

    # interpolazione lineare (di solito sufficiente per accelerazioni reali)
    a_interp_x = interp1d(time, a_x, kind='linear',
                        fill_value="extrapolate")
    a_interp_y = interp1d(time, a_y, kind='linear',
                        fill_value="extrapolate")

    # Preallocazione
    t_eval = np.linspace(0, tf-t0, 100)
    x_modes = np.zeros((n_modes, len(t_eval)))
    y_modes = np.zeros((n_modes, len(t_eval)))

    # Per ogni modo n
    for n in range(1, n_modes + 1):

        xi = XI_1N[n]
        wn = natural_frequency(R, h, g, xi)
        zeta = damping_ratio(mu, rho, g, R, h)
        m_n = sloshing_mass(mf, R, h, xi)

        ode_x = ode_func(wn, zeta, a_interp_x)
        sol_x = solve_ivp(ode_x, [t_eval[0], t_eval[-1]], [0, 0], t_eval=t_eval, rtol=1e-8, atol=1e-8)
        x_modes[n-1, :] = sol_x.y[0]

        ode_y = ode_func(wn, zeta, a_interp_y)
        sol_y = solve_ivp(ode_y, [t_eval[0], t_eval[-1]], [0, 0], t_eval=t_eval, rtol=1e-8, atol=1e-8)
        y_modes[n-1, :] = sol_y.y[0]

    # ============================================================
    # 4) Sloshing height η(t) usando eq. (10)
    #    η = 8 ∑ x_n / ( xi * (xi^2 -1)) * tanh(xi*h/R)
    # ============================================================

    eta_x = np.zeros(len(t_eval))
    eta_y = np.zeros(len(t_eval))
    for n in range(1, n_modes + 1):
        xi = XI_1N[n]
        coef = 8 / (xi * ((xi**2 - 1))) * np.tanh(xi*h/R)
        eta_x += coef * x_modes[n-1, :]
        eta_y += coef * y_modes[n-1, :]

    return eta_x, x_modes, eta_y, y_modes, t_eval

def reward_sloshing(trj: dict, Vol_init: float, n_modes:int=5, view:bool=False):

    print("Simulating sloshing")
    V_init = Vol_init * 1e-6 if Vol_init>1 else Vol_init
    V = V_init
    V_spilled = 0

    eta_x, _, eta_y, _, t_eval = simulate_linear_sloshing(trj, V_init=V_init, n_modes=n_modes)
    h = V_init / (np.pi * R**2)
    
    if view:
        plt.plot(t_eval, eta_x+h, label="eta_x")
        plt.plot(t_eval, eta_y+h, label="eta_y")
        plt.plot(t_eval, np.ones_like(t_eval)*H)
        plt.legend()
        plt.xlabel("tempo")
        plt.ylabel("ampiezza")
        plt.show()

    for i in range(len(eta_x)):
        eta = max(eta_x[i], eta_y[i])

        if eta + h > H:
            dh = eta + h - H
            dh = np.clip(dh,0,2*R)
            L = 2 * np.arccos((R-dh)/R) * R 
    
            # Update volumes:
            Q = 2/3 * Cd * np.sqrt(2*g) * L * dh**1.5  # 1.25 to account for not rect sect (exp param)
            V_i = min(Q * dt, V)
            V = np.clip(V - V_i, 0, V_init)
            V_spilled = np.clip(V_spilled + V_i, 0, V_init)
    
    print(f"Simulation completed")
    print(f"[Vol_init: {V_init*1e6}]-[Vol_spilled: {V_spilled*1e6}]")
    reward = np.clip(1 - V_spilled / V_init, 0, 1) 
    return reward

def main():

    PARAMS_FILE = "/tmp/best_path.yaml"

    if not os.path.exists(PARAMS_FILE):
        print(f"File doesn't exist")
    else:
        with open(PARAMS_FILE, "r") as f:
            data = yaml.safe_load(f)

    dt = 0.01
    transp_mot = data["best_path"]["transport"]
    time_mot = np.arange(0.0, len(transp_mot)*dt, dt)
    trj = {
        "positions": transp_mot,
        "time": time_mot,
    }
    

    Vol_init = 90
    reward = reward_sloshing(trj, Vol_init, n_modes=1)
    print(reward)

if __name__ == '__main__':
    main()  