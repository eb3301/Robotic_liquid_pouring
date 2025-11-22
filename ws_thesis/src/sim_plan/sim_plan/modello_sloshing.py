import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import jv, jvp
from scipy.interpolate import interp1d
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper


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

def trajectory_to_arrays(trj):
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
    return (np.array(t),
            np.array(qs),
            np.array(qds),
            np.array(qdds))

def simulate_linear_sloshing(
        trj,     
        fluid_volume=0,
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
    H = 9.5 * 1e-2
    R = 3 * 1e-2
    g = 9.81
    rho = 998
    mu = 1e-3  
    dt = 0.01


    # Carica modello robot da urdf
    urdf_path = "/percorso/al/tuo_robot.urdf"
    robot = RobotWrapper.BuildFromURDF(urdf_path, root_joint=None)
    model = robot.model
    data = model.createData()

    frame_name = "tip"   # nome del link a cui è fissato il contenitore
    frame_id = model.getFrameId(frame_name)


    mf = fluid_volume * rho
    h = fluid_volume / (np.pi * R**2)

    
    name_to_idx = {name: i for i, name in enumerate(trj.joint_names)}
    time, Q, Qd, Qdd = trajectory_to_arrays(trj)
    t0 = time[0]
    tf = time[-1]
    time = [t-t0 for t in time]

    acc_container = []   # lista di accelerazioni [ax, ay, az] nel world

    for q, qd, qdd in zip(Q, Qd, Qdd):
        # Cinematica diretta + cin. differenziale
        pin.forwardKinematics(model, data, q, qd, qdd)
        pin.updateFramePlacement(model, data, frame_id)

        # This gives “classical” acceleration (lineare + angolare del frame)
        a_frame = pin.getFrameClassicalAcceleration(model, data, frame_id)

        lin_acc = a_frame.linear    # np.array([ax, ay, az])
        # ang_acc = a_frame.angular  # se ti serve

        acc_container.append(lin_acc)

    acc_container = np.vstack(acc_container)   # shape: (N, 3)
    a_x = acc_container[:, 0]
    a_y = acc_container[:, 1]

    # interpolazione lineare (di solito sufficiente per accelerazioni reali)
    a_interp_x = interp1d(time, a_x, kind='linear',
                        fill_value="extrapolate")
    a_interp_y = interp1d(time, a_y, kind='linear',
                        fill_value="extrapolate")

    # Preallocazione
    t_eval = np.linspace(0, tf-t0, 1000)
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
