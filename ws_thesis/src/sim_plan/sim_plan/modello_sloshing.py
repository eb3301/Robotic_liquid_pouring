import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import jv, jvp
from scipy.interpolate import interp1d


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


# ============================================================
# 2) ODE lineare del modello (eq. (3))
# ============================================================

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


# ============================================================
# 3) Funzione principale di simulazione
# ============================================================

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

    mf = fluid_volume * rho
    h = fluid_volume / (np.pi * R**2)

    name_to_idx  = {n:i for i,n in enumerate(trj.joint_names)}
    
    q_pos=[]
    q_acc=[]
    time=[]
    for p in trj.points:
        time_from_start = p.time_from_start
        t = time_from_start.sec + time_from_start.nanosec * 1e-9
        q_pos.append(p.positions)
        q_acc.append(p.accelerations)
        time.append(t)

    t0 = time[0]
    tf = time[-1]
    time = [t-t0 for t in time]
    time = np.asarray(time, dtype=float)
    q_pos = np.asarray(q_pos, dtype=float)
    q_acc = np.asarray(q_acc, dtype=float)
    
    a_x=[]
    a_y=[]

    # interpolazione lineare (di solito sufficiente per accelerazioni reali)
    a_interp_x = interp1d(time, a_x, kind='linear',
                        fill_value="extrapolate")
    a_interp_y = interp1d(time, a_y, kind='linear',
                        fill_value="extrapolate")

    # Preallocazione
    t_eval = np.linspace(t0, tf, 1000)
    x_modes = np.zeros((n_modes, len(t_eval)))
    y_modes = np.zeros((n_modes, len(t_eval)))

    # DA SISTEMARE -->
    # Per ogni modo n
    for n in range(1, n_modes + 1):

        xi = XI_1N[n]
        wn = natural_frequency(R, h, g, xi)
        zeta = damping_ratio(mu, rho, g, R, h)
        m_n = sloshing_mass(mf, R, h, xi)

        ode_x = ode_func(wn, zeta, a_interp)
        sol_x = solve_ivp(ode, [t_eval[0], t_eval[-1]], [0, 0], t_eval=t_eval, rtol=1e-8, atol=1e-8)
        x_modes[n-1, :] = sol.y[0]

    # ============================================================
    # 4) Sloshing height η(t) usando eq. (10)
    #    η = 8 ∑ x_n / ( xi * (xi^2 -1)) * tanh(xi*h/R)
    # ============================================================

    eta = np.zeros(len(t_eval))
    for n in range(1, n_modes + 1):
        xi = XI_1N[n]
        coef = 8 / (xi * ((xi**2 - 1))) * np.tanh(xi*h/R)
        eta += coef * x_modes[n-1, :]

    return eta, x_modes, t_eval
