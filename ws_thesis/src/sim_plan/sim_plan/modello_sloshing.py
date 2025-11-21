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

def make_ode_linear(wn, damp, x0_ddot):
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

    # DA SISTEMARE -->
    acc = trj # da sost con calcolo acc
    t_eval = np.linspace(0,dt*len(acc),dt)

    

    # interpolazione lineare (di solito sufficiente per accelerazioni reali)
    a_interp = interp1d(t_data, a_data, kind='linear',
                        fill_value="extrapolate")


    # Preallocazione
    x_modes = np.zeros((n_modes, len(t_eval)))

    # Per ogni modo n
    for n in range(1, n_modes + 1):

        xi = XI_1N[n]
        wn = natural_frequency(R, h, g, xi)
        zeta = damping_ratio(mu, rho, g, R, h)
        m_n = sloshing_mass(mf, R, h, xi)

        ode = make_ode_linear(wn, zeta, x0_ddot)
        sol = solve_ivp(ode, [t_eval[0], t_eval[-1]], [0, 0], t_eval=t_eval, rtol=1e-8, atol=1e-8)
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
