
import numpy as np
import pandas as pd

GRAVITY = 9.81

def _finite_diff_second_derivative(t, x):
    """
    Second derivative of x(t) with numpy.gradient. Assumes t is strictly increasing.
    Returns array same shape as x.
    """
    t = np.asarray(t)
    x = np.asarray(x)
    # First derivative
    vx = np.gradient(x, t, edge_order=2)
    # Second derivative
    ax = np.gradient(vx, t, edge_order=2)
    return ax

def _angle_from_accel(ax, ay, g=GRAVITY):
    """
    Ritorna angolo di inclinazione (rad) della superficie libera rispetto all'orizzontale.
    theta = arctan( a_h / g ), dove a_h = sqrt(ax^2 + ay^2).
    """
    ah = np.sqrt(ax*ax + ay*ay)
    theta = np.arctan2(ah, g)
    return theta, ah

def _instant_overfill_volume(R, H, V, tan_theta, c):
    """
    Volume 'istantaneo' che eccede il bordo superiore in un cilindro raggio R, altezza H.
    Parametri:
        V          : volume attuale nel contenitore
        tan_theta  : tan(theta) nel passo temporale
        c          : H - hbar  (quanto manca al bordo)
    Ritorna V_over >= 0.
    """
    if tan_theta <= 0:
        return 0.0
    # Se la cresta non supera H, zero
    hbar = V / (np.pi * R**2)
    hmax = hbar + R * tan_theta
    if hmax <= H:
        return 0.0
    # Punto di taglio xc lungo la direzione di massima pendenza
    xc = c / tan_theta  # può essere negativo se già oltre
    # Limita xc all'interno del disco
    xc = np.clip(xc, -R, R)
    # Primitive utili
    def I1(x):
        return 0.5*x*np.sqrt(R**2 - x**2) + 0.5*R**2*np.arcsin(x/R)
    # Volumetto eccedente (forma chiusa)
    term1 = (tan_theta/3.0) * (R**2 - xc**2)**1.5
    term2 = c * ( (np.pi*R**2)/4.0 - I1(xc) )
    V_over = 2.0*(term1 - term2)
    return max(float(V_over), 0.0)

def _weir_outflow(R, H, V, tan_theta, c, Cd=0.6, g=GRAVITY):
    """
    Portata di sfioro sul bordo tipo soglia affilata (stima).
    Ritorna Q [m^3/s].
    """
    if tan_theta <= 0:
        return 0.0
    hbar = V / (np.pi * R**2)
    hmax = hbar + R * tan_theta
    if hmax <= H:
        return 0.0
    # regione che supera il bordo
    xc = c / tan_theta
    xc = np.clip(xc, -R, R)
    # lunghezza di bordo superata L = 2R*phi_c con phi_c = arccos(xc/R)
    phi_c = np.arccos(xc / R)
    L = 2.0 * R * phi_c
    # dislivello medio sopra il bordo
    h_head = 0.5 * (R * tan_theta - c)
    if h_head <= 0:
        return 0.0
    Q = (2.0/3.0) * Cd * L * np.sqrt(2.0*g) * (h_head**1.5)
    return float(Q)

def simulate_sloshing(
    t, x, y, z,
    R, H, V0,
    method="instant", Cd=0.6, g=GRAVITY
):
    """
    Simula lo sloshing in un contenitore cilindrico verticale soggetto a traslazioni.
    Ipotesi: contenitore sempre 'upright' (senza roll/pitch/yaw), liquido incomprimibile,
    superficie libera piana allineata a g_eff.

    Parametri
    ---------
    t : array-like [s]            tempi
    x, y, z : array-like [m]      posizioni cartesiane del baricentro del contenitore
    R : float [m]                 raggio del cilindro
    H : float [m]                 altezza interna utile
    V0 : float [m^3]              volume iniziale di liquido
    method : {"instant","weir"}   algoritmo di spill: volume oltre-bordo istantaneo oppure portata di sfioro
    Cd : float                    coefficiente di deflusso per "weir"
    g : float [m/s^2]             gravità

    Ritorna
    -------
    results : dict con chiavi
        't'           : array tempi
        'theta'       : array angolo della superficie libera [rad]
        'tan_theta'   : array tan(theta)
        'V'           : array volume rimanente [m^3]
        'V_spilled'   : array volume spillato cumulato [m^3]
        'ah'          : array modulo accelerazione orizzontale [m/s^2]
    """

    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    if t.ndim != 1 or any(arr.shape != t.shape for arr in (x,y,z)):
        raise ValueError("t, x, y, z devono avere stessa lunghezza 1D.")

    # accelerazioni traslazionali
    ax = _finite_diff_second_derivative(t, x)
    ay = _finite_diff_second_derivative(t, y)
    az = _finite_diff_second_derivative(t, z)

    # angolo superficie libera e modulo accel orizz.
    theta, ah = _angle_from_accel(ax, ay, g=g)
    tan_theta = np.tan(theta)

    V = np.empty_like(t, dtype=float)
    V_spilled = np.empty_like(t, dtype=float)
    V_curr = float(V0)
    V[0] = V_curr
    V_spilled[0] = 0.0

    # integrazione temporale
    for i in range(1, len(t)):
        dt = t[i] - t[i-1]
        if dt <= 0:
            raise ValueError("t deve essere strettamente crescente.")

        # quota media del liquido rispetto al bordo
        hbar = V_curr / (np.pi * R**2)
        c = H - hbar

        if method == "instant":
            V_over = _instant_overfill_volume(R, H, V_curr, tan_theta[i], c)
            # Rimuovi il volume eccedente in questo passo
            V_curr = max(V_curr - V_over, 0.0)
        elif method == "weir":
            Q = _weir_outflow(R, H, V_curr, tan_theta[i], c, Cd=Cd, g=g)
            V_loss = Q * dt
            V_curr = max(V_curr - V_loss, 0.0)
        else:
            raise ValueError("method deve essere 'instant' oppure 'weir'.")

        V[i] = V_curr
        V_spilled[i] = V0 - V_curr

    return {
        "t": t,
        "theta": theta,
        "tan_theta": tan_theta,
        "V": V,
        "V_spilled": V_spilled,
        "ah": ah,
        "ax": ax, "ay": ay, "az": az,
    }

def load_path_csv(path, t_col="t", x_col="x", y_col="y", z_col="z"):
    """
    Carica un CSV con colonne t, x, y, z. Restituisce t,x,y,z come ndarray.
    Le unità devono essere t in secondi e posizioni in metri.
    """
    df = pd.read_csv(path)
    for c in (t_col, x_col, y_col, z_col):
        if c not in df.columns:
            raise ValueError(f"Colonna '{c}' mancante nel CSV.")
    return df[t_col].to_numpy(), df[x_col].to_numpy(), df[y_col].to_numpy(), df[z_col].to_numpy()

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Stima spillato per sloshing in cilindro verticale da traiettoria traslazionale.")
    p.add_argument("--csv", required=True, help="Percorso CSV con colonne t,x,y,z")
    p.add_argument("--R", type=float, required=True, help="Raggio del cilindro [m]")
    p.add_argument("--H", type=float, required=True, help="Altezza utile [m]")
    p.add_argument("--V0", type=float, required=True, help="Volume iniziale [m^3]")
    p.add_argument("--method", choices=["instant","weir"], default="weir", help="Modello di spill: instant oppure weir")
    p.add_argument("--Cd", type=float, default=0.6, help="Coeff. di deflusso per 'weir'")
    args = p.parse_args()

    t,x,y,z = load_path_csv(args.csv)
    res = simulate_sloshing(t,x,y,z, R=args.R, H=args.H, V0=args.V0, method=args.method, Cd=args.Cd)
    # Stampa risultati sintetici
    total_spilled = float(res["V_spilled"][-1])
    print(f"Spillato totale [m^3]: {total_spilled:.6f}")
    # Esporta time series
    out = pd.DataFrame({
        "t": res["t"],
        "theta_rad": res["theta"],
        "tan_theta": res["tan_theta"],
        "V_remaining_m3": res["V"],
        "V_spilled_m3": res["V_spilled"],
        "a_h_mps2": res["ah"],
        "ax": res["ax"],
        "ay": res["ay"],
        "az": res["az"],
    })
    out_path = "sloshing_results.csv"
    out.to_csv(out_path, index=False)
    print(f"Serie temporali salvate in {out_path}")
