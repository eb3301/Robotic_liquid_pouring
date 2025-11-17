import numpy as np
from scipy.spatial.transform import Rotation as Rot
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

def get_d(y: float, c: float, theta: float,R: float)->float:
    """
    Ottieni la distanza del pelo libero del liquido dal centro del cilindro 
    misurata su un piano parallelo al fondo data
    la distanza dal fondo, l'angolo d'inclinazione e il raggio
    """
    if np.isclose(np.tan(theta), 0.0): # Pelo libero quasi orizzontale
        d = -c
    else:
        d=np.clip((y -c) /np.tan(theta), -R, R)
    return d

def area(d: float, R: float) -> float:
    """
    Restituisce area del disco meno segmento circolare nel caso in cui
    il liquido sia inclinato e non copra tutta la superficie del cerchio
    tenendo conto dei limiti geometrici

    Input:
    d = distanza del pelo libero del liquido dal centro del cilindro misurata su un piano parallelo al fondo
    R = raggio del cilindro

    Output:
    A_liq = area di una slice di liquido
    """

    if R <= 0:
        return 0.0
    
    # Limiti cilindro
    if d <= -R:
        return np.pi*R*R
    if d >= R:
        return 0.0
    
    A_sett = 0.5 * 2 * R**2 * np.arccos(d/R)
    A_triang = 2 * 0.5 * np.sqrt(R**2 - d**2) * d
    A_segm = A_sett - A_triang

    A_liq = R**2*np.pi - A_segm
    
    return A_liq

def volume(H: float, R: float, c: float, theta: float, n: int=1000)->float:
    """
    Ottieni il volume del liquido contenuto nel contenitore
    """
    a=0
    b=H
    h = (b - a) / n
    
    # Def estremi
    d0 = get_d(a, c, theta, R)
    A0 = area(d0, R)
    df = get_d(b, c, theta, R)
    Af = area(df, R)
    s = 0.5 * (A0 + Af)

    for i in range(1, n):
        y_i = a + i * h
        d_i = get_d(y_i, c, theta, R)
        A_i = area(d_i, R)
        s += A_i
    V = s * h
    
    return s * h

def find_c(H: float, R: float, theta: float, V_target: float,
           c_min: float=-10.0, c_max: float=10.0, n: int=500, tol: float=1e-6)-> float:
    """
    Trova il valore di c tale che il volume calcolato coincida con V_target.
    Bisezione, assumendo monotonicità di V(c) nell'intervallo.
    """

    def F(c):
        return volume(H, R, c, theta, n) - V_target

    f_min = F(c_min)
    f_max = F(c_max)

    if f_min * f_max > 0:
        raise ValueError(
            f"Nessuna radice nell'intervallo iniziale. "
            f"[c_min={c_min}, f_min={f_min}] "
            f"[c_max={c_max}, f_max={f_max}]"
        )
    
    while c_max - c_min > tol:
        c_mid = 0.5 * (c_min + c_max)
        f_mid = F(c_mid)

        if abs(f_mid) <= tol:
            return c_mid
        if f_min * f_mid < 0:
            c_max = c_mid
            f_max = f_mid
        else:
            c_min = c_mid
            f_min = f_mid

    return 0.5 * (c_min + c_max)

def get_h_spill(c: float, theta: float, H: float, R: float) -> float:
    d_h = get_d(H,c,theta,R)
    h_spill = R - np.abs(d_h)

    return max(h_spill, 0.0)

def calc_v_trasc(trj):
    return 0

def reward_pouring(num_waypoints: int, theta_f: float, vol_target: float, parameters: dict)->int:
    # Parameters
    H = 9.5 * 1e-2
    R = 3 * 1e-2
    dt=0.01
    Cd = 0.6
    g = 9.81
    
    plotter = FreeSurfacePlotter(width=2*R, height=H)

    V_init = parameters["vol_init"] * 1e-6 if parameters["vol_init"]>1 else parameters["vol_init"]
    V_target = vol_target * 1e-6 if vol_target>1 else vol_target #40 * 1e-6 #parameters["vol_target"]
    tol_pour = 0.15 * V_target
    tol_spill = 0.1 * V_init

    n_steps = int(num_waypoints/2)
    theta_f = np.deg2rad(theta_f) if theta_f>2*np.pi else theta_f
    theta_arr = np.concatenate((np.linspace(0, theta_f, n_steps), np.linspace(theta_f, 0.0, n_steps)))

    pos_cont= np.array([parameters['pos_cont_goal'][0],parameters['pos_cont_goal'][1],parameters['pos_cont_goal'][2]])
    
    x_shift=0.15
    z_min=0.967
    lip_height = parameters['pos_cont_goal'][2] + H +0.07
    quat_orizz = np.array([0.5,-0.5,0.5,-0.5])
    pos4 = pos_cont
    pos4[0]-=x_shift
    pos4[1] -= (R+0.03)
    pos4[2] += R
    pos4[2]=max(pos4[2],z_min,lip_height)
    quat4 = quat_orizz
    CoR3D = np.array([
        parameters['pos_cont_goal'][0] + parameters['dCoR'][0], # 0.0
        parameters['pos_cont_goal'][1] - 0.005 + parameters['dCoR'][1], # - 0.01 
        parameters['pos_cont_goal'][2] + parameters['dCoR'][2], # + 0.04
    ])
    p_tcp0 = pos4.copy()
    R0 = Rot.from_quat(quat4) # matrice rot init
    l = R0.inv().apply(CoR3D - p_tcp0) # offset tool0 --> CoR3D
    lip_l = np.array([0.0, (R + 0.003), H])
    axis_world  = np.array([1.0, 0.0, 0.0]) # asse x nel frame tool

    V_poured=0
    V=V_init
    V_spilled = 0

    for i,theta in enumerate(theta_arr):
       
        c = find_c(H, R, theta, V)
        
        plotter.update(theta, c)

        h_spill = get_h_spill(c, theta, H, R)
        print(f"c: {c}, h: {h_spill}")
        L = 2 * np.arccos((R-h_spill)/R) * R
        
        # Update volumes:
        Q = 2/3 * Cd * np.sqrt(2*g) * L * h_spill**1.5
        V_i=Q*dt
        V_poured = np.clip(V_poured + V_i, 0, V_init)
        V = np.clip(V - V_i, 0, V_init)
        print(f"[iter {i} - theta: {np.rad2deg(theta)}] - [Vol: {int(V*1e6)} Vol_poured: {int(V_poured*1e6)}]")
        # Evaluate positions:                   # TODO aggiungere v_trascinamento e calc_v_trasc
        v = np.sqrt(2*g*h_spill)
        
        R_theta = Rot.from_rotvec(theta * axis_world) * R0 # matrice rotazione lungo x
        delta_pos=R_theta.apply(l)
        p_tcp = CoR3D - delta_pos
        p_tcp[2] = max(pos4[2],z_min, lip_height)

        p_lip = p_tcp + R_theta.apply(lip_l)
        # il movimento è nel piano Y-Z
        x=p_lip[0]
        y=p_lip[1]
        z=p_lip[2]

        z_f = z_min+0.003
        y_f = y + v * np.sqrt(2*(z-z_f)/g)
        x_f = x
        p_liq = np.array([x_f, y_f, z_f])
        err_pos=np.linalg.norm(p_liq[:-1]-pos_cont[:-1])
        if err_pos >= R:
            V_spilled += V_i
            V_poured -= V_i

    err_vol = np.abs(V_poured-V_target)
    print(f"Volume poured: {V_poured}")
    print(f"Volume spilled: {V_spilled}")
    if err_vol < tol_pour and V_spilled < tol_spill:
        return 1
    else:
        return 0

class FreeSurfacePlotter:
    def __init__(self, width, height, xlim=None, ylim=None):
        """
        Inizializza la figura, il rettangolo e la retta.

        width, height : dimensioni del rettangolo
        xlim, ylim    : limiti display (opzionali)
        """
        self.width = width
        self.height = height
        
        # Vertici del rettangolo non ruotato (centrato in 0,0)
        self.rect_local = np.array([
            [-width/2, -height/2],
            [ width/2, -height/2],
            [ width/2,  height/2],
            [-width/2,  height/2]
        ])

        # Figura
        self.fig, self.ax = plt.subplots(figsize=(5,5))

        # Limiti automatici se non forniti
        L = max(width, height)
        if xlim is None: xlim = (-L, L)
        if ylim is None: ylim = (-L, L)

        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)
        self.ax.set_aspect("equal")

        # Patch del rettangolo (inizialmente non ruotato)
        self.rect_patch = Polygon(self.rect_local, closed=True, fc='none', ec='black', lw=2)
        self.ax.add_patch(self.rect_patch)

        # Linea del pelo libero
        self.x_line = np.linspace(xlim[0], xlim[1], 300)
        self.line_plot, = self.ax.plot(self.x_line, np.zeros_like(self.x_line), 'b', lw=2)

        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")

        plt.ion()
        plt.show()

    def rotate(self, theta):
        """Rotazione 2D dei punti locali del rettangolo."""
        Rm = np.array([
            [ np.cos(theta), -np.sin(theta)],
            [ np.sin(theta),  np.cos(theta)]
        ])
        return self.rect_local @ Rm.T

    def update(self, theta, c):
        """
        Aggiorna rettangolo ruotato e retta del pelo libero.
        """
        # Aggiorna rettangolo
        rect_rot = self.rotate(theta)
        #self.rect_patch.set_xy(rect_rot)

        # Aggiorna retta
        y_line = np.tan(theta)*self.x_line + c
        self.line_plot.set_data(self.x_line, y_line)

        # Aggiorna titolo e disegno
        self.ax.set_title(f"θ = {np.rad2deg(theta):.1f}°, c = {c}")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


def main():
    parameters = {
                "pos_init_cont": [0.8021465039268282, 0.2847160024669491, 0.9564999991059303],
                "pos_cont_goal": [0.746, 0.961, 0.960],
                "pos_init_ee":  (0.12394027698787038, 0.2665910764094347, 1.1638763741892955, -0.5837257860699608, 0.5925713099055904, -0.388471134978874, 0.3965017359886389),
                "pos_grip_ee": (0.6509734785173125, 0.2847438888358813, 0.9764986855761588, -0.5000033337808922, 0.49998929956451965, -0.4999940109139501, 0.5000133554007884),
                "offset": (0, 0.15, -0.02),
                "dCoR": [0.0, 0.06, -0.004],
                "vol_init": 60.0, #2e-5, +-MAE
                "densità": 998.0,
                "viscosità": 0.001,
                "tens_sup": 0.072,
                "vol_target": 20.0, #0.75e-5,
                "err_target": 5e-6,
                "theta_f": 87, #+-15°
                "num_wp": 320,
            }
    theta_f=90
    num_waypoints=320  
    vol_target=40
    reward=reward_pouring(num_waypoints, theta_f, vol_target, parameters)
    
if __name__ == '__main__':
    main()  


