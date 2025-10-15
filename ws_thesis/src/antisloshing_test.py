import numpy as np
from scipy.spatial.transform import Rotation as R

# --- utility base ---
def surface_normal(points, ratio_threshold=0.1):
    """Stima la normale a una superficie con SVD."""
    c = np.mean(points, axis=0)
    _, S, Vt = np.linalg.svd(points - c)
    if S[-1] < ratio_threshold * S[0] and S[-1] < ratio_threshold * S[1]:
        normal = Vt[-1]
        return normal / np.linalg.norm(normal)
    else:
        return None

class exp_filt_rot:
    """Filtro esponenziale su SO(3)."""
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        self.R_prev = R.identity()
    def update(self, R_new):
        R_rel = self.R_prev.inv() * R_new
        rotvec = R_rel.as_rotvec() * self.alpha
        self.R_prev = R.from_rotvec(rotvec) * self.R_prev
        return self.R_prev

# --- funzione principale ---
def liq_compensate(particles_world, quat_init_wxyz, motion_axis_tool=np.array([1.,0.,0.]),
                   top_percent=10, use_ransac=False, alpha=0.2, omega_max=0.8):

    z = particles_world[:,2]
    thr = np.percentile(z, 100 - top_percent)
    surf = particles_world[z >= thr]
    if surf.shape[0] < 3:
        return quat_init_wxyz

    n = surface_normal(surf)

    zhat = np.array([0.,0.,1.])
    cos = np.clip(np.dot(n, zhat), -1.0, 1.0)
    angle = np.arccos(cos)
    axis = np.cross(n, zhat)
    s = np.linalg.norm(axis)
    R_corr = R.identity() if (s < 1e-9 or angle < 1e-6) else R.from_rotvec((axis / s) * angle)

    # Solo attorno a x utensile
    rotvec = R_corr.as_rotvec()
    
    # orientazione corrente dell’utensile
    q_xyzw = np.roll(quat_init_wxyz, -1)
    R0 = R.from_quat(q_xyzw)

    # porta l’asse x utensile nel mondo
    axis_world = R0.apply(motion_axis_tool)

    # proietta il rotvec sull’asse x utensile (espresso nel mondo)
    proj = np.dot(rotvec, axis_world) * axis_world
    R_corr = R.from_rotvec(proj)

    # filtro esponenziale su SO(3) con stato persistente per sessione
    static_lpf = getattr(liq_compensate, "_lpf", None)
    if static_lpf is None:
        static_lpf = exp_filt_rot(alpha=alpha)
        liq_compensate._lpf = static_lpf
    R_corr_f = static_lpf.update(R_corr)

    # rate limit per passo
    rotvec = R_corr_f.as_rotvec()
    nrm = np.linalg.norm(rotvec)
    if nrm > omega_max:
        rotvec *= (omega_max / nrm)
    R_step = R.from_rotvec(rotvec)

    R_des = R_step * R0
    q_out_xyzw = R_des.as_quat()
    q_out_wxyz = np.roll(q_out_xyzw, 1)
    return q_out_wxyz

# --- test ---
if __name__ == "__main__":
    # fittizie particelle con piano inclinato
    N = 1000
    x = np.random.uniform(-0.05, 0.05, N)
    y = np.random.uniform(-0.05, 0.05, N)
    z = 0.001 * x + 0.087 * y + 0.5  # piano inclinato 5°
    particles = np.stack([x, y, z], axis=1)

    q_init = np.array([1, 0, 0, 0])  # orientamento neutro (wxyz)

    # test single-axis
    q_out_single = liq_compensate(particles, q_init)

    R_single = R.from_quat(np.roll(q_out_single, -1))

    print("q_init: ",q_init)
    print("q_out_single:", q_out_single)
    print("Single-axis correction (deg):", R_single.as_euler('xyz', degrees=True))
