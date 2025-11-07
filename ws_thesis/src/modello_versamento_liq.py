"""
Pouring model: cylindrical source → cylindrical receiver.
- Receiver pose fixed. Source pose as time series of position+quaternion.
- Free-surface plane from gravity in source frame.
- Volume closure by solving for plane offset c via 1D root on exact circle-slice integral.
- Overflow handled as sharp-crested weir over the rim arc. Orifice formula included, commented.
- Jet trajectory ballistic; hit test with receiver mouth. Euler time integration, dt=0.01 s.
- Fluid defaults: water @ 20°C.

Frames and geometry:
- Source cylinder S: radius R, height H. Local axis +z from base z=0 to top rim z=H. Rim thickness t.
- Receiver cylinder R: radius r, height h. Axis +z'. Fixed SE(3) pose.
- World gravity g = [0,0,-9.81] m/s^2.

Inputs expected by simulate():
- H, D for source; h, d for receiver. Rim thickness t=0.002 m. Rim radius equals container radius.
- V0: initial liquid volume in source. Receiver starts empty.
- poses_source: list of dicts {t, p: (3,), q: (w,x,y,z)}
- pose_receiver: dict {p: (3,), q: (w,x,y,z)} fixed.
- Cw in [0.55,0.7].
- dt=0.01.

Outputs:
- dict with time histories: V_src, V_rcv, Q, Hs, b, hit_flag, loss_flag, jet_xyz, etc.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import List, Dict, Tuple, Optional

# ---------- Math utils ----------

def quat_to_R(q: Tuple[float,float,float,float]) -> np.ndarray:
    w,x,y,z = q
    n = w*w+x*x+y*y+z*z
    if n == 0:
        return np.eye(3)
    s = 2.0/n
    wx, wy, wz = s*w*x, s*w*y, s*w*z
    xx, xy, xz = s*x*x, s*x*y, s*x*z
    yy, yz = s*y*y, s*y*z
    zz = s*z*z
    R = np.array([
        [1-(yy+zz), xy-wz, xz+wy],
        [xy+wz, 1-(xx+zz), yz-wx],
        [xz-wy, yz+wx, 1-(xx+yy)]
    ])
    return R

# Circle one-side area for line x<=d in a radius-R disk is: A_left = πR^2 - A_right(d)
# where A_right(d) = R^2*arccos(d/R) + d*sqrt(R^2 - d^2), for |d|<=R, and saturations outside.

def disk_area_halfspace(d: float, R: float) -> float:
    if R <= 0:
        return 0.0
    if d <= -R:
        return 0.0
    if d >= R:
        return np.pi*R*R
    A_right = R*R*np.arccos(d/R) + d*np.sqrt(max(R*R - d*d, 0.0))
    return np.pi*R*R - A_right

# Compute volume inside cylinder {x^2+y^2<=R^2, 0<=z<=H} and halfspace n·x <= c
# exact by integrating z-slices of disk areas with d(z) = (c - n_z z)/||n_xy||.

def cylinder_halfspace_volume(n: np.ndarray, c: float, R: float, H: float, Nz: int=256) -> float:
    n = n/np.linalg.norm(n)
    nx,ny,nz = n
    nxy = np.hypot(nx, ny)
    z = np.linspace(0.0, H, Nz)
    if nxy < 1e-9:
        # plane parallel to disk normals, inequality nx=ny≈0 → nz*z <= c
        # Area is full disk if z <= c/nz (assuming nz>0); else 0.
        if abs(nz) < 1e-12:
            return 0.0
        zstar = c/nz
        zstar = np.clip(zstar, 0.0, H)
        return np.pi*R*R*zstar
    # general case
    cprime = c - nz*z  # vectorized
    d = cprime / nxy
    A = np.vectorize(disk_area_halfspace)(d, R)
    # integrate area over z
    V = np.trapz(A, z)
    return V

# Solve for c so that volume equals V_target using bracket + bisection

def solve_c_for_volume(n: np.ndarray, V_target: float, R: float, H: float) -> float:
    n = n/np.linalg.norm(n)
    V_tot = np.pi*R*R*H
    V_target = float(np.clip(V_target, 0.0, V_tot))
    # c range: min and max of n·x over cylinder to cover all volumes
    # n·x spans [cmin, cmax] over the cylinder: min occurs at extreme corners. Conservative bracket:
    # over disk radius R and z in [0,H]: n·x ∈ [min_z + min_xy, max_z + max_xy]
    nx,ny,nz = n
    nxy = np.hypot(nx, ny)
    cmin = nz*0 - nxy*R  # at z=0 and opposite rim
    cmax = nz*H + nxy*R  # at z=H and aligned rim
    # bisection
    lo, hi = cmin-1.0, cmax+1.0
    for _ in range(80):
        mid = 0.5*(lo+hi)
        Vmid = cylinder_halfspace_volume(n, mid, R, H)
        if Vmid < V_target:
            lo = mid
        else:
            hi = mid
    return 0.5*(lo+hi)

@dataclass
class Cylinder:
    H: float
    D: float
    @property
    def R(self) -> float:
        return 0.5*self.D
    @property
    def volume(self) -> float:
        return np.pi*self.R**2*self.H

@dataclass
class Pose:
    p: np.ndarray  # world position (3,)
    R: np.ndarray  # world rotation (3,3)

@dataclass
class Fluid:
    rho: float = 998.2        # kg/m^3 water 20°C
    mu: float = 1.002e-3      # Pa·s
    sigma: float = 0.0728     # N/m

# ---------- Core model ----------

def pouring_step(state: Dict, src: Cylinder, rcv: Cylinder, pose_s: Pose, pose_r: Pose,
                 g_world: np.ndarray, Cw: float, dt: float, rim_t: float,
                 use_weir: bool=True) -> Dict:
    """
    state carries: V_src, V_rcv
    returns per-step diagnostics and updated state inside state.
    """
    V_src = state["V_src"]
    V_rcv = state["V_rcv"]

    # gravity in source frame
    n_s = pose_s.R.T @ (g_world / np.linalg.norm(g_world))  # unit vector

    # solve plane offset c so that volume below plane equals V_src
    c = solve_c_for_volume(n_s, V_src, src.R, src.H)

    # rim overflow geometry at top rim z=H in source frame
    nx,ny,nz = n_s
    nxy = float(np.hypot(nx, ny))
    M = src.R * nxy
    s0 = c - nz*src.H  # threshold at rim plane z=H
    # arc width b where n·x <= c on rim circle
    if M < 1e-12:
        b = 2*np.pi*src.R if s0 >= 0 else 0.0
    else:
        k = np.clip(s0/M, -1.0, 1.0)
        if s0 <= -M:
            b = 0.0
        elif s0 >= M:
            b = 2*np.pi*src.R
        else:
            b = (2*np.pi - 2*np.arccos(k)) * src.R
    # head above lowest rim point along +n
    Hs = max(0.0, c - (nz*src.H - M))

    # discharge
    g = float(np.linalg.norm(g_world))
    if use_weir:
        Q = (2.0/3.0) * Cw * b * np.sqrt(2.0*g) * (Hs**1.5)
        # Orifice alternative: uncomment to use instead of weir
        # Cd = 0.62
        # A_eff = max(b*rim_t, 0.0)
        # Q = Cd * A_eff * np.sqrt(2.0*g*Hs)
    else:
        # Fallback: zero if not using weir
        Q = 0.0

    Q = min(Q, V_src/dt)  # cannot pour more than available

    # Pick release point at rim: direction of minimum n·x projected to rim
    # angle phi of projection
    phi = np.arctan2(ny, nx)  # direction of +x' aligned with n_xy
    theta_release = phi + np.pi  # opposite to projection → lowest along +n
    x_local = np.array([src.R*np.cos(theta_release), src.R*np.sin(theta_release), src.H])
    x_world = pose_s.p + pose_s.R @ x_local

    # jet initial velocity: container motion + efflux along +n_s (down in source)
    v_eff = np.sqrt(2.0*g*Hs) if Hs > 0 else 0.0
    v0_world = v_eff * (pose_s.R @ n_s)
    # approximate container linear velocity by finite difference stored in state
    v_cont = state.get("v_src_world", np.zeros(3))
    v0_world = v0_world + v_cont

    # ballistic intersection with receiver mouth plane (z'=h in receiver frame)
    # Solve for t_hit >=0 where z'(x_world + v0*t + 0.5*g*t^2) == h
    Rw_r = pose_r.R.T  # world→receiver rotation
    pw_r = Rw_r @ (x_world - pose_r.p)
    vw_r = Rw_r @ v0_world
    gw_r = Rw_r @ g_world

    t_hit = None
    hit = False
    if Q > 0:
        # z'(t) = pw_r.z + vw_r.z*t + 0.5*gw_r.z*t^2
        a = 0.5*gw_r[2]
        bq = vw_r[2]
        c_q = pw_r[2] - rcv.H  # mouth plane at z'=H (top rim)
        roots = []
        if abs(a) < 1e-12:
            if abs(bq) > 1e-12:
                t = -c_q/bq
                if t >= 0:
                    roots.append(t)
        else:
            disc = bq*bq - 4*a*c_q
            if disc >= 0:
                rt = np.sqrt(disc)
                t1 = (-bq - rt)/(2*a)
                t2 = (-bq + rt)/(2*a)
                if t1 >= 0: roots.append(t1)
                if t2 >= 0: roots.append(t2)
        if roots:
            t_hit = min(roots)
            # position at hit
            x_hit_w = x_world + v0_world*t_hit + 0.5*g_world*(t_hit**2)
            x_hit_r = Rw_r @ (x_hit_w - pose_r.p)
            rr = np.hypot(x_hit_r[0], x_hit_r[1])
            hit = rr <= rcv.R + 1e-6

    eta = 1.0 if hit else 0.0

    # update volumes
    V_src_new = max(0.0, V_src - Q*dt)
    V_rcv_new = min(rcv.volume, V_rcv + eta*Q*dt)

    diag = {
        "Q": Q,
        "Hs": Hs,
        "b": b,
        "hit": bool(hit),
        "x_release_world": x_world,
        "v0_world": v0_world,
        "t_hit": t_hit,
        "x_hit_world": x_world if t_hit is None else x_world + v0_world*t_hit + 0.5*g_world*(t_hit**2)
    }

    state["V_src"], state["V_rcv"] = V_src_new, V_rcv_new
    return diag

# ---------- Simulation driver ----------

def simulate(H: float, D: float, h: float, d: float,
             V0: float,
             poses_source: List[Dict],
             pose_receiver: Dict,
             Cw: float=0.62,
             dt: float=0.01,
             g: float=9.81,
             rim_t: float=0.002,
             fluid: Fluid=Fluid()) -> Dict[str, np.ndarray]:
    """Run time-marched simulation.
    poses_source is a list sorted by time with fields t, p, q.
    pose_receiver has p, q (fixed).
    Returns dict with arrays sampled at source timestamps.
    """
    src = Cylinder(H=H, D=D)
    rcv = Cylinder(H=h, D=d)

    # clamp V0
    V0 = float(np.clip(V0, 0.0, src.volume))

    g_world = np.array([0.0, 0.0, -abs(g)])

    # build Pose objects
    ts = np.array([row["t"] for row in poses_source], dtype=float)
    ps = np.array([row["p"] for row in poses_source], dtype=float)
    Rs = np.array([quat_to_R(tuple(row["q"])) for row in poses_source])

    pose_r = Pose(p=np.array(pose_receiver["p"], dtype=float),
                  R=quat_to_R(tuple(pose_receiver["q"])) )

    # estimates of source linear velocity via finite difference
    vs = np.zeros_like(ps)
    if len(ts) >= 2:
        dtv = np.diff(ts)
        dtv[dtv==0] = dt  # avoid zero
        v_mid = np.diff(ps, axis=0)/dtv[:,None]
        vs[1:-1] = 0.5*(v_mid[:-1] + v_mid[1:])
        vs[0] = v_mid[0]
        vs[-1] = v_mid[-1]

    state = {"V_src": V0, "V_rcv": 0.0, "v_src_world": np.zeros(3)}

    out_Q = []
    out_Vs = []
    out_Vr = []
    out_Hs = []
    out_b = []
    out_hit = []

    t_prev = ts[0]
    for k in range(len(ts)):
        pose_s = Pose(p=ps[k], R=Rs[k])
        state["v_src_world"] = vs[k]

        # substeps if input timestamps are sparse; integrate in steps of dt
        t_now = ts[k]
        n_sub = max(1, int(np.ceil(max(1e-12, t_now - t_prev)/dt)))
        for _ in range(n_sub):
            diag = pouring_step(state, src, rcv, pose_s, pose_r, g_world, Cw, dt, rim_t, use_weir=True)
        t_prev = t_now

        out_Q.append(diag["Q"])
        out_Vs.append(state["V_src"])
        out_Vr.append(state["V_rcv"])
        out_Hs.append(diag["Hs"])
        out_b.append(diag["b"])
        out_hit.append(1.0 if diag["hit"] else 0.0)

    return {
        "t": ts,
        "Q": np.array(out_Q),
        "V_src": np.array(out_Vs),
        "V_rcv": np.array(out_Vr),
        "Hs": np.array(out_Hs),
        "b": np.array(out_b),
        "hit": np.array(out_hit),
        "meta": {
            "fluid": fluid.__dict__,
            "Cw": Cw,
            "dt": dt,
            "rim_t": rim_t,
            "g": g
        }
    }

# ---------- Example usage stub ----------
if __name__ == "__main__":
    # Minimal sanity test with a simple tilt: tip 90° over 5 seconds.
    import math
    H,D = 0.20, 0.08
    h,d = 0.25, 0.10
    V0 = 0.7*np.pi*(D/2)**2*H

    T = 5.0
    N = int(T/0.05)+1
    ts = np.linspace(0, T, N)
    poses = []
    for t in ts:
        # rotate around y by up to 100°
        angle = min(100.0, 20.0*t) * math.pi/180.0
        cy, sy = math.cos(angle/2), math.sin(angle/2)
        q = (cy, 0.0, sy, 0.0)  # w,x,y,z
        p = (0.0, 0.0, 0.3)
        poses.append({"t": float(t), "p": p, "q": q})

    pose_r = {"p": (0.20, 0.0, 0.25), "q": (1.0, 0.0, 0.0, 0.0)}

    out = simulate(H,D,h,d,V0, poses, pose_r, Cw=0.62, dt=0.01)
    for k in range(0, len(out["t"]), max(1,len(out["t"])//10)):
        print(f"t={out['t'][k]:.2f} Q={out['Q'][k]:.5f} Vsrc={out['V_src'][k]:.4f} Vrcv={out['V_rcv'][k]:.4f} hit={out['hit'][k]:.0f}")
