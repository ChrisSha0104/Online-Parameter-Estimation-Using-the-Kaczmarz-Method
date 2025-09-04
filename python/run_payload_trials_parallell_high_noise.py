#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import numpy as np
from collections import deque
from scipy.spatial.transform import Rotation as R

from python.quadrotor.quadrotor import QuadrotorDynamics
from python.common.LQR_controller import LQRController
from python.common.noise_models import apply_noise
from python.common.estimation_methods import *  # RLS, KF, RK, TARK, IGRK, MGRK, REK, DEKA

np.set_printoptions(precision=7, suppress=True)

ALGOS_DEFAULT = ["fdbk", "mgrk", "igrk", "rek", "rk", "tark", "rls", "kf", "gt"]
REF_CHOICES   = ["figure8", "circle", "spiral", "hover", "lissajous", "ellipse", "helix"]

# ----------------------- helpers -----------------------
def rel_residual(A, x, b, eps=1e-12):
    r = A @ x - b
    return np.linalg.norm(r) / max(np.linalg.norm(b), eps)

def closest_spd(theta, epsilon=1e-6):
    """Project inertia to nearest SPD by symmetrization + eigval clamp."""
    m, cx, cy, cz, Ixx, Iyy, Izz, Ixy, Ixz, Iyz = theta
    A = np.array([[Ixx, Ixy, Ixz],
                  [Ixy, Iyy, Iyz],
                  [Ixz, Iyz, Izz]])
    A_sym = 0.5*(A + A.T)
    w, V = np.linalg.eigh(A_sym)
    w = np.clip(w, epsilon, None)
    A_spd = V @ np.diag(w) @ V.T
    Ixx_spd, Iyy_spd, Izz_spd = A_spd[0,0], A_spd[1,1], A_spd[2,2]
    Ixy_spd, Ixz_spd, Iyz_spd = A_spd[0,1], A_spd[0,2], A_spd[1,2]
    return np.array([m, cx, cy, cz, Ixx_spd, Iyy_spd, Izz_spd, Ixy_spd, Ixz_spd, Iyz_spd])

# ----------------------- trajectories -----------------------
def make_traj_hover(T, dt, rng):
    t = np.arange(0, T, dt)
    pos = np.zeros((len(t), 3))
    vel = np.zeros_like(pos)
    return t, pos, vel

def make_traj_figure8(T, dt, rng):
    A = rng.uniform(0.4, 0.8)
    B = rng.uniform(0.2, 0.4)
    z0 = rng.uniform(-0.05, 0.05)
    omega = (2*np.pi/20.0) * rng.uniform(0.8, 1.25)
    t = np.arange(0, T, dt)
    x = A * np.sin(omega * t)
    y = B * np.sin(2*omega * t)
    z = z0 * np.ones_like(t)
    vx = A * omega * np.cos(omega * t)
    vy = 2 * B * omega * np.cos(2*omega * t)
    vz = np.zeros_like(t)
    return t, np.vstack([x,y,z]).T, np.vstack([vx,vy,vz]).T

def make_traj_circle(T, dt, rng):
    Rr = rng.uniform(0.3, 0.6)
    omega = (2*np.pi/15.0) * rng.uniform(0.8, 1.2)
    z0 = rng.uniform(-0.05, 0.05)
    t = np.arange(0, T, dt)
    x = Rr * np.cos(omega * t)
    y = Rr * np.sin(omega * t)
    z = z0 * np.ones_like(t)
    vx = -Rr * omega * np.sin(omega * t)
    vy =  Rr * omega * np.cos(omega * t)
    vz = np.zeros_like(t)
    return t, np.vstack([x,y,z]).T, np.vstack([vx,vy,vz]).T

def make_traj_lissajous(T, dt, rng):
    Ax = rng.uniform(0.35, 0.7); Ay = rng.uniform(0.2, 0.5)
    wx = (2*np.pi/18.0) * rng.uniform(0.8, 1.2)
    wy = 3 * wx
    phi = rng.uniform(0, np.pi/2)
    z0 = rng.uniform(-0.05, 0.05)
    t = np.arange(0, T, dt)
    x = Ax * np.sin(wx * t + phi)
    y = Ay * np.sin(wy * t)
    z = z0 * np.ones_like(t)
    vx = Ax * wx * np.cos(wx * t + phi)
    vy = Ay * wy * np.cos(wy * t)
    vz = np.zeros_like(t)
    return t, np.vstack([x,y,z]).T, np.vstack([vx,vy,vz]).T


# -------- NEW 1: ellipse (racetrack/oval) --------
def make_traj_ellipse(T, dt, rng):
    Ax = rng.uniform(0.5, 0.9)   # semi-major
    By = rng.uniform(0.2, 0.5)   # semi-minor
    z0 = rng.uniform(-0.05, 0.05)
    omega = (2*np.pi/18.0) * rng.uniform(0.9, 1.2)
    t = np.arange(0, T, dt)
    x = Ax * np.cos(omega * t)
    y = By * np.sin(omega * t)
    z = z0 * np.ones_like(t)
    vx = -Ax * omega * np.sin(omega * t)
    vy =  By * omega * np.cos(omega * t)
    vz = np.zeros_like(t)
    return t, np.vstack([x,y,z]).T, np.vstack([vx,vy,vz]).T

# -------- NEW 2: helix (circle in XY, sinusoid in Z) --------
def make_traj_helix(T, dt, rng):
    Rr = rng.uniform(0.3, 0.6)
    omega_xy = (2*np.pi/16.0) * rng.uniform(0.9, 1.2)
    z0 = rng.uniform(-0.05, 0.05)
    z_amp = rng.uniform(0.04, 0.10)
    omega_z = omega_xy * rng.uniform(0.4, 0.8)
    t = np.arange(0, T, dt)
    x = Rr * np.cos(omega_xy * t)
    y = Rr * np.sin(omega_xy * t)
    z = z0 + z_amp * np.sin(omega_z * t)
    vx = -Rr * omega_xy * np.sin(omega_xy * t)
    vy =  Rr * omega_xy * np.cos(omega_xy * t)
    vz =  z_amp * omega_z * np.cos(omega_z * t)
    return t, np.vstack([x,y,z]).T, np.vstack([vx,vy,vz]).T

# -------- NEW 3: spiral (slowly expanding circle in XY) --------
def make_traj_spiral(T, dt, rng):
    R0 = rng.uniform(0.2, 0.35)
    R1 = rng.uniform(0.45, 0.7)
    z0 = rng.uniform(-0.05, 0.05)
    omega = (2*np.pi/18.0) * rng.uniform(0.8, 1.2)
    t = np.arange(0, T, dt)
    k = (R1 - R0) / max(T, 1e-6)   # radial growth rate
    R_t = R0 + k * t
    x = R_t * np.cos(omega * t)
    y = R_t * np.sin(omega * t)
    z = z0 * np.ones_like(t)
    # analytic velocities: r' = k
    vx = k * np.cos(omega * t) - R_t * omega * np.sin(omega * t)
    vy = k * np.sin(omega * t) + R_t * omega * np.cos(omega * t)
    vz = np.zeros_like(t)
    return t, np.vstack([x,y,z]).T, np.vstack([vx,vy,vz]).T

TRAJ_FACTORY = {
    "hover":     make_traj_hover,
    "figure8":   make_traj_figure8,
    "circle":    make_traj_circle,
    "lissajous": make_traj_lissajous,
    "ellipse":   make_traj_ellipse,   # NEW
    "helix":     make_traj_helix,     # NEW
    "spiral":    make_traj_spiral,    # NEW
}

def get_error_state(quad, x, xg):
    pos_err = x[0:3] - xg[0:3]
    phi     = quad.qtorp(x[3:7])
    vel_err = x[7:10] - xg[7:10]
    om_err  = x[10:13] - xg[10:13]
    return np.hstack([pos_err, phi, vel_err, om_err])

# ----------------------- estimators -----------------------
def make_estimator(name: str, theta0: np.ndarray, window: int):
    n = int(theta0.shape[0])
    name = name.lower()
    if name == "rls":
        return RLS(num_params=n, theta_hat=theta0, forgetting_factor=0.7, c=10)
    if name == "kf":
        Q = 1e-3 * np.eye(n)
        Rm = 1e-4 * np.eye(window * 6)
        return KF(num_params=n, process_noise=Q, measurement_noise=Rm, theta_hat=theta0, c=10)
    if name == "rk":
        return RK(num_params=n, x0=theta0)
    if name == "tark":
        return TARK(num_params=n, x0=theta0)
    if name == "igrk":
        return IGRK(num_params=n, x0=theta0)
    if name == "mgrk":
        return MGRK(num_params=n, x0=theta0)
    if name == "rek":
        return REK(num_params=n, x0=theta0)
    if name == "deka":
        return DEKA(num_params=n, x0=theta0)
    if name == "fdbk":
        return FDBK(num_params=n, x0=theta0)
    if name in ("gt", "none"):
        return None
    raise ValueError(f"Unknown estimator: {name}")

# ----------------------- single trial -----------------------
def run_single_trial(estimator_name: str, base_seed: int, trial_idx: int,
                     est_freq: int, window: int, ref_type: str, T_sim: float = 20.0):
    """
    Core simulation for ONE trial with given algo & seed. Raises on numerical errors.
    """
    rng = np.random.default_rng(base_seed + trial_idx)
    quad = QuadrotorDynamics()
    ug   = quad.hover_thrust_true.copy()

    # LQR around hover
    xg0 = np.hstack([np.zeros(3), np.array([1,0,0,0]), np.zeros(6)])
    A_lin, B_lin = quad.get_linearized_true(xg0, ug)
    max_dev_x = np.array([0.1,0.1,0.1,  0.5,0.5,0.05,  0.5,0.5,0.5,  0.7,0.7,0.2])
    max_dev_u = np.array([0.5,0.5,0.5,0.5])
    Q = np.diag(1.0/max_dev_x**2)
    Rmat = np.diag(1.0/max_dev_u**2)
    lqr = LQRController(A_lin, B_lin, Q, Rmat)

    # Trajectory
    dt = quad.dt
    t, pos_ref, vel_ref = TRAJ_FACTORY[ref_type](T_sim, dt, rng)

    # Initial state
    x_true = np.zeros(13)
    x_true[0:3]   = pos_ref[0] + 0.1*rng.standard_normal(3)
    x_true[3:7]   = quad.rptoq(0.02*rng.standard_normal(3))
    x_true[7:10]  = vel_ref[0] + 0.01*rng.standard_normal(3)
    x_true[10:13] = 0.01*rng.standard_normal(3)
    x_meas = x_true.copy()

    # Randomized events
    add_step  = rng.integers(200, 301)
    drop_step = rng.integers(500, 701)
    payload_m  = rng.uniform(0.01, 0.02)
    payload_dr = rng.uniform(-0.002, 0.002, size=3)

    # Estimator
    theta0 = quad.get_true_inertial_params()
    est    = make_estimator(estimator_name, theta0, window)

    # Rolling buffers
    A_buf, B_buf = deque(maxlen=window), deque(maxlen=window)

    # Storage
    x_meas_traj, x_ref_traj, est_err_traj = [], [], []

    u = ug.copy()
    for k in range(len(t)):
        rg, vg = pos_ref[k], vel_ref[k]
        qg, omgg = np.array([1,0,0,0]), np.zeros(3)
        xg = np.hstack([rg, qg, vg, omgg])

        x_ref_traj.append(xg.copy())
        x_meas_traj.append(x_meas.copy())

        # control
        x_err12 = get_error_state(quad, x_meas, xg)
        u = ug + lqr.control(x_err12)

        # propagate + measure
        x_true_next = quad.dynamics_rk4_true(x_true, u, dt=dt)
        x_meas_next = x_true_next.copy() #apply_noise(x_true_next)

        # residual row-block
        dx_meas = (x_meas_next - x_meas) / dt
        A_mat = quad.get_data_matrix(x_meas, dx_meas)         # (6,10)
        b_vec = quad.get_force_vector_from_A(A_mat)
        noise_std = np.linalg.norm(b_vec) * 0.10
        b_vec += noise_std * rng.standard_normal(b_vec.shape)
        # b_vec = quad.get_force_vector(x_meas, dx_meas, u)     # (6,)
        A_buf.append(A_mat); B_buf.append(b_vec)

        # update (skip t=0)
        if (k % est_freq == 0) and (k > 0):
            A_stack = np.vstack(list(A_buf))                  # (6*window, 10)
            b_stack = np.concatenate(list(B_buf), axis=0)     # (6*window,)

            if estimator_name == "gt":
                theta_gt = quad.get_true_inertial_params()
                est_err_traj.append(rel_residual(A_stack, theta_gt, b_stack))
                ug = quad.get_hover_thrust_true()
                A_new, B_new = quad.get_linearized_true(xg, ug)
                lqr.update_linearized_dynamics(A_new, B_new)

            elif estimator_name == "lstsq":
                theta_est = np.linalg.lstsq(A_stack, b_stack, rcond=None)[0]
                theta_est[4] = theta_est[5] = np.mean(theta_est[4:6])
                theta_est = closest_spd(theta_est)
                quad.set_estimated_inertial_params(theta_est)
                ug = quad.get_hover_thrust_est()
                A_new, B_new = quad.get_linearized_est(xg, ug)
                lqr.update_linearized_dynamics(A_new, B_new)
                est_err_traj.append(rel_residual(A_stack, theta_est, b_stack))

            elif estimator_name in ("deka","rk","tark","rls","kf","mgrk","igrk","rek"):
                theta_est = est.iterate(A_stack, b_stack)
                theta_est[4] = theta_est[5] = np.mean(theta_est[4:6])
                theta_est = closest_spd(theta_est)
                quad.set_estimated_inertial_params(theta_est)
                ug = quad.get_hover_thrust_est()
                A_new, B_new = quad.get_linearized_est(xg, ug)
                lqr.update_linearized_dynamics(A_new, B_new)
                est_err_traj.append(rel_residual(A_stack, theta_est, b_stack))

            elif estimator_name == "none":
                est_err_traj.append(rel_residual(A_stack, theta0, b_stack))

        # events
        if k == add_step:
            quad.attach_payload(m_p=payload_m, delta_r_p=payload_dr)
        elif k == drop_step:
            quad.detach_payload()

        x_true = x_true_next
        x_meas = x_meas_next

    # Arrays
    x_meas_traj = np.array(x_meas_traj)  # (T,13)
    x_ref_traj  = np.array(x_ref_traj)   # (T,13)

    # Errors
    pos_err = x_meas_traj[:, 0:3]  - x_ref_traj[:, 0:3]
    vel_err = x_meas_traj[:, 7:10] - x_ref_traj[:, 7:10]
    qu_meas = np.roll(x_meas_traj[:, 3:7], -1, axis=1)  # [x,y,z,w]
    qu_ref  = np.roll(x_ref_traj[:,  3:7], -1, axis=1)
    eul_meas = R.from_quat(qu_meas).as_euler('xyz', degrees=True)
    eul_ref  = R.from_quat(qu_ref ).as_euler('xyz', degrees=True)
    ori_err  = eul_meas - eul_ref

    abs_mean_pos_err_t = np.mean(np.abs(pos_err), axis=1)
    abs_mean_vel_err_t = np.mean(np.abs(vel_err), axis=1)
    abs_mean_ori_err_t = np.mean(np.abs(ori_err), axis=1)

    return {
        "abs_mean_pos_err_t": abs_mean_pos_err_t,
        "abs_mean_vel_err_t": abs_mean_vel_err_t,
        "abs_mean_ori_err_t": abs_mean_ori_err_t,
        "est_err_t":          np.array(est_err_traj),
        "x_meas_traj":        x_meas_traj,
        "x_ref_traj":         x_ref_traj,
        "t":                  t,
        "event_idx":          (int(add_step), int(drop_step)),
    }

# ----------------------- worker wrapper (picklable) -----------------------
def _trial_worker(estimator_name, base_seed, seed_offset, est_freq, window, ref_type, T_sim):
    """
    Returns (ok: bool, seed_offset: int, payload: dict|str)
    ok=True -> payload=out dict; ok=False -> payload=error string
    """
    try:
        out = run_single_trial(
            estimator_name=estimator_name,
            base_seed=base_seed + seed_offset,
            trial_idx=0,               # we vary base_seed per attempt
            est_freq=est_freq,
            window=window,
            ref_type=ref_type,
            T_sim=T_sim,
        )
        # consider NaN-filled outputs as failure (paranoid)
        if np.isnan(out["abs_mean_pos_err_t"]).any():
            return (False, seed_offset, "NaNs in output")
        return (True, seed_offset, out)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        return (False, seed_offset, repr(e))

# ----------------------- main (parallel per algo, multi-refs) -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="results_param_estimation")
    ap.add_argument("--ntrials", type=int, default=8)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--algos", type=str, nargs="*", default=ALGOS_DEFAULT)
    ap.add_argument("--refs", type=str, nargs="+", default=REF_CHOICES,
                    help="One or more trajectory types; results are concatenated and labeled")
    ap.add_argument("--est_freq", type=int, default=20)
    ap.add_argument("--window",   type=int, default=1)
    ap.add_argument("--save_traj", action="store_true")

    # parallel controls
    ap.add_argument("--workers", type=int, default=16,
                    help="Process workers (default: os.cpu_count())")
    ap.add_argument("--seed_budget_factor", type=float, default=5.0,
                    help="Max attempts per algo per ref = ntrials * factor; failures are skipped.")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    ALGOS   = list(args.algos)
    N_TRIAL = int(args.ntrials)
    BASE_SEED = int(args.seed)
    est_freq  = int(args.est_freq)
    window    = int(args.window)
    ref_types = [str(r) for r in args.refs]
    workers   = args.workers
    seed_budget_per_ref = max(N_TRIAL, int(np.ceil(N_TRIAL * float(args.seed_budget_factor))))
    T_sim = 20.0

    # probe time axes (same dt/T for all refs by construction)
    probe = QuadrotorDynamics()
    t_all, _, _ = make_traj_hover(T_sim, probe.dt, np.random.default_rng(0))
    t = t_all
    t_est = t[::est_freq][1:]

    # storage across ALL refs
    store = {
        algo: {"pos": [], "vel": [], "ori": [], "est": [],
               "xm": [] if args.save_traj else None,
               "xr": [] if args.save_traj else None}
        for algo in ALGOS
    }
    # event idx & labels per *trial across refs*
    event_idx_all = []
    traj_labels   = []  # one label per trial (across refs)

    # run per ref type, trials in parallel for each algo
    for ref_type in ref_types:
        print(f"\n=== Running ref_type='{ref_type}' ===")
        # collect successes count so we can pad as needed at the end of each algo for this ref
        successes_per_algo = {algo: 0 for algo in ALGOS}
        # For sharing event_idx: only capture from the first algorithm processed below
        first_algo_events_this_ref = []

        for a_idx, algo in enumerate(ALGOS):
            print(f"\n[algo={algo}] collecting {N_TRIAL} successful trials with up to "
                  f"{seed_budget_per_ref} attempts ({workers} workers)…")
            successes = 0
            attempts = 0
            seed_offset_next = 0

            with ProcessPoolExecutor(max_workers=workers) as ex:
                inflight = {}

                # Prime
                while len(inflight) < workers and attempts < seed_budget_per_ref and successes < N_TRIAL:
                    fut = ex.submit(_trial_worker, algo, BASE_SEED, seed_offset_next,
                                    est_freq, window, ref_type, T_sim)
                    inflight[fut] = seed_offset_next
                    seed_offset_next += 1
                    attempts += 1

                while successes < N_TRIAL and attempts <= seed_budget_per_ref and inflight:
                    done, _ = wait(inflight, return_when=FIRST_COMPLETED)
                    for fut in done:
                        seed_used = inflight.pop(fut)
                        ok, _, payload = fut.result()
                        if ok:
                            out = payload
                            # success => store
                            store[algo]["pos"].append(out["abs_mean_pos_err_t"])
                            store[algo]["vel"].append(out["abs_mean_vel_err_t"])
                            store[algo]["ori"].append(out["abs_mean_ori_err_t"])
                            store[algo]["est"].append(out["est_err_t"])
                            if args.save_traj:
                                store[algo]["xm"].append(out["x_meas_traj"])
                                store[algo]["xr"].append(out["x_ref_traj"])

                            # events (from the FIRST algo only) + labels
                            if a_idx == 0:
                                first_algo_events_this_ref.append(out["event_idx"])
                                traj_labels.append(ref_type)

                            successes += 1
                            successes_per_algo[algo] += 1
                            print(f"[{ref_type} | {algo}] success {successes}/{N_TRIAL} (seed_offset={seed_used})")
                        else:
                            print(f"[{ref_type} | {algo}] failed attempt (seed_offset={seed_used}): {payload}")

                        # Refill
                        if attempts < seed_budget_per_ref and successes < N_TRIAL:
                            fut_new = ex.submit(_trial_worker, algo, BASE_SEED, seed_offset_next,
                                                est_freq, window, ref_type, T_sim)
                            inflight[fut_new] = seed_offset_next
                            seed_offset_next += 1
                            attempts += 1

                # Pad if needed to keep shapes consistent across algos and refs
                if successes < N_TRIAL:
                    print(f"[WARN] {ref_type} | {algo}: only {successes}/{N_TRIAL} successes; padding with NaNs.")
                    Tlen = len(t)
                    pad = N_TRIAL - successes
                    store[algo]["pos"].extend([np.full(Tlen, np.nan)] * pad)
                    store[algo]["vel"].extend([np.full(Tlen, np.nan)] * pad)
                    store[algo]["ori"].extend([np.full(Tlen, np.nan)] * pad)
                    store[algo]["est"].extend([np.full(max(0, len(t_est)), np.nan)] * pad)
                    if args.save_traj:
                        store[algo]["xm"].extend([np.full((Tlen,13), np.nan)] * pad)
                        store[algo]["xr"].extend([np.full((Tlen,13), np.nan)] * pad)
                    # For event/label alignment, also fill labels for missing trials on first algo
                    if a_idx == 0:
                        first_algo_events_this_ref.extend([( -1, -1 )] * pad)
                        traj_labels.extend([ref_type] * pad)

        # After finishing all algos for this ref, extend event list once
        event_idx_all.extend(first_algo_events_this_ref)

    # Convert lists to arrays and save
    event_idx_all = np.asarray(event_idx_all, dtype=int).reshape(-1, 2)
    traj_labels   = np.asarray(traj_labels, dtype=str)
    N_TOTAL = traj_labels.shape[0]

    # pack & save
    save_kwargs = {
        "algos":       np.array(ALGOS),
        "ref_types":   np.array(ref_types, dtype=str),     # unique types used in this run
        "traj_labels": traj_labels,                        # per-trial labels (length N_TOTAL)
        "ntrials":     np.array([N_TOTAL]),
        "base_seed":   np.array([BASE_SEED]),
        "est_freq":    np.array([est_freq]),
        "window":      np.array([window]),
        "t":           t,
        "t_est":       t_est,
        "event_idx":   event_idx_all,                      # (N_TOTAL, 2) from first algo’s trials
        "event_time":  t[np.clip(event_idx_all, 0, len(t)-1)],  # convenience
        "event_labels": np.array(["add", "drop"]),
    }
    for algo in ALGOS:
        save_kwargs[f"{algo}__pos"] = np.stack(store[algo]["pos"], axis=0)
        save_kwargs[f"{algo}__vel"] = np.stack(store[algo]["vel"], axis=0)
        save_kwargs[f"{algo}__ori"] = np.stack(store[algo]["ori"], axis=0)
        save_kwargs[f"{algo}__est"] = np.stack(store[algo]["est"], axis=0)
        if args.save_traj:
            save_kwargs[f"{algo}__xm"] = np.stack(store[algo]["xm"], axis=0)
            save_kwargs[f"{algo}__xr"] = np.stack(store[algo]["xr"], axis=0)

    out_npz = Path(args.outdir) / "results.npz"
    np.savez_compressed(out_npz, **save_kwargs)
    print(f"\nSaved results to: {out_npz.resolve()}")

if __name__ == "__main__":
    main()
