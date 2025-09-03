#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import argparse
import numpy as np
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R

from python.quadrotor.quadrotor import QuadrotorDynamics
from python.common.LQR_controller import LQRController
from python.common.noise_models import *   # apply_noise
from python.common.estimation_methods import *  # RLS, KF, RK, TARK

ALGOS_DEFAULT = ["deka", "rk", "tark","rls", "kf", "gt", "none"] # mgrk, igrk
np.set_printoptions(precision=7, suppress=True)

# ----------------------- Helpers -----------------------
def rel_residual(A, x, b, eps=1e-12):
    r = A @ x - b
    return np.linalg.norm(r) / max(np.linalg.norm(b), eps)

def closest_spd(theta, epsilon=1e-6):
    m, cx, cy, cz, Ixx, Iyy, Izz, Ixy, Ixz, Iyz = theta
    A = np.array([[Ixx, Ixy, Ixz],
                  [Ixy, Iyy, Iyz],
                  [Ixz, Iyz, Izz]])
    A_sym = 0.5*(A + A.T)
    eigvals, eigvecs = np.linalg.eigh(A_sym)
    eigvals = np.clip(eigvals, epsilon, None)
    A_spd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    Ixx_spd, Iyy_spd, Izz_spd = A_spd[0,0], A_spd[1,1], A_spd[2,2]
    Ixy_spd, Ixz_spd, Iyz_spd = A_spd[0,1], A_spd[0,2], A_spd[1,2]
    return np.array([m, cx, cy, cz, Ixx_spd, Iyy_spd, Izz_spd, Ixy_spd, Ixz_spd, Iyz_spd])

def generate_figure8(T, dt, rng, A=None, B=None, z0=None, omega=None):
    A = A if A is not None else rng.uniform(0.4, 0.8)
    B = B if B is not None else rng.uniform(0.2, 0.4)
    z0 = z0 if z0 is not None else rng.uniform(-0.05, 0.05)
    omega_base = 2*np.pi/20.0
    omega = omega if omega is not None else omega_base * rng.uniform(0.8, 1.25)

    t = np.arange(0, T, dt)
    x_ref = A * np.sin(omega * t)
    y_ref = B * np.sin(2*omega * t)
    z_ref = z0 * np.ones_like(t)
    vx_ref = A * omega * np.cos(omega * t)
    vy_ref = 2 * B * omega * np.cos(2*omega * t)
    vz_ref = np.zeros_like(t)
    pos_ref = np.vstack([x_ref, y_ref, z_ref]).T
    vel_ref = np.vstack([vx_ref, vy_ref, vz_ref]).T
    return t, pos_ref, vel_ref

def get_error_state(quad, x, xg):
    pos_err = x[0:3] - xg[0:3]
    phi     = quad.qtorp(x[3:7])
    vel_err = x[7:10] - xg[7:10]
    om_err  = x[10:13] - xg[10:13]
    return np.hstack([pos_err, phi, vel_err, om_err])

# ----------------------- Single Trial -----------------------
def run_single_trial(estimator_option: str, trial_idx: int, base_seed: int,
                     est_freq: int, window: int):
    """
    Returns dict with:
      abs_mean_pos_err_t (T,), abs_mean_vel_err_t (T,), abs_mean_ori_err_t (T,),
      est_err_t (U,), x_meas_traj (T,13), x_ref_traj (T,13),
      t (T,), event_idx (int)
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
    T  = 20.0
    t, pos_ref, vel_ref = generate_figure8(T, dt, rng)

    # Initial state
    x_true = np.zeros(13)
    x_true[0:3]   = pos_ref[0] + 0.1*rng.standard_normal(3)
    phi0          = 0.02*rng.standard_normal(3)
    x_true[3:7]   = quad.rptoq(phi0)
    x_true[7:10]  = vel_ref[0] + 0.01*rng.standard_normal(3)
    x_true[10:13] = 0.01*rng.standard_normal(3)
    x_meas = apply_noise(x_true)

    # Randomized event
    add_payload_step = rng.integers(200, 301)  # timestep index
    drop_payload_step = rng.integers(500, 701)
    payload_m  = rng.uniform(0.01, 0.02)
    payload_dr = rng.uniform(-0.002, 0.002, size=3)

    # Estimator setup
    theta_0 = quad.get_true_inertial_params()
    theta_est  = None
    estimator  = None
    if estimator_option == "rls":
        estimator = RLS(num_params=theta_0.shape[0], theta_hat=theta_0, forgetting_factor=0.7, c=1000)
    elif estimator_option == "kf":
        estimator = KF(num_params=theta_0.shape[0],
                       process_noise=1e-3*np.eye(theta_0.shape[0]),
                       measurement_noise=1e-4*np.eye(window*6),
                       theta_hat=theta_0, c=1000)
    elif estimator_option == "rk":
        estimator = RK(num_params=theta_0.shape[0], x0=theta_0)
    elif estimator_option == "tark":
        estimator = TARK(num_params=theta_0.shape[0], x0=theta_0)
    elif estimator_option == "igrk":
        estimator = IGRK(num_params=theta_0.shape[0], x0=theta_0)
    elif estimator_option == "mgrk":
        estimator = MGRK(num_params=theta_0.shape[0], x0=theta_0)
    elif estimator_option == "rek":
        estimator = REK(num_params=theta_0.shape[0], x0=theta_0)
    elif estimator_option == "deka":
        estimator = DEKA(num_params=theta_0.shape[0], x0=theta_0)

    # Rolling buffers
    A_buf = deque(maxlen=window)
    B_buf = deque(maxlen=window)

    # Storage
    x_meas_traj = []
    x_ref_traj  = []
    est_err_traj = []

    u = ug.copy()
    for k in range(len(t)):
        # goal
        rg, vg = pos_ref[k], vel_ref[k]
        qg, omgg = np.array([1,0,0,0]), np.zeros(3)
        xg = np.hstack([rg, qg, vg, omgg])

        x_ref_traj.append(xg.copy())
        x_meas_traj.append(x_meas.copy())

        # control
        x_err12 = get_error_state(quad, x_meas, xg)
        u = ug + lqr.control(x_err12)

        # propagate + measure
        x_true_next = quad.dynamics_rk4_true(x_true, u, dt=quad.dt)
        x_meas_next = apply_noise(x_true_next)

        # residual
        dx_meas = (x_meas_next - x_meas) / quad.dt
        A_mat = quad.get_data_matrix(x_meas, dx_meas)
        b_vec = quad.get_force_vector(x_meas, dx_meas, u)
        A_buf.append(A_mat); B_buf.append(b_vec)

        # update (skip t=0)
        if (k % est_freq == 0) and (k > 0):
            A_stack = np.vstack(list(A_buf))
            b_stack = np.concatenate(list(B_buf), axis=0)

            if estimator_option == "gt":
                theta_gt = quad.get_true_inertial_params()
                est_err_traj.append(rel_residual(A_stack, theta_gt, b_stack))
                ug = quad.get_hover_thrust_true()
                A_new, B_new = quad.get_linearized_true(xg, ug)
                lqr.update_linearized_dynamics(A_new, B_new)

            elif estimator_option == "lstsq":
                theta_est = np.linalg.lstsq(A_stack, b_stack, rcond=None)[0]
                theta_est[4] = theta_est[5] = np.mean(theta_est[4:6])
                theta_est = closest_spd(theta_est)
                quad.set_estimated_inertial_params(theta_est)
                ug = quad.get_hover_thrust_est()
                A_new, B_new = quad.get_linearized_est(xg, ug)
                lqr.update_linearized_dynamics(A_new, B_new)
                est_err_traj.append(rel_residual(A_stack, theta_est, b_stack))

            elif estimator_option in ("deka", "rk", "tark", "rls", "kf", "mgrk", "igrk", "rek"):
                theta_est = estimator.iterate(A_stack, b_stack)
                theta_est[4] = theta_est[5] = np.mean(theta_est[4:6])
                theta_est = closest_spd(theta_est)
                quad.set_estimated_inertial_params(theta_est)
                ug = quad.get_hover_thrust_est()
                A_new, B_new = quad.get_linearized_est(xg, ug)
                lqr.update_linearized_dynamics(A_new, B_new)
                est_err_traj.append(rel_residual(A_stack, theta_est, b_stack))

            elif estimator_option == "none":
                est_err_traj.append(rel_residual(A_stack, theta_0, b_stack))

            else:
                raise ValueError(f"Unknown estimator option: {estimator_option}")

        # event
        if k == add_payload_step:
            quad.attach_payload(m_p=payload_m, delta_r_p=payload_dr)
        elif k == drop_payload_step:
            quad.detach_payload()

        x_true = x_true_next
        x_meas = x_meas_next

    # Arrays
    x_meas_traj = np.array(x_meas_traj)  # (T,13)
    x_ref_traj  = np.array(x_ref_traj)   # (T,13)

    # Component errors
    pos_err_comp = x_meas_traj[:, 0:3]   - x_ref_traj[:, 0:3]
    vel_err_comp = x_meas_traj[:, 7:10]  - x_ref_traj[:, 7:10]
    qu_meas = np.roll(x_meas_traj[:, 3:7], -1, axis=1)
    qu_ref  = np.roll(x_ref_traj[:,  3:7], -1, axis=1)
    eul_meas = R.from_quat(qu_meas).as_euler('xyz', degrees=True)
    eul_ref  = R.from_quat(qu_ref ).as_euler('xyz', degrees=True)
    orient_err_comp = eul_meas - eul_ref

    abs_mean_pos_err_t = np.mean(np.abs(pos_err_comp),    axis=1)
    abs_mean_vel_err_t = np.mean(np.abs(vel_err_comp),    axis=1)
    abs_mean_ori_err_t = np.mean(np.abs(orient_err_comp), axis=1)

    return {
        "abs_mean_pos_err_t": abs_mean_pos_err_t,
        "abs_mean_vel_err_t": abs_mean_vel_err_t,
        "abs_mean_ori_err_t": abs_mean_ori_err_t,
        "est_err_t":          np.array(est_err_traj),
        "x_meas_traj":        x_meas_traj,
        "x_ref_traj":         x_ref_traj,
        "t":                  t,
        "event_idx":          (int(add_payload_step), int(drop_payload_step)),
    }

# ----------------------- Runner -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="results_param_estimation",
                    help="Directory to save NPZ results")
    ap.add_argument("--ntrials", type=int, default=1)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--algos", type=str, nargs="*", default=ALGOS_DEFAULT)
    ap.add_argument("--est_freq", type=int, default=20, help="Estimator cadence (steps)")
    ap.add_argument("--window",   type=int, default=20, help="Rolling window length (steps)")
    ap.add_argument("--save_traj", action="store_true",
                    help="Save (N,T,13) trajectories for 3D viz")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    ALGOS = list(args.algos)
    N_TRIALS = int(args.ntrials)
    BASE_SEED = int(args.seed)
    est_freq = int(args.est_freq)
    window   = int(args.window)

    # probe time axes
    probe = QuadrotorDynamics()
    T = 20.0
    t = np.arange(0, T, probe.dt)
    t_est = t[::est_freq][1:]  # skip t=0

    # storage
    store = {
        algo: {"pos": [], "vel": [], "ori": [], "est": [],
               "xm": [] if args.save_traj else None,
               "xr": [] if args.save_traj else None}
        for algo in ALGOS
    }
    event_idx_trials = np.full((N_TRIALS, 2), -1, dtype=int)

    for algo in ALGOS:
        print(f"\nRunning {algo} trials...")
        for trial in tqdm(range(N_TRIALS), desc=f"{algo}", leave=False):
            out = run_single_trial(algo, trial_idx=trial, base_seed=BASE_SEED,
                                   est_freq=est_freq, window=window)
            store[algo]["pos"].append(out["abs_mean_pos_err_t"])
            store[algo]["vel"].append(out["abs_mean_vel_err_t"])
            store[algo]["ori"].append(out["abs_mean_ori_err_t"])
            store[algo]["est"].append(out["est_err_t"])
            if args.save_traj:
                store[algo]["xm"].append(out["x_meas_traj"])
                store[algo]["xr"].append(out["x_ref_traj"])

            # record event index once (shared across algos for same trial/seed)
            if event_idx_trials[trial, 0] == -1:
                # out["event_idx"] is a tuple: (add_step, drop_step)
                event_idx_trials[trial] = out["event_idx"]

    # pack & save
    save_kwargs = {
        "algos": np.array(ALGOS),
        "ntrials": np.array([N_TRIALS]),
        "base_seed": np.array([BASE_SEED]),
        "est_freq": np.array([est_freq]),
        "window": np.array([window]),
        "t": t,
        "t_est": t_est,
        "event_idx": event_idx_trials,                  # shape (N_TRIALS, 2)
        "event_time": t[event_idx_trials],              # shape (N_TRIALS, 2)
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

    out_npz = outdir / "results.npz"
    np.savez_compressed(out_npz, **save_kwargs)
    print(f"\nSaved results to: {out_npz.resolve()}")

if __name__ == "__main__":
    main()
