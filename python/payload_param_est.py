#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import uniform_filter1d
from python.quadrotor.quadrotor import QuadrotorDynamics
from python.common.LQR_controller import LQRController
from python.common.noise_models import *   # for apply_noise
from python.common.estimation_methods import RLS, KF, RK

np.random.seed(42)

def regularize_inertial_params(theta, eps=1e-10):
    """
    Project an estimated inertial parameter vector onto the
    physically-consistent set.

    Args:
      theta (array-like, shape=(10,)):
        [m, m*cx, m*cy, m*cz, Ixx, Iyy, Izz, Ixy, Ixz, Iyz]
      eps (float): small positive floor for eigenvalues and mass.

    Returns:
      theta_reg (np.ndarray, shape=(10,)):
        the regularized parameter vector.
    """
    # 1) Unpack
    m      = float(theta[0])
    first  = np.array(theta[1:4], dtype=float)  # m * [cx, cy, cz]
    I_vec  = theta[4:]                          # [Ixx, Iyy, Izz, Ixy, Ixz, Iyz]

    # 2) Ensure mass > eps
    m_reg = max(m, eps)

    # 3) Compute COM position r = first / m
    r = first / m_reg

    # 4) Build the inertia matrix and symmetrize
    I_est = np.array([
        [I_vec[0], I_vec[3], I_vec[4]],
        [I_vec[3], I_vec[1], I_vec[5]],
        [I_vec[4], I_vec[5], I_vec[2]],
    ], dtype=float)
    B = 0.5*(I_est + I_est.T)

    # 5) Nearest SPD via eigen‑clamp (Higham’s method)
    eigvals, Q = np.linalg.eigh(B)
    eigvals_clamped = np.clip(eigvals, eps, None)
    I_spd = (Q * eigvals_clamped) @ Q.T

    # 6) Enforce COM‑ellipsoid constraint:
    #    require m_reg * ||r||^2 <= λ_min(I_spd)
    lam_min = eigvals_clamped.min()
    r_norm2 = r.dot(r)
    if r_norm2 > 0:
        max_r2 = lam_min / m_reg
        if r_norm2 > max_r2:
            scale = np.sqrt(max_r2 / r_norm2)
            r = r * scale

    # 7) Repack
    first_reg = m_reg * r
    Ixx, Iyy, Izz = I_spd[0,0], I_spd[1,1], I_spd[2,2]
    Ixy, Ixz, Iyz = I_spd[0,1], I_spd[0,2], I_spd[1,2]

    theta_reg = np.hstack([m_reg, first_reg, [Ixx, Iyy, Izz, Ixy, Ixz, Iyz]])
    return theta_reg

def compute_tracking_error(x_all, x_ref_all):
    """Compute L2 position error over the trajectory"""
    errors = [np.linalg.norm(x[0:3] - x_ref[0:3])
              for x, x_ref in zip(x_all, x_ref_all)]
    return np.mean(errors), np.max(errors), np.array([x[0:3] - x_ref[0:3]
                                                       for x, x_ref in zip(x_all, x_ref_all)])

def generate_figure8(T, dt, A=0.5, B=0.25, z0=0.0, omega=2*np.pi/10):
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

def main():
    estimator_options = "rls" # gt, none, rk, rls, kf
    assert estimator_options in ["gt", "none", "rk", "rls", "kf"], \
        "Invalid estimator option. Choose from 'gt', 'none', 'rk', 'rls', 'kf'."

    quad = QuadrotorDynamics()
    ug   = quad.hover_thrust_true

    # LQR around hover
    xg0 = np.hstack([np.zeros(3), np.array([1,0,0,0]), np.zeros(6)])
    A, B = quad.get_linearized_true(xg0, ug)
    max_dev_x = np.array([0.1,0.1,0.1,  0.5,0.5,0.05,  0.5,0.5,0.5,  0.7,0.7,0.2])
    max_dev_u = np.array([0.5,0.5,0.5,0.5])
    Q = np.diag(1.0/max_dev_x**2)
    Rmat = np.diag(1.0/max_dev_u**2)
    lqr = LQRController(A, B, Q, Rmat)

    # Trajectory
    dt = quad.dt; T = 20.0
    t, pos_ref, vel_ref = generate_figure8(T, dt,
                                           A=0.6, B=0.3, z0=0.0,
                                           omega=2*np.pi/20)

    # Initial true state
    x_true = np.zeros(13)
    x_true[0:3] = pos_ref[0] + 0.05*np.random.randn(3)
    phi0 = 0.02*np.random.randn(3)
    x_true[3:7] = quad.rptoq(phi0)
    x_true[7:10] = vel_ref[0] + 0.01*np.random.randn(3)
    x_true[10:13] = 0.01*np.random.randn(3)

    # Measurement: first sample
    x_meas = apply_noise(x_true)

    # Storage
    x_true_traj = []
    x_meas_traj = []
    x_ref_traj  = []
    meas_noise  = []
    gt_err_traj  = []   # residual using true parameters
    est_err_traj = []   # residual using estimator (if any)

    # Estimator
    theta = quad.get_true_inertial_params()
    theta_est = None
    _have_estimator = False
    if estimator_options == "rls":
        estimator = RLS(num_params=theta.shape[0], theta_hat=theta, forgetting_factor=0.95, c=1000)
        _have_estimator = True
    elif estimator_options == "kf":
        estimator = KF(num_params=theta.shape[0], process_noise=1e-4*np.eye(theta.shape[0]),
                       measurement_noise=1e-4*np.eye(3), theta_hat=theta, c=1000)
        _have_estimator = True
    elif estimator_options == "rk":
        estimator = RK(num_params=theta.shape[0], x0=theta, c=1000)
        _have_estimator = True

    event_step = 500

    # Simulation loop
    for k in range(len(t)):
        # goal state
        rg, vg = pos_ref[k], vel_ref[k]
        qg, omgg = np.array([1,0,0,0]), np.zeros(3)
        xg = np.hstack([rg, qg, vg, omgg])

        # record
        x_true_traj.append(x_true.copy())
        x_meas_traj.append(x_meas.copy())
        x_ref_traj.append(xg.copy())

        # control on measured
        x_err12 = get_error_state(quad, x_meas, xg)
        u = ug + lqr.control(x_err12)

        # true propagation
        x_true_next = quad.dynamics_rk4_true(x_true, u, dt=dt)
            
        # measurement noise
        x_meas_next = apply_noise(x_true_next)
        noise_vec   = x_meas_next - x_true_next

        # force residual on measured
        dx_meas = (x_meas_next - x_meas)/dt
        A_mat = quad.get_data_matrix(x_meas, dx_meas)
        A_norm = np.linalg.norm(A_mat, ord='fro')
        A_mat = A_mat / A_norm
        b_vec = quad.get_force_vector(x_meas, u) + np.random.normal(0, 3e-4, size=(A_mat.shape[0],))
        b_vec = b_vec / A_norm

        theta_gt   = quad.get_true_inertial_params()
        err_gt     = np.linalg.norm(A_mat @ theta_gt - b_vec)
        gt_err_traj.append(err_gt)

        if _have_estimator:
            theta_est = estimator.iterate(A_mat, b_vec)
            theta_est = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]  # alternative method
            theta_est = regularize_inertial_params(theta_est)
            err_est   = np.linalg.norm(A_mat @ theta_est - b_vec)
            est_err_traj.append(err_est)

        meas_noise.append(noise_vec.copy())

        print(f"Step {k+1}/{len(t)}: "
                f"True pos: {x_true_next[0:3]}, "
                f"Measured pos: {x_meas_next[0:3]}, "
                f"GT error: {err_gt:.4f}, "
                f"Est error: {err_est:.4f}" if _have_estimator else "")

        # perturb theta
        if k == event_step:
            # import pdb; pdb.set_trace()  # Debugging breakpoint
            quad.attach_payload(m_p=0.0005, delta_r_p=np.array([0.02, 0.01, 0.0]))
            if estimator_options == "none":
                # use estimated parameters
                theta_est = quad.get_estimated_inertial_params()            # choose gt or estimated theta
                ug = quad.get_hover_thrust_est()
                A_new, B_new = quad.get_linearized_est(xg, ug)
            elif estimator_options == "gt":
                # use true parameters
                theta_gt = quad.get_true_inertial_params()
                ug = quad.get_hover_thrust_true()
                A_new, B_new = quad.get_linearized_true(xg, ug)
            else:
                # theta_gt = quad.get_true_inertial_params()
                quad.set_estimated_inertial_params(theta_est)
                ug = quad.get_hover_thrust_est()
                A_new, B_new = quad.get_linearized_est(xg, ug)

            lqr.update_linearized_dynamics(A_new, B_new)
            print(f"Payload attached at t={t[k]:.2f}s, new theta:\n{theta}")

        # step
        x_true = x_true_next
        x_meas = x_meas_next

    # to arrays
    x_true_traj = np.array(x_true_traj)
    x_meas_traj = np.array(x_meas_traj)
    x_ref_traj  = np.array(x_ref_traj)
    meas_noise  = np.array(meas_noise)
    gt_err_traj  = np.array(gt_err_traj)
    est_err_traj = np.array(est_err_traj) if _have_estimator else None

    # compute errors
    pos_err_comp = x_meas_traj[:,0:3] - x_ref_traj[:,0:3]
    vel_err_comp = x_meas_traj[:,7:10] - x_ref_traj[:,7:10]

    # orientation errors in Euler (deg)
    qu_meas = np.roll(x_meas_traj[:,3:7], -1, axis=1)
    qu_ref  = np.roll(x_ref_traj[:,3:7], -1, axis=1)
    eul_meas = R.from_quat(qu_meas).as_euler('xyz', degrees=True)
    eul_ref  = R.from_quat(qu_ref).as_euler('xyz', degrees=True)
    orient_err_comp = eul_meas - eul_ref

    # measurement error norms
    pos_me_noise = np.linalg.norm(meas_noise[:,0:3], axis=1)
    vel_me_noise = np.linalg.norm(meas_noise[:,7:10], axis=1)
    # orientation measurement error magnitude (deg)
    qu_true = np.roll(x_true_traj[:,3:7], -1, axis=1)
    rot_err    = (R.from_quat(qu_meas) * R.from_quat(qu_true).inv()).as_rotvec()
    rot_me_noise = np.linalg.norm(rot_err, axis=1) * 180/np.pi
    angv_me_noise = np.linalg.norm(meas_noise[:,10:13], axis=1)

    def moving_average(x, window_size=20):
        return np.convolve(x, np.ones(window_size)/window_size, mode='same')

    # --- Plots ---
    # 1) Component-wise errors
    fig, axs = plt.subplots(3,1, figsize=(9,7), sharex=True)
    axs[0].plot(t, pos_err_comp); axs[0].set_ylabel('Pos error [m]'); axs[0].legend(('x','y','z')); axs[0].grid(True); axs[0].axvline(event_step * quad.dt, color='red', linestyle='--')
    axs[1].plot(t, vel_err_comp); axs[1].set_ylabel('Vel error [m/s]'); axs[1].legend(('v_x','v_y','v_z')); axs[1].grid(True); axs[1].axvline(event_step * quad.dt, color='red', linestyle='--')
    axs[2].plot(t, uniform_filter1d(orient_err_comp, size=3, axis=1)); axs[2].set_ylabel('Ori error [deg]'); axs[2].legend(('roll','pitch','yaw')); axs[2].set_xlabel('Time [s]'); axs[2].grid(True); axs[2].axvline(event_step * quad.dt, color='red', linestyle='--')
    plt.tight_layout()
    plt.show()

    # 2) 3d Vis
    fig = plt.figure(figsize=(14,6))
    ax1 = fig.add_subplot(1,2,1, projection='3d')
    ax1.plot(x_ref_traj[:,0], x_ref_traj[:,1], x_ref_traj[:,2], 'g--', label='Reference 8')
    ax1.plot(x_meas_traj[:,0],     x_meas_traj[:,1],     x_meas_traj[:,2],     'b',   label='Tracked')
    ax1.set_title('3D Figure‑8 Tracking')
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    ax1.legend()

    # error vs time
    # ax2 = fig.add_subplot(1,3,2)
    # ax2.plot(t, pos_me_noise, label='pos noise [m]')
    # ax2.plot(t, rot_me_noise, label='rot noise [deg]')
    # ax2.plot(t, vel_me_noise, label='vel noise [m/s]')
    # ax2.plot(t, angv_me_noise, label='angvel noise [rad/s]')
    # ax2.set_title('Measurement Noise Norms')
    # ax2.set_xlabel('Time [s]'); ax2.set_ylabel('Noise magnitude')
    # ax2.legend(); ax2.grid(True)

    # A*theta - b error norm
    ax2 = fig.add_subplot(1,2,2)
    ax2.set_title('Normalized Force Residuals (||Aθ - b|| / ||A||)')
    ax2.plot(t, moving_average(gt_err_traj), lw=1.5, color='crimson', label='GT residual')
    if _have_estimator:
        ax2.plot(t, moving_average(est_err_traj), lw=1.5, color='navy',   label='Est residual')
    ax2.legend()
    ax2.set_xlabel('Time [s]'); ax2.set_ylabel('Residual Norm')
    ax2.set_yscale('log')  # Use logarithmic scale for y-axis
    ax2.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
