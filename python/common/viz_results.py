import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from typing import Literal, Dict, Any, Tuple


def visualize_3d_traj(
    result: Dict[str, Any],
    quat_fmt: Literal["wxyz", "xyzw"] = "wxyz",
    n_arrows: int = 50,
    arrow_length: float = 0.05,
    figsize=(8, 6),
    title: str | None = None,
):
    """
    Minimal 3D visualizer for measured vs. reference trajectory.

    Args:
        result: dict containing at least:
            - 'x_meas_traj' : (T, 13) array [pos(3), quat(4), lin vel(3), ang vel(3)]
            - 'x_ref_traj'  : (T, 13) or (T, 3) (if only positions for ref)
        quat_fmt: 'wxyz' (default) or 'xyzw' for quaternion ordering
        n_arrows: number of uniformly downsampled orientation arrows (measured)
        arrow_length: length of each orientation arrow (world units)
        figsize: matplotlib figure size
        title: optional plot title
    """
    x_meas = np.asarray(result["x_meas_traj"])
    x_ref  = np.asarray(result["x_ref_traj"])

    pos_m = x_meas[:, 0:3]
    pos_r = x_ref[:, 0:3]  # works if 3D or full state

    quat_m = x_meas[:, 3:7]

    # Quaternion helpers
    def q_conj(q): return np.array([q[0], -q[1], -q[2], -q[3]])
    def q_mul(q1, q2):
        w1,x1,y1,z1 = q1; w2,x2,y2,z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
    def to_wxyz(q):
        return q if quat_fmt=="wxyz" else np.array([q[3], q[0], q[1], q[2]])
    def q_normalize(q): return q/np.linalg.norm(q) if np.linalg.norm(q)>0 else np.array([1,0,0,0])
    def rotate_by_quat(v, q):
        q = q_normalize(to_wxyz(q))
        return q_mul(q_mul(q, np.hstack([[0], v])), q_conj(q))[1:]

    # Arrow indices
    T = pos_m.shape[0]
    idxs = np.unique(np.linspace(0, T-1, min(n_arrows, T), dtype=int)) if T>0 else []

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Plot ref + meas
    ax.plot(pos_r[:,0], pos_r[:,1], pos_r[:,2], 'k:', label='ref')
    ax.plot(pos_m[:,0], pos_m[:,1], pos_m[:,2], 'b-', label='meas')

    # Orientation arrows (local z-axis)
    z_local = np.array([0.,0.,1.])
    for i in idxs:
        z_world = rotate_by_quat(z_local, quat_m[i])
        p = pos_m[i]
        ax.quiver(p[0],p[1],p[2], z_world[0],z_world[1],z_world[2],
                  length=arrow_length, normalize=True, color='r', linewidth=0.8)

    # Mark start and end
    if T > 0:
        start = pos_m[0]; end = pos_m[-1]
        ax.scatter(*start, c='g', s=50, marker='o', label="start")
        ax.text(*start, "START", color='g')
        ax.scatter(*end, c='r', s=50, marker='X', label="end")
        ax.text(*end, "END", color='r')

    # Equal axes
    def set_axes_equal(ax):
        xlim, ylim, zlim = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
        ranges = [abs(xlim[1]-xlim[0]), abs(ylim[1]-ylim[0]), abs(zlim[1]-zlim[0])]
        max_range = max(ranges) or 1.0
        mid = [np.mean(xlim), np.mean(ylim), np.mean(zlim)]
        ax.set_xlim(mid[0]-max_range/2, mid[0]+max_range/2)
        ax.set_ylim(mid[1]-max_range/2, mid[1]+max_range/2)
        ax.set_zlim(mid[2]-max_range/2, mid[2]+max_range/2)
    set_axes_equal(ax)

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    if title: ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()

def visualize_residual_errors(result, eps=1e-12, logy=False):
    """
    Plot absolute and relative residual errors for θ_gt and θ_est:
      r = A @ θ - b

    Handles NaNs by ignoring them in norms.
    """
    A     = np.array(result["A_snapshots"])
    b     = np.array(result["b_snapshots"])
    th_gt = np.array(result["theta_gt_traj"])
    th_es = np.array(result["theta_est_traj"])
    t_hi  = np.array(result["t"])

    K = th_gt.shape[0]
    est_freq = max(1, int(round(len(t_hi) / (K + 1))))
    t_resid = t_hi[est_freq: (K + 1) * est_freq : est_freq]

    def safe_norm(x, axis=1):
        return np.linalg.norm(np.nan_to_num(x), axis=axis)

    r_gt = (A @ th_gt[..., None]).squeeze(-1) - b
    r_es = (A @ th_es[..., None]).squeeze(-1) - b

    abs_gt = safe_norm(r_gt)
    abs_es = safe_norm(r_es)
    bnorm  = np.maximum(safe_norm(b), eps)
    rel_gt = abs_gt / bnorm
    rel_es = abs_es / bnorm

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    ax1.plot(t_resid, abs_gt, label="||Aθ_gt − b||")
    ax1.plot(t_resid, abs_es, label="||Aθ_est − b||")
    ax1.set_ylabel("Absolute")
    ax1.legend(); ax1.grid(True, ls=":")

    ax2.plot(t_resid, rel_gt, label="rel_gt")
    ax2.plot(t_resid, rel_es, label="rel_est")
    ax2.set_ylabel("Relative"); ax2.set_xlabel("Time")
    ax2.legend(); ax2.grid(True, ls=":")

    if logy:
        ax1.set_yscale("log"); ax2.set_yscale("log")

    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Sequence, Dict, List, Optional

def plot_error_cdfs(
    npz_paths: Sequence[str | Path],
    algos: Optional[Sequence[str]] = None,   # order of legend; if None, infer from files
    figsize=(11, 4),
    title: Optional[str] = None,
    savepath: Optional[str | Path] = None,
):
    """
    Plot CDFs of per-trial position/velocity/orientation errors for all algorithms
    across one or more .npz files (e.g., different noise levels).

    Assumptions about .npz layout (per your script):
      - For each algo name `a`, arrays exist:
          f"{a}__pos"  -> shape (N_trials, T)
          f"{a}__vel"  -> shape (N_trials, T)
          f"{a}__ori"  -> shape (N_trials, T)
        where each row is abs-mean error over time *per timestamp*. We reduce to
        a single scalar per trial via time nanmean again.

    Args:
        npz_paths: list of result files to include (concatenated).
        algos: optional explicit list/order of algorithms to show.
        figsize: matplotlib figure size.
        title: optional suptitle.
        savepath: if given, save the figure.

    Returns:
        fig, axes (axes is [pos_ax, vel_ax, ori_ax])
    """
    # ---------- load & collect ----------
    data: Dict[str, Dict[str, List[np.ndarray]]] = {}  # algo -> {"pos":[...], "vel":[...], "ori":[...]}
    seen_algos: List[str] = []

    for p in npz_paths:
        with np.load(p, allow_pickle=True) as Z:
            # discover algos if not provided
            file_algos = list(map(str, Z["algos"])) if "algos" in Z else []
            if not file_algos:  # fallback: infer from keys
                for k in Z.files:
                    if k.endswith("__pos"):
                        file_algos.append(k[:-5])
                file_algos = sorted(set(file_algos))
            for a in file_algos:
                pos_k, vel_k, ori_k = f"{a}__pos", f"{a}__vel", f"{a}__ori"
                if pos_k not in Z.files or vel_k not in Z.files or ori_k not in Z.files:
                    continue
                if a not in data:
                    data[a] = {"pos": [], "vel": [], "ori": []}
                    seen_algos.append(a)
                data[a]["pos"].append(Z[pos_k])  # (N,T)
                data[a]["vel"].append(Z[vel_k])  # (N,T)
                data[a]["ori"].append(Z[ori_k])  # (N,T)

    if algos is None:
        algos = seen_algos  # keep discovery order

    # ---------- reduce to trial-level scalars & concatenate across files ----------
    def trial_scalars(stacks: List[np.ndarray]) -> np.ndarray:
        if not stacks:
            return np.array([])
        X = np.concatenate(stacks, axis=0)  # (sum_N, T)
        # Each trial -> one scalar (time-averaged absolute error)
        with np.errstate(invalid="ignore", divide="ignore"):
            scalars = np.nanmean(X, axis=1)  # (sum_N,)
        scalars = scalars[~np.isnan(scalars)]
        return scalars

    scalars_by_algo = {
        a: {
            "pos": trial_scalars(data.get(a, {}).get("pos", [])),
            "vel": trial_scalars(data.get(a, {}).get("vel", [])),
            "ori": trial_scalars(data.get(a, {}).get("ori", [])),
        }
        for a in algos
    }

    # ---------- color map (consistent across plots) ----------
    # Use a fixed palette; extend if many algos
    base_colors = plt.get_cmap("tab10").colors + plt.get_cmap("tab20").colors
    color_map = {a: base_colors[i % len(base_colors)] for i, a in enumerate(algos)}

    # ---------- small helper to draw a CDF subplot ----------
    def add_cdf(ax, field: str, xlabel: str):
        for a in algos:
            x = scalars_by_algo[a][field]
            if x.size == 0:
                continue
            x_sorted = np.sort(x)
            y = np.linspace(0, 100, len(x_sorted), endpoint=True)  # percent of trials
            ax.plot(x_sorted, y, label=a, color=color_map[a], linewidth=1.8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("% of trials ≤ x")
        ax.grid(True, linestyle=":", linewidth=0.7)

    # ---------- plot ----------
    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)
    add_cdf(axes[0], "pos", "Position error")
    add_cdf(axes[1], "vel", "Velocity error")
    add_cdf(axes[2], "ori", "Orientation error")
    axes[0].legend(title="Algorithm", fontsize=9)
    if title:
        fig.suptitle(title)

    if savepath is not None:
        fig.savefig(savepath, dpi=200)
    return fig, axes
