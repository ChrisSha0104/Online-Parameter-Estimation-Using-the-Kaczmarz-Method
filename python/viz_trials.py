#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ============ small utils ============

def _safe(y, eps=1e-12):
    return np.clip(y, eps, None)

def _sanitize(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(name))

def set_axes_equal_3d(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = x_limits[1]-x_limits[0]
    y_range = y_limits[1]-y_limits[0]
    z_range = z_limits[1]-z_limits[0]
    max_range = max([x_range, y_range, z_range]) / 2.0
    x_mid = np.mean(x_limits)
    y_mid = np.mean(y_limits)
    z_mid = np.mean(z_limits)
    ax.set_xlim3d([x_mid - max_range, x_mid + max_range])
    ax.set_ylim3d([y_mid - max_range, y_mid + max_range])
    ax.set_zlim3d([z_mid - max_range, z_mid + max_range])


# ============ grouping by trajectory ============

def get_traj_groups(data):
    """
    Returns (labels, groups) where:
      labels: list[str] of trajectory labels
      groups: list[np.ndarray[int]] indices for each traj label

    Accepts any of:
      - 'traj_labels'  (preferred)
      - 'traj_types'
    Falls back to a single group if absent.
    """
    files = set(data.files)
    if "ref_type" in files:
        raw = np.array(data["ref_type"]).astype(str)
    elif "traj_types" in files:
        raw = np.array(data["traj_types"]).astype(str)
    else:
        # no traj labels saved -> single group covering all trials
        # infer N from any metric array
        # pick the first metric we can find
        for k in files:
            if k.endswith("__pos"):
                N = data[k].shape[0]
                break
        else:
            # last resort: infer N from event_idx
            ev = np.array(data["event_idx"])
            N = ev.shape[0] if ev.ndim >= 1 else int(data["ntrials"][0])
        return (["all"], [np.arange(N, dtype=int)])

    # group unique labels in stable order
    labels = []
    groups = []
    for lab in np.unique(raw):
        idx = np.flatnonzero(raw == lab).astype(int)
        if idx.size > 0:
            labels.append(lab)
            groups.append(idx)
    return labels, groups


# ============ classic time series (split by traj) ============

def plot_timeseries_grouped(data, outdir):
    ALGOS = [str(a) for a in data["algos"]]
    N_TRIALS = int(data["ntrials"][0])
    t = data["t"]; t_est = data["t_est"]

    titles_y = {
        "pos": ("Abs mean pos err [m]",       "abs_mean_pos_error_log.png"),
        "vel": ("Abs mean vel err [m/s]",     "abs_mean_vel_error_log.png"),
        "ori": ("Abs mean ori err [deg]",     "abs_mean_orientation_error_log.png"),
    }
    metric_keys = ("pos", "vel", "ori")

    traj_labels, traj_groups = get_traj_groups(data)
    n_traj = len(traj_labels)

    # One figure per metric, with subplots for traj types
    for mk in metric_keys:
        fig, axes = plt.subplots(
            1, n_traj, figsize=(6*n_traj, 5), sharey=True
        )
        if n_traj == 1:
            axes = [axes]

        for ax, lab, idxs in zip(axes, traj_labels, traj_groups):
            for algo in ALGOS:
                arr = data[f"{algo}__{mk}"]  # shape (N,T)
                # subset trials for this traj group
                E = arr[idxs, :]
                mean_curve = np.nanmean(E, axis=0)
                ax.plot(t, _safe(mean_curve), label=algo)
            ax.grid(True, which="both")
            ax.set_xlabel("Time [s]")
            ax.set_yscale("log")
            ax.set_title(f"{mk.upper()} — traj: {lab}")
            if ax is axes[0]:
                ax.set_ylabel(titles_y[mk][0])
            ax.legend()

        fig.suptitle(f"{mk.upper()} error (avg over dims & trials) — split by trajectory", y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        out = outdir / f"{_sanitize(mk)}_timeseries_by_traj.png"
        fig.savefig(out, dpi=200); plt.close(fig)
        print(f"Saved: {out.resolve()}")

    # Residual plot (same grouping but uses t_est)
    fig, axes = plt.subplots(1, n_traj, figsize=(6*n_traj, 5), sharey=True)
    if n_traj == 1:
        axes = [axes]
    for ax, lab, idxs in zip(axes, traj_labels, traj_groups):
        for algo in ALGOS:
            est = data[f"{algo}__est"]  # shape (N, U)
            E = est[idxs, :]
            mean_curve = np.nanmean(E, axis=0)
            ax.plot(t_est, _safe(mean_curve), label=algo)
        ax.set_yscale("log"); ax.grid(True, which="both")
        ax.set_xlabel("Time [s]")
        ax.set_title(f"Residual — traj: {lab}")
        if ax is axes[0]:
            ax.set_ylabel("Relative residual")
        ax.legend()

    fig.suptitle(f"Residual on Ax≈b (avg over trials) — split by trajectory", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = outdir / "residual_by_traj.png"
    fig.savefig(out, dpi=200); plt.close(fig)
    print(f"Saved: {out.resolve()}")


# ============ 3D trajectory (unchanged, per-trial) ============

def plot_3d_single(data, outdir, algo=None, trial_idx=0):
    ALGOS = [str(a) for a in data["algos"]]
    algo = algo if algo is not None else ALGOS[0]
    if f"{algo}__xm" not in data or f"{algo}__xr" not in data:
        raise RuntimeError("Trajectories not saved. Re-run run_trials.py with --save_traj.")
    xm = data[f"{algo}__xm"]; xr = data[f"{algo}__xr"]
    N, T, _ = xm.shape
    if not (0 <= trial_idx < N): raise IndexError(f"trial_idx out of range: 0..{N-1}")
    xm_traj = xm[trial_idx]; xr_traj = xr[trial_idx]

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(xr_traj[:,0], xr_traj[:,1], xr_traj[:,2], 'g--', label='Reference 8')
    ax.plot(xm_traj[:,0], xm_traj[:,1], xm_traj[:,2], 'b',   label='Tracked')
    ax.set_title(f'3D Tracking ({algo}, trial {trial_idx})')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.legend(); set_axes_equal_3d(ax)
    plt.tight_layout(); out = outdir / f"traj3d_{_sanitize(algo)}_trial{trial_idx}.png"
    plt.savefig(out, dpi=200); plt.close(); print(f"Saved 3D plot: {out.resolve()}")


# ============ Event-based ECDF (split by traj; 2 panels per traj: add/drop) ============

def plot_event_error_ecdf_grouped(
    data,
    outdir,
    metrics=("pos", "vel", "ori"),
    window_sec=5.0,
    xmax=(0.05, 0.05),
    stat="mean",
):
    """
    For each trajectory type (row), and for each event (col: add/drop),
    compute ECDF of an aggregated error over [event+1, event+window_sec].
    Aggregation per trial = mean/median/max/p95/end of that window.
    """
    ALGOS = [str(a) for a in data["algos"]]
    t = data["t"]; dt = float(t[1]-t[0]) if len(t)>1 else 0.01
    T = len(t)
    eps = 1e-12

    # event matrix
    ev_raw = data["event_idx"].astype(int)
    if ev_raw.ndim == 1:
        events_all = ev_raw.reshape(-1, 1)
        event_labels = ["event"]
    else:
        events_all = ev_raw  # (N,2)
        event_labels = [str(x) for x in (data["event_labels"] if "event_labels" in data.files else np.array(["add","drop"]))]

    traj_labels, traj_groups = get_traj_groups(data)
    n_traj = len(traj_labels)
    n_events = events_all.shape[1]

    # parse xmax per event
    if isinstance(xmax, (list, tuple, np.ndarray)):
        xmax_per_event = list(xmax) + [xmax[-1]]*(n_events - len(xmax))
        xmax_per_event = [float(v) for v in xmax_per_event[:n_events]]
    else:
        xmax_per_event = [float(xmax)] * n_events

    w_steps = max(1, int(np.ceil(window_sec / dt)))

    titles = {"pos":"Position error [m]",
              "vel":"Velocity error [m/s]",
              "ori":"Orientation error [deg]"}

    def reduce_window(arr):
        if arr.size == 0: return np.nan
        if stat == "mean":   return float(np.mean(arr))
        if stat == "median": return float(np.median(arr))
        if stat == "max":    return float(np.max(arr))
        if stat == "p95":    return float(np.percentile(arr, 95))
        if stat == "end":    return float(arr[-1])
        return float(np.mean(arr))

    for metric in metrics:
        # grid: rows = traj types, cols = events (add/drop)
        fig, axes = plt.subplots(
            n_traj, n_events, figsize=(6*n_events, 4*n_traj), sharey='row'
        )
        if n_traj == 1 and n_events == 1:
            axes = np.array([[axes]])
        elif n_traj == 1:
            axes = np.array([axes])
        elif n_events == 1:
            axes = axes[:, None]

        for r, (lab, idxs) in enumerate(zip(traj_labels, traj_groups)):
            ev_grp = events_all[idxs, :]  # (Ng, n_events)

            for c in range(n_events):
                ax = axes[r, c]
                this_xmax = xmax_per_event[c]

                for algo in ALGOS:
                    E_all = data[f"{algo}__{metric}"]  # (N,T)
                    E = E_all[idxs, :]                 # (Ng,T)
                    Ng = E.shape[0]
                    vals = []
                    for i in range(Ng):
                        ev = int(ev_grp[i, c])
                        if ev < 0 or ev >= T-1: 
                            continue
                        start = min(ev + 1, T)
                        end   = min(T, start + w_steps)
                        if start >= end:
                            continue
                        v = reduce_window(E[i, start:end])
                        if np.isfinite(v): vals.append(v)

                    if not vals:
                        continue
                    xs = np.sort(np.array(vals))
                    ys = (np.arange(1, xs.size+1) / xs.size) * 100.0  # percentage
                    ys = np.clip(ys, 0.0, 100.0)
                    ax.step(xs, ys, where="post", label=f"{algo} (N={len(vals)})")

                if r == n_traj - 1:
                    ax.set_xlabel("Error")
                if c == 0:
                    ax.set_ylabel("Trials ≤ x  [%]")
                ax.set_xlim(0.0, this_xmax)
                ax.set_ylim(0.0, 100.0)
                ax.set_title(f"{titles[metric]} — {event_labels[c]}  |  traj: {lab}")
                ax.grid(True, which="both")
                ax.legend(fontsize=8)

        fig.suptitle(
            f"Event-based ECDF — {titles[metric]}  (window={window_sec:.2f}s, stat={stat})",
            y=0.995
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        out = outdir / f"ecdf_event_error_{_sanitize(metric)}_by_traj_win{window_sec:.2f}s_{stat}.png"
        fig.savefig(out, dpi=200); plt.close(fig)
        print(f"Saved: {out.resolve()}")


# ============ Full-trajectory ECDF (split by traj) ============

def plot_error_ecdf_fulltraj_grouped(
    data, outdir,
    metrics=("pos","vel","ori"),
    mode="samples",           # "samples","trial_mean","trial_median","trial_p95","trial_max"
    tmin=None,                # seconds; None = from start
    tmax=None,                # seconds; None = to end
    stride=1,                 # subsample timesteps
    xlog=False,
    xmax_per_metric=None,     # dict like {"pos":0.03,"vel":0.15,"ori":2.0}
):
    ALGOS = [str(a) for a in data["algos"]]
    t = data["t"]; T = len(t)
    dt = float(t[1]-t[0]) if len(t)>1 else 0.01

    # time window → indices
    k0 = 0 if tmin is None else max(0, int(np.floor(tmin / dt)))
    k1 = T if tmax is None else min(T, int(np.ceil(tmax / dt)))
    if k1 <= k0: k0, k1 = 0, T

    def reduce_trial(arr_2d):
        # arr_2d: (N_trials_group, K)
        if mode == "samples":
            return arr_2d[:, ::stride].ravel()
        if mode == "trial_mean":
            return np.nanmean(arr_2d[:, ::stride], axis=1)
        if mode == "trial_median":
            return np.nanmedian(arr_2d[:, ::stride], axis=1)
        if mode == "trial_p95":
            return np.nanpercentile(arr_2d[:, ::stride], 95, axis=1)
        if mode == "trial_max":
            return np.nanmax(arr_2d[:, ::stride], axis=1)
        raise ValueError(f"Unknown mode: {mode}")

    titles = {"pos":"Position error [m]", "vel":"Velocity error [m/s]", "ori":"Orientation error [deg]"}
    label_mode = {"samples":"(all samples)", "trial_mean":"(per-trial mean)",
                  "trial_median":"(per-trial median)", "trial_p95":"(per-trial p95)",
                  "trial_max":"(per-trial max)"}[mode]
    win_txt = f"[{t[k0]:.2f}s, {t[min(k1,T)-1]:.2f}s]"

    traj_labels, traj_groups = get_traj_groups(data)
    n_traj = len(traj_labels)

    for metric in metrics:
        fig, axes = plt.subplots(1, n_traj, figsize=(6*n_traj, 5), sharey=True)
        if n_traj == 1:
            axes = [axes]

        for ax, lab, idxs in zip(axes, traj_labels, traj_groups):
            for algo in ALGOS:
                E_all = data[f"{algo}__{metric}"]  # (N,T)
                E = E_all[idxs, :]
                vals = reduce_trial(E[:, k0:k1])
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    continue
                xs = np.sort(vals)
                ys = np.arange(1, xs.size+1) / xs.size
                ax.plot(xs, ys, label=algo)

            ax.set_xlabel(titles[metric])
            if ax is axes[0]:
                ax.set_ylabel("Fraction of trials" if mode != "samples" else "Fraction of samples")
            ax.set_title(f"{metric.upper()} — traj: {lab}\n{label_mode}  window={win_txt}, stride={stride}")
            ax.grid(True, which="both")
            if xlog: ax.set_xscale("log")
            if xmax_per_metric and metric in xmax_per_metric:
                ax.set_xlim(left=0, right=xmax_per_metric[metric])
            ax.legend()

        fig.tight_layout()
        out = outdir / f"error_ecdf_fulltraj_{_sanitize(metric)}_{mode}_by_traj_t{tmin if tmin is not None else 0}-{tmax if tmax is not None else int(t[-1])}_s{stride}.png"
        fig.savefig(out, dpi=200); plt.close(fig)
        print(f"Saved: {out.resolve()}")


# ============ main ============

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=str, required=True, help="Path to results.npz")
    ap.add_argument("--outdir", type=str, default=None, help="Dir to save plots (default: alongside npz)")

    # classic views
    ap.add_argument("--plot_timeseries", action="store_true")
    ap.add_argument("--plot_3d", action="store_true")
    ap.add_argument("--traj_algo", type=str, default=None)
    ap.add_argument("--trial_idx", type=int, default=0)

    # Event-based ECDF (add/drop)
    ap.add_argument("--plot_event_ecdf", action="store_true",
                    help="ECDF of errors over a fixed window after each event, split by trajectory")
    ap.add_argument("--ecdf_window_sec", type=float, default=5.0,
                    help="Window length (s) after event")
    ap.add_argument("--ecdf_xmax", type=float, nargs="+", default=[0.05],
                    help="x-axis max for ECDF; one value or two values (add drop)")
    ap.add_argument("--ecdf_stat", type=str, default="mean",
                    choices=["mean","median","max","p95","end"],
                    help="Reduce the window to a single value per trial")

    # Full-trajectory ECDF (no events)
    ap.add_argument("--plot_ecdf", action="store_true",
                    help="ECDF of error over whole trajectory (no events), split by trajectory")
    ap.add_argument("--ecdf_mode", type=str, default="samples",
                    choices=["samples","trial_mean","trial_median","trial_p95","trial_max"])
    ap.add_argument("--ecdf_tmin", type=float, default=None, help="Start time (s)")
    ap.add_argument("--ecdf_tmax", type=float, default=None, help="End time (s)")
    ap.add_argument("--ecdf_stride", type=int, default=1, help="Subsample timesteps")
    ap.add_argument("--ecdf_xlog", action="store_true", help="Log-scale x-axis")
    ap.add_argument("--ecdf_xmax_pos", type=float, default=None)
    ap.add_argument("--ecdf_xmax_vel", type=float, default=None)
    ap.add_argument("--ecdf_xmax_ori", type=float, default=None)

    args = ap.parse_args()

    npz_path = Path(args.npz)
    outdir = Path(args.outdir) if args.outdir else npz_path.parent
    outdir.mkdir(parents=True, exist_ok=True)
    data = np.load(npz_path, allow_pickle=False)

    if args.plot_timeseries:
        plot_timeseries_grouped(data, outdir)

    if args.plot_3d:
        plot_3d_single(data, outdir, algo=args.traj_algo, trial_idx=args.trial_idx)

    if args.plot_event_ecdf:
        xmax_param = args.ecdf_xmax if len(args.ecdf_xmax) > 1 else args.ecdf_xmax[0]
        plot_event_error_ecdf_grouped(
            data, outdir,
            metrics=("pos", "vel", "ori"),
            window_sec=args.ecdf_window_sec,
            xmax=xmax_param,
            stat=args.ecdf_stat,
        )

    if args.plot_ecdf:
        xmax_per_metric = {}
        if args.ecdf_xmax_pos is not None: xmax_per_metric["pos"] = args.ecdf_xmax_pos
        if args.ecdf_xmax_vel is not None: xmax_per_metric["vel"] = args.ecdf_xmax_vel
        if args.ecdf_xmax_ori is not None: xmax_per_metric["ori"] = args.ecdf_xmax_ori
        if not xmax_per_metric:
            xmax_per_metric = {"pos":0.03, "vel":0.15, "ori":2.0}

        plot_error_ecdf_fulltraj_grouped(
            data, outdir,
            metrics=("pos","vel","ori"),
            mode=args.ecdf_mode,
            tmin=args.ecdf_tmin,
            tmax=args.ecdf_tmax,
            stride=max(1, int(args.ecdf_stride)),
            xlog=args.ecdf_xlog,
            xmax_per_metric=xmax_per_metric,
        )


if __name__ == "__main__":
    main()
