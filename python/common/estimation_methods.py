import numpy as np

# ============================== RLS ==============================

class RLS:
    """RLS with adaptive forgetting factor (lambda) using 1-D state."""
    def __init__(self, num_params, theta_hat=None, forgetting_factor=0.3, c=1000.0):
        self.n = num_params
        self.P = np.eye(num_params) * float(c)          # (n, n)
        self.theta_hat = (np.zeros(self.n) if theta_hat is None
                          else np.asarray(theta_hat).reshape(-1))       # (n,)
        self.lambda_ = float(forgetting_factor)

    def iterate(self, A, b):
        A = np.asarray(A)                                # (m, n)
        b = np.asarray(b).reshape(-1)                    # (m,)
        m, n = A.shape
        assert n == self.n, "A.shape[1] must match num_params"

        S = A @ self.P @ A.T + self.lambda_ * np.eye(m)  # (m, m)
        K = self.P @ A.T @ np.linalg.inv(S)              # (n, m)

        innov = b - A @ self.theta_hat                   # (m,)
        self.theta_hat = self.theta_hat + K @ innov      # (n,)
        self.P = (self.P - K @ A @ self.P) / self.lambda_
        return self.theta_hat


# ============================== KF ===============================

class KF:
    """
    Basic (flipped-model) Kalman filter treating parameters as the state.
    Uses 1-D theta_hat, 2-D P/Q/R.
    """
    def __init__(self, num_params, process_noise, measurement_noise, theta_hat=None, c=10.0):
        self.n = num_params
        self.theta_hat = (np.zeros(self.n) if theta_hat is None
                          else np.asarray(theta_hat).reshape(-1))       # (n,)
        self.P = np.eye(self.n) * float(c)                              # (n, n)
        self.Q = np.asarray(process_noise, dtype=float)                 # (n, n)
        self.R = np.asarray(measurement_noise, dtype=float)             # (m, m) at runtime

    def iterate(self, A, b):
        A = np.asarray(A)                               # (m, n)
        b = np.asarray(b).reshape(-1)                   # (m,)
        m, n = A.shape
        assert n == self.n, "A.shape[1] must match num_params"
        assert self.R.shape == (m, m), "R must be (m, m) for current A"

        # Predict
        self.P = self.P + self.Q

        # Update
        S = A @ self.P @ A.T + self.R                   # (m, m)
        K = self.P @ A.T @ np.linalg.inv(S)             # (n, m)
        innov = b - A @ self.theta_hat                  # (m,)
        self.theta_hat = self.theta_hat + K @ innov     # (n,)
        self.P = self.P - K @ A @ self.P
        return self.theta_hat


# ============================== RK ===============================

class RK:
    """Classic Randomized Kaczmarz with 1-D state; m*n inner updates per call."""
    def __init__(self, num_params, x0=None):
        self.num_params = num_params
        self.x0 = np.zeros(num_params) if x0 is None else np.asarray(x0).reshape(-1)

    def iterate(self, A, b, eps=1e-12):
        A = np.asarray(A)                    # (m, n)
        b = np.asarray(b).reshape(-1)        # (m,)
        m, n = A.shape
        assert n == self.num_params

        x = self.x0.copy()                   # (n,)
        row_norms_sq = (A * A).sum(axis=1) + eps
        probs = row_norms_sq / row_norms_sq.sum()

        for _ in range(m * n):
            i = np.random.choice(m, p=probs)
            ai = A[i]                        # (n,)
            residual = b[i] - ai @ x
            x = x + (residual / row_norms_sq[i]) * ai

        self.x0 = x
        return x


# ============================== TARK =============================

class TARK:
    """
    Tail-Averaged Randomized Kaczmarz (m*n updates per call).
    Returns the tail-averaged estimate; keeps last x in self.x0.
    """
    def __init__(self, num_params, x0=None):
        self.num_params = num_params
        self.x0 = np.zeros(num_params) if x0 is None else np.asarray(x0).reshape(-1)

    def iterate(self, A, b, burnin: int = 0, eps: float = 1e-12):
        A = np.asarray(A)                    # (m, n)
        b = np.asarray(b).reshape(-1)        # (m,)
        m, n = A.shape
        assert n == self.num_params

        x = self.x0.copy()
        x_sum = np.zeros_like(x)
        count = 0

        row_norms_sq = (A * A).sum(axis=1) + eps
        probs = row_norms_sq / row_norms_sq.sum()

        total_iters = m * n
        for s in range(total_iters):
            i = np.random.choice(m, p=probs)
            ai = A[i]
            residual = b[i] - ai @ x
            x = x + (residual / row_norms_sq[i]) * ai
            if s >= burnin:
                x_sum += x
                count += 1

        x_avg = x_sum / max(count, 1)
        self.x0 = x
        return x_avg


# ============================== IGRK =============================

class IGRK:
    def __init__(self, num_params, x0=None, eps=1e-12):
        self.num_params = num_params
        self.x0 = np.zeros(num_params) if x0 is None else np.asarray(x0).reshape(-1)
        self.eps = float(eps)

    def iterate(self, A, b):
        A = np.asarray(A); b = np.asarray(b).reshape(-1)
        m, n = A.shape
        assert n == self.num_params

        x = self.x0.copy()
        row_norm2 = (A * A).sum(axis=1)
        pos = row_norm2[row_norm2 > self.eps]
        floor = self.eps if pos.size == 0 else max(self.eps, 1e-6 * float(np.median(pos)))
        row_norm2 = np.maximum(row_norm2, floor)
        valid = row_norm2 > 10*self.eps
        num_iter = m * n

        for _ in range(num_iter):
            r = A @ x - b
            crit = (r * r) / row_norm2
            crit[~valid] = -np.inf

            if not np.isfinite(crit).any():
                break

            Nk = np.flatnonzero(valid & (np.abs(r) > 0))
            Gamma_k = row_norm2[Nk].sum() if Nk.size else row_norm2[valid].sum()
            crit_valid = crit[valid]
            if crit_valid.size == 0: break
            thresh = 0.5 * (crit_valid.max() + (np.linalg.norm(r) ** 2) / max(Gamma_k, self.eps))

            Jk = np.flatnonzero(valid & (crit >= thresh))
            if Jk.size == 0:
                best_local = int(np.argmax(crit_valid))
                Jk = np.array([np.flatnonzero(valid)[best_local]])

            weights = crit[Jk]
            if not np.isfinite(weights).any() or weights.sum() <= 0:
                weights = np.ones(Jk.size) / Jk.size
            else:
                weights = weights / weights.sum()

            i = np.random.choice(Jk, p=weights)

            ai = A[i]; den = row_norm2[i]
            ri = ai @ x - b[i]
            step = (ri / den) * ai
            # optional: clip step
            max_step = 10.0 * (np.linalg.norm(x) + 1e-12)
            sn = np.linalg.norm(step)
            if sn > max_step:
                step *= max_step / (sn + 1e-12)

            x = x - step

        self.x0 = x
        return x


# ============================== MGRK =============================

class MGRK:
    def __init__(self, num_params, x0=None, alpha=1.0, beta=0.0, theta=0.5, eps=1e-12):
        assert alpha > 0 and beta >= 0 and 0.0 <= theta <= 1.0
        self.num_params = num_params
        self.x0 = np.zeros(num_params) if x0 is None else np.asarray(x0).reshape(-1)
        self.x_prev = self.x0.copy()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.theta = float(theta)
        self.eps = float(eps)

    def iterate(self, A, b):
        A = np.asarray(A); b = np.asarray(b).reshape(-1)
        m, n = A.shape
        assert n == self.num_params

        x = self.x0.copy()
        x_prev = self.x_prev.copy()

        row_norm2 = (A * A).sum(axis=1)
        pos = row_norm2[row_norm2 > self.eps]
        floor = self.eps if pos.size == 0 else max(self.eps, 1e-6 * float(np.median(pos)))
        row_norm2 = np.maximum(row_norm2, floor)
        valid = row_norm2 > 10*self.eps

        num_iter = m * n
        for _ in range(num_iter):
            r = A @ x - b
            crit = (r * r) / row_norm2
            crit[~valid] = -np.inf

            if not np.isfinite(crit).any():
                break

            Nk = np.flatnonzero(valid & (np.abs(r) > 0))
            Gamma_k = row_norm2[Nk].sum() if Nk.size else row_norm2[valid].sum()
            crit_valid = crit[valid]
            if crit_valid.size == 0: break
            thresh = self.theta * crit_valid.max() + (1.0 - self.theta) * (np.linalg.norm(r) ** 2) / max(Gamma_k, self.eps)
            Sk = np.flatnonzero(valid & (crit >= thresh))
            if Sk.size == 0:
                best_local = int(np.argmax(crit_valid))
                Sk = np.array([np.flatnonzero(valid)[best_local]])

            weights = crit[Sk]
            if not np.isfinite(weights).any() or weights.sum() <= 0:
                weights = np.ones(Sk.size) / Sk.size
            else:
                weights = weights / weights.sum()

            i = np.random.choice(Sk, p=weights)

            ai = A[i]; den = row_norm2[i]
            ri = ai @ x - b[i]
            grad = (ri / den) * ai

            # optional: clip grad
            max_step = 10.0 * (np.linalg.norm(x) + 1e-12)
            gn = np.linalg.norm(grad)
            if gn > max_step:
                grad *= max_step / (gn + 1e-12)

            x_new = x - self.alpha * grad + self.beta * (x - x_prev)
            x_prev, x = x, x_new

        self.x_prev = x_prev
        self.x0 = x
        return x


# ============================== REK ==============================

class REK:
    """
    Randomized Extended Kaczmarz with 1-D state; m*n paired (column,row) updates per call.
    Maintains an auxiliary z ∈ R^m.
    """
    def __init__(self, num_params, x0=None, eps=1e-12):
        self.num_params = num_params
        self.x0 = np.zeros(num_params) if x0 is None else np.asarray(x0).reshape(-1)
        self.z = None                                # (m,) allocated lazily
        self.eps = float(eps)

    def iterate(self, A, b):
        A = np.asarray(A)                    # (m, n)
        b = np.asarray(b).reshape(-1)        # (m,)
        m, n = A.shape
        assert n == self.num_params

        x = self.x0.copy()                   # (n,)
        if self.z is None or self.z.shape[0] != m:
            self.z = b.copy()
        z = self.z.copy()                    # (m,)

        row_norm2 = (A * A).sum(axis=1) + self.eps   # (m,)
        col_norm2 = (A * A).sum(axis=0) + self.eps   # (n,)
        p_rows = row_norm2 / row_norm2.sum()
        p_cols = col_norm2 / col_norm2.sum()

        for _ in range(m * n):
            # Column step: update z
            j = np.random.choice(n, p=p_cols)
            a_col = A[:, j]                          # (m,)
            z = z - ((a_col @ z) / col_norm2[j]) * a_col

            # Row step: update x
            i = np.random.choice(m, p=p_rows)
            a_row = A[i]                              # (n,)
            residual_i = b[i] - z[i] - a_row @ x
            x = x + (residual_i / row_norm2[i]) * a_row

        self.z = z
        self.x0 = x
        return x

import numpy as np

class DEKA:
    """
    Deterministic Kaczmarz with Smoothing & Damping (DeKA).
    Greedy row-set selection each outer step; optional multi-step per call.

    Args (constructor):
        num_params (int): dimension n of the parameter vector.
        x0 (np.ndarray | None): initial (n,). Defaults to zeros.
        eps (float): tiny number to avoid divide-by-zero.
        gamma (float): damping factor (scales update).
        reg (float): Tikhonov regularization for denominator.
        alpha (float): exponential smoothing factor in [0,1]. Higher = smoother.
        steps_per_call (int): number of DeKA outer iterations per .iterate() call.
        tol (float): early-stop on residual^2 if <= tol (0 disables).
    """
    def __init__(
        self,
        num_params,
        x0=None,
        eps=1e-12,
        gamma=0.5,
        reg=1e-12,
        alpha=0.3,
        tol=1e-8,
    ):
        self.num_params = int(num_params)
        self.x0 = np.zeros(self.num_params) if x0 is None else np.asarray(x0, dtype=float).reshape(-1)
        assert self.x0.shape == (self.num_params,)
        self.eps   = float(eps)
        self.gamma = float(gamma)
        self.reg   = float(reg)
        self.alpha = float(alpha)
        self.tol   = float(tol)

        # smoothed estimate (kept across calls)
        self.x_smooth = self.x0.copy()

    def iterate(self, A, b):
        """
        Run 'steps_per_call' DeKA outer iterations.
        Shapes:
            A: (m, n)
            b: (m,)
        Returns:
            np.ndarray: current smoothed estimate (n,)
        """
        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float).reshape(-1)
        m, n = A.shape
        assert n == self.num_params, f"expected n={self.num_params}, got {n}"
        assert b.shape == (m,), f"b must be (m,), got {b.shape}"

        x = self.x0.copy()          # raw (unsmoothed) estimate carried within this call

        # Precompute norms used in thresholds
        row_norm2 = (A * A).sum(axis=1) + self.eps       # (m,)
        fro2 = (A * A).sum() + self.eps                  # ||A||_F^2

        for _ in range(m * n * n):
            r = b - A @ x                                # residual vector (m,)
            r2 = float(r @ r)                            # ||r||_2^2

            if self.tol > 0.0 and r2 <= self.tol:
                break

            # Per-row “energy”: |b_i - a_i x|^2 / ||a_i||^2
            num = (r ** 2)                               # since r_i = b_i - a_i x
            ratios = num / row_norm2                     # (m,)
            ratios_max = float(ratios.max())

            # Threshold epsilon_k (half of: max-ratio scaled by 1/||r||^2 plus 1/||A||_F^2)
            # If r2 is ~0, fall back to the 1/||A||_F^2 term.
            if r2 > self.eps:
                eps_k = 0.5 * (ratios_max / r2 + 1.0 / fro2)
            else:
                eps_k = 0.5 * (1.0 / fro2)

            # Greedy candidate set τ_k:
            # Keep rows whose normalized error exceeds eps_k * ||r||^2 * ||a_i||^2
            # Equivalent check using already computed terms:
            #   num[i]/row_norm2[i] >= eps_k * r2 * row_norm2[i]
            # -> num[i] >= eps_k * r2 * (row_norm2[i]**2)
            thresh = eps_k * r2 * (row_norm2 ** 2)
            mask = (num >= thresh)

            if not np.any(mask):
                # Fallback: pick the single worst row by ratio
                mask[np.argmax(ratios)] = True

            # Aggregated residual η_k: only entries in τ_k kept
            eta = np.where(mask, r, 0.0)                 # (m,)

            # Update direction with regularization:
            AT_eta = A.T @ eta                           # (n,)
            denom = float(AT_eta @ AT_eta) + self.reg    # ||A^T η||^2 + reg
            # Scale by η^T r (energy alignment)
            scale = float(eta @ r) / denom
            delta = scale * AT_eta                        # (n,)

            # Damping
            delta *= self.gamma

            # Raw update
            x = x + delta

            # Exponential smoothing (stateful across calls)
            self.x_smooth = self.alpha * self.x_smooth + (1.0 - self.alpha) * x

        # Persist for next call
        self.x0 = x.copy()
        return self.x_smooth.copy()
