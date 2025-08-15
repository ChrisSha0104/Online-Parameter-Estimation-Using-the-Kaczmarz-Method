import numpy as np

class RLS:
    """RLS with adaptive forgetting factor (lambda)."""

    def __init__(self, num_params, theta_hat=None, forgetting_factor=0.3, c=1000):
        """
        :param num_params: Number of parameters to estimate.
        :param theta_hat: Initial estimate of parameters, otherwise set to zeros.
        :param forgetting_factor: Forgetting factor (typically referred to as lambda).
        :param c: Constant factor for initial covariance matrix scale.
        """
        self.n = num_params
        self.P = np.eye(num_params) * c
        self.theta_hat = (
            theta_hat if theta_hat is not None else np.zeros((num_params, 1))
        )
        self.lambda_ = forgetting_factor

    def iterate(self, A, b):
        """
        :param A: Jacobian of system w.r.t. parameters.
        :param b: Measurement vector.
        """
        K = (
            self.P
            @ A.T
            @ np.linalg.inv(A @ self.P @ A.T + self.lambda_ * np.eye(A.shape[0]))
        )
        self.theta_hat += K @ (b - A @ self.theta_hat)
        self.P = (self.P - K @ A @ self.P) / self.lambda_
        return self.theta_hat
    
class KF:
    """
    Basic Kalman Filter

    Assumes that the process noise and measurement noise are known, and that their Jacobians are I. Additionally, the process model jacobian is assumed to be I.

    The process model and measurement model are flipped!

    process model x_k is parameters theta
    measurement z_k is state x
    """

    def __init__(
        self, num_params, process_noise, measurement_noise, theta_hat=None, c=10
    ):
        """
        :param num_params: Number of parameters to estimate (p).
        :param theta_hat: Initial estimate of parameters, otherwise set to zeros (p x 1 vector).
        :param forgetting_factor: Forgetting factor (typically referred to as lambda).
        :param c: Constant factor for initial covariance matrix scale.
        :param P: Estimation error covariance (p x p matrix)
        :param Q: Process noise covariance (p x p matrix)
        :param R: Measurement noise covariance (d x d matrix)
        """
        self.n = num_params
        self.theta_hat = (
            theta_hat if theta_hat is not None else np.zeros((num_params, 1))
        )
        self.P = np.eye(num_params) * c
        self.Q = process_noise
        self.R = measurement_noise

    def iterate(self, A, b):
        """
        :param A: Jacobian of system w.r.t. parameters (d x p matrix).
        :param b: Measurement vector (d x 1 vector).
        """
        Ftheta = np.eye(self.P.shape[0])
        self.P = Ftheta @ self.P @ Ftheta.T + self.Q
        K = self.P @ A.T @ np.linalg.inv(A @ self.P @ A.T + self.R)
        self.theta_hat += K @ (b - A @ self.theta_hat)
        self.P -= K @ A @ self.P
        return self.theta_hat
    
class RK:
    def __init__(self, num_params, x0=None, num_iters=100):
        """
        Classic Randomized Kaczmarz algorithm.

        Args:
            A: (m, n) matrix
            b: (m,) or (m, 1) vector
            x0: (n,) or (n, 1) initial guess
            num_iters: number of iterations

        Returns:
            x: (n,) solution estimate
        """
        self.num_params = num_params
        self.x0 = x0 if x0 is not None else np.zeros(num_params, )
        self.num_iters = num_iters
    
    def iterate(self, A, b):
        m, n = A.shape
        b = b.reshape(-1, 1)
        x = np.zeros((self.num_params, 1)) if self.x0 is None else self.x0.reshape(self.num_params, 1)

        row_norms_sq = np.sum(A**2, axis=1)
        prob = row_norms_sq / np.sum(row_norms_sq)

        num_iter = m

        for _ in range(num_iter):
            i = np.random.choice(m, p=prob)
            a_i = A[i].reshape(1, -1)   # shape (1, n)
            b_i = b[i]                  # scalar
            norm_sq = row_norms_sq[i]

            # Projection update
            residual = b_i - a_i @ x
            # print("residual_norm:", f"{np.linalg.norm(residual):.4g}")
            x += (residual / norm_sq) * a_i.T

        return x.flatten()
    
class TARK:
    def __init__(self, num_params, x0=None):
        """
        Tail-Averaged Randomized Kaczmarz.

        Args:
            A: (m, n) matrix
            b: (m,) or (m,1) vector
            x0: optional initial guess (n,) or (n,1)
        """
        self.num_params = num_params
        self.x = np.zeros((num_params, 1)) if x0 is None else x0.reshape(num_params, 1)

    def iterate(self, A, b, burnin: int = 0, total_iters: int = 100):
        """
        Run TARK iterations and return the tail-averaged solution.

        Args:
            burnin: number of initial iterations to ignore in averaging
            total_iters: total number of iterations

        Returns:
            x_avg: (n,) tail-averaged solution
        """

        x = self.x.copy()
        x_sum = np.zeros_like(x)
        count = 0

        row_norms_sq = np.sum(A**2, axis=1)
        probs = row_norms_sq / np.sum(row_norms_sq)

        num_iters = A.shape[0]

        for s in range(num_iters):
            i = np.random.choice(A.shape[0], p=probs)
            a_i = A[i].reshape(1, -1)
            b_i = b[i]
            norm_sq = row_norms_sq[i]

            # RK update
            x += ((b_i - a_i @ x) / norm_sq) * a_i.T

            # Tail averaging
            if s >= burnin:
                x_sum += x
                count += 1

        x_avg = x_sum / count
        return x_avg.flatten()
    
class DEKA:
    pass