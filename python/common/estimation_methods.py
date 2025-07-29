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
    def __init__(self, num_params, alpha=0.99, epsilon=1e-8, x0=None):
        self.num_params = num_params
        self.alpha = alpha
        self.epsilon = epsilon  # Small value to prevent division by zero
        self.x = (
            x0.reshape(num_params, 1) if x0 is not None else np.zeros((num_params, 1))
        )

    def iterate(self, A, b, x0=None, num_iterations=1000, tol=0.01):
        """
        A: (num_states, num_param)
        x: (num_param, 1)
        b: (num_states, 1)
        """
        self.A = A
        self.b = b
        self.m = A.shape[0]  # m is the number of rows, n is the number of columns
        self.n = A.shape[1]

        max_iter = 0
        
        if x0 is not None:
            self.x = x0.reshape(self.num_params, 1)

        for _ in range(num_iterations): 
            max_iter += 1

            # Compute exponential weighting for the rows
            row_norms = np.linalg.norm(A, axis=1) ** 2

            # Normalize by subtracting the maximum (robust to large values)
            row_norms -= np.max(row_norms)

            # Scale by alpha (can be adjusted for better performance)
            row_norms *= self.alpha

            # Add epsilon to prevent division by zero or log(0)
            row_norms += self.epsilon

            # Exponentiate (more stable now due to normalization)
            exponential_weights = np.exp(row_norms)

            # Calculate probabilities (should be stable now)
            probabilities = exponential_weights / np.sum(exponential_weights)

            # Ensure probabilities sum to 1 (handle potential rounding errors)
            probabilities /= np.sum(probabilities)

            i = np.random.choice(self.m, p=probabilities.flatten())
            a_i = np.array(self.A[i]).reshape(1, -1)  # Ensure a_i is a row vector
            b_i = np.array(self.b[i]).reshape(1, 1)  # Ensure b_i is a column vector

            # Ensure that a_i has more than one element before calculating the norm
            if a_i.size > 1:
                norm_a_i = np.linalg.norm(a_i)
            else:
                norm_a_i = np.abs(a_i)  # Fallback for single-element case

            residual = np.dot(self.A, self.x) - b

            if np.abs(residual).sum() < tol:
                print("tolerance hit")
                break

            # Ensure the denominator is not too close to zero before division
            if norm_a_i > 1e-6:
                increment = ((b_i - np.dot(a_i, self.x)) / (norm_a_i**2)) * a_i.T
            else:
                increment = np.zeros((self.n, 1))

            increment = increment.reshape(self.n, 1)
            self.x = self.x + increment

        return self.x, max_iter