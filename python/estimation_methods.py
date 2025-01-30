import autograd.numpy as np

class RLS:
    def __init__(self, num_params, forgetting_factor=0.99, beta=0.95, noise_power=1e-6):
        """
        Initialize the Variable Forgetting Factor RLS solver.

        Args:
            num_params (int): Number of parameters in x (size of x).
            forgetting_factor (float): Initial forgetting factor λ (default: 0.99).
            beta (float): Smoothing factor for error power estimation (default: 0.95).
            noise_power (float): Estimated noise power σ_v^2 (default: small).
        """
        self.num_params = num_params
        self.forgetting_factor = forgetting_factor
        self.beta = beta
        self.sigma_v2 = noise_power  # Estimated noise power
        
        # Initialize covariance matrix (P) with a large positive value for stability
        self.P = np.eye(num_params) * 1e6  
        self.x = np.zeros((num_params, 1))

        # Initialize error power estimate
        self.sigma_e2 = 1e-6  

    def initialize(self, A0, b0):
        """
        Initialize P and x based on the initial data.
        """
        reg = 1e-6 * np.eye(A0.shape[1])  # Regularization for stability
        self.P = np.linalg.inv(A0.T @ A0 + reg)  
        self.x = (self.P @ (A0.T @ b0)).reshape(-1,1)

    def update(self, A, b):
        """
        Update the RLS estimate using the full matrix A and vector b.

        Args:
            A (numpy.ndarray): Matrix A with shape (num_states, num_params).
            b (numpy.ndarray): Vector b with shape (num_states, 1).
        """
        num_states, num_params = A.shape
        b = b.reshape(num_states, 1)

        # Compute a priori error
        e = b - A @ self.x  # Shape (num_states, 1)

        # Compute Kalman gain
        P_A = self.P @ A.T  # Shape (num_params, num_states)
        gain_denominator = self.forgetting_factor * np.eye(A.shape[0]) + A @ P_A  # Shape (num_states, num_states)
        K = P_A @ np.linalg.inv(gain_denominator)  # Shape (num_params, num_states)

        # Update parameter estimate
        self.x += K @ e  # Shape (num_params, 1)

        # Update covariance matrix (fixed formula)
        self.P = (self.P - K @ A @ self.P) / self.forgetting_factor

        # Update power estimates and forgetting factor
        self.sigma_e2 = self.beta * self.sigma_e2 + (1 - self.beta) * np.linalg.norm(e)**2
        self.forgetting_factor = 1 - (self.sigma_v2 / self.sigma_e2)

    def predict(self):
        """
        Get the current parameter estimate x.

        Returns:
            numpy.ndarray: The current estimate of x, shape (num_params, 1).
        """
        return self.x
    
class EKF:
    def __init__(self, process_noise=1e-3, measurement_noise=1e-1):
        self.theta =0.                    # Initial gravity estimate
        self.P = 100.0                       # Initial covariance
        self.Q = process_noise                # Process noise covariance
        self.R = np.eye(13) * measurement_noise  # Measurement noise covariance

    def predict(self):
        # Prediction step
        self.P = self.P + self.Q

    def update(self, x, y):
        """
        x: 13x1 Jacobian vector (sensitivity of dynamics to gravity)
        y: 13x1 observation vector (state differences)
        """
        # Ensure x and y are column vectors
        x = np.reshape(x, (13, 1))
        y = np.reshape(y, (13, 1))
        print(x,y)
        # Innovation covariance
        S = (x @ (self.P * x.T)) + self.R
        # Kalman gain
        K = (self.P * x.T) @ np.linalg.inv(S)  # K is 1x13

        # Update state estimate
        self.theta = self.theta + (K @ y).item()
        print(self.theta)
        # Update covariance
        self.P = self.P - (K @ S @ K.T).item()
        return self.theta
    
class RK:
    def __init__(self, alpha=0.99, epsilon=1e-8):
        self.alpha = alpha
        self.epsilon = epsilon  # Small value to prevent division by zero

    def iterate(self, A, b, x0, num_iterations, tol=0.01):
        """
        A: (num_states, num_param)
        x: (num_param, 1)
        b: (num_states, 1)
        """
        self.A = A
        self.b = b
        self.m = A.shape[0]  # m is the number of rows, n is the number of columns
        self.n = A.shape[1]
        self.x = np.array([x0]).reshape(self.n, 1)  # initial estimate of solution

        for _ in range(num_iterations):
            # Compute exponential weighting for the rows
            row_norms = np.linalg.norm(A, axis=1)**2

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

            if np.abs(residual).sum() / self.m < tol:
                print("tolerance hit")
                break

            # Ensure the denominator is not too close to zero before division
            if norm_a_i > 1e-6:
                increment = ((b_i - np.dot(a_i, self.x)) / (norm_a_i**2)) * a_i.T
            else:
                increment = np.zeros((self.n, 1))

            increment = increment.reshape(self.n, 1)
            self.x = self.x + increment
        return self.x