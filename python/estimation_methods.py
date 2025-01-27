import autograd.numpy as np

class RLS:
    def __init__(self, num_params, forgetting_factor=0.97):
        """
        Initialize the RLS solver.

        Args:
            num_params (int): Number of parameters in x (size of x).
            forgetting_factor (float): Forgetting factor (lambda) for RLS. Default is 1 (no forgetting).
        """
        self.num_params = num_params
        self.forgetting_factor = forgetting_factor

        # Initialize the covariance matrix (P) and parameter vector (x)
        self.P = np.eye(num_params) * 1e6  # Large initial covariance #TODO: use A and b to initialize P
        self.x = np.zeros((num_params, 1))

    def initialize(self, A0, b0):
        """
        Initialize P and x based on the initial data.

        """
        self.P = np.linalg.inv(A0.T @ A0)  # Initial covariance matrix
        self.x = (self.P @ (A0.T @ b0)).reshape(-1,1)     # Initial parameter estimate

    def update(self, A, b):
        """
        Update the RLS estimate using the full matrix A and vector b.

        Args:
            A (numpy.ndarray): Matrix A with shape (num_states, num_params).
            b (numpy.ndarray): Vector b with shape (num_states, 1).
        """
        num_states, num_params = A.shape
        b = b.reshape(num_states, 1)

        # Compute the Kalman gain
        P_A = self.P @ A.T  # Shape (num_params, num_states)
        gain_denominator = self.forgetting_factor * np.eye(A.shape[0]) + A @ P_A  # Shape (num_states, num_states)
        K = P_A @ np.linalg.inv(gain_denominator)  # Shape (num_params, num_states)

        # Update the parameter estimate
        self.x += K @ (b - A @ self.x)  # Shape (num_params, 1)

        # Update the covariance matrix
        self.P = (np.eye(num_params) - K @ A @ self.P) / self.forgetting_factor

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
    def __init__(self, alpha=0.99):
        self.alpha = alpha

    def iterate(self, A, b, x0, num_iterations, tol = 0.01):
        """
        A: (num_states, num_param)
        x: (num_param, 1)
        b: (num_states, 1)
        """
        self.A = A
        self.b = b
        self.m = A.shape[0]  # m is the number of rows, n is the number of columns
        self.n = A.shape[1]
        self.x = np.array([x0]).reshape(self.n,1)  # Initial estimate of solution
        # print("cond: ",np.linalg.cond(A))
        # print("A: ", self.A)
        # print("b: ", self.b)
        # print("x0: ", self.x)
        for _ in range(num_iterations):
            # Compute exponential weighting for the rows
            row_norms = np.linalg.norm(A,axis=1)**2
            # probabilities = row_norms / np.sum(row_norms)
            mask = (row_norms > 1e-6)
            # print(mask)
            exponential_weights = mask * np.exp(self.alpha * row_norms)
            probabilities = exponential_weights / (np.sum(exponential_weights))
            i = np.random.choice(self.m, p = probabilities)            # Update rule using the selected row
            # print(i)
            #print(i)
            a_i = np.array(self.A[i]) #.reshape(13,1)
            b_i = np.array(self.b[i])#.reshape(1,13)
            residual = np.dot(self.A, self.x) - b
            # print("A: ", self.A)
            # print("b", self.b)
            # print("residual size: ", np.abs(residual).sum())
            if np.abs(residual).sum()/self.m < tol:
                print("tolerance hit")
                break
            increment = ((b_i - np.dot(a_i,self.x)) / (np.linalg.norm(a_i)**2))*a_i
            increment = increment.reshape(self.n,1)

            self.x = self.x + increment
        return self.x