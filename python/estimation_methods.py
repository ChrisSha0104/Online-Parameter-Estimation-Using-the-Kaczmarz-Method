import autograd.numpy as np

class RLS:
    def __init__(self, num_params, lambda_factor=0.97):
        self.num_params = num_params
        self.lambda_factor = lambda_factor
        self.theta = 0.                    # Parameter estimate
        self.P = 1000.                       # Large initial covariance matrix

    def update(self, x, y):
        # Reshape inputs to column vectors
        x = np.reshape(x, (13, 1))          # TODO: 13 is hard coded! Change this
        y = np.reshape(y, (13, 1))          # Compute Kalman gain
        P_x = self.P * x
        gain_denominator = self.lambda_factor + x.T @ P_x
        K = P_x / gain_denominator        # Update estimate
        y_pred = x * self.theta
        self.theta = self.theta + K.T @ (y - y_pred)        # Update covariance matrix
        self.P = (self.P - K.T @ x * self.P) / self.lambda_factor
        return self.theta
    
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