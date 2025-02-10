import autograd.numpy as np
import unittest

class RLS:
    def __init__(self, num_params, forgetting_factor=0.98):
        self.P = np.eye(num_params) * 1000  # Large initial covariance
        self.theta_hat = np.zeros((num_params, 1))
        self.lambda_ = forgetting_factor

    def update(self, A, b):
        """Update rule for RLS with forgetting factor"""
        A = np.atleast_2d(A)  # Ensure A is a matrix
        b = np.atleast_2d(b).reshape(-1, 1)  # Ensure b is a column vector

        K = self.P @ A.T @ np.linalg.inv(A @ self.P @ A.T + self.lambda_ * np.eye(A.shape[0]))
        self.theta_hat += K @ (b - A @ self.theta_hat)
        self.P = (self.P - K @ A @ self.P) / self.lambda_  # Forget old data

    def predict(self):
        return self.theta_hat

# class RLS:
#     def __init__(self, num_params, forgetting_factor=0.95, beta=0.95, noise_power=1e-6):
#         """
#         Initialize the Variable Forgetting Factor RLS solver.

#         Args:
#             num_params (int): Number of parameters in x (size of x).
#             forgetting_factor (float): Initial forgetting factor λ (default: 0.99).
#             beta (float): Smoothing factor for error power estimation (default: 0.95).
#             noise_power (float): Estimated noise power σ_v^2 (default: small).
#         """
#         self.num_params = num_params
#         self.forgetting_factor = forgetting_factor
#         self.beta = beta
#         self.sigma_v2 = noise_power  # Estimated noise power
        
#         # Initialize covariance matrix (P) with a large positive value for stability
#         self.P = np.eye(num_params) * 1e6  
#         self.x = np.zeros((num_params, 1))

#         # Initialize error power estimate
#         self.sigma_e2 = 1e-6  

#     def initialize(self, A0, b0):
#         """
#         Initialize P and x based on the initial data.
#         """
#         reg = 1e-6 * np.eye(A0.shape[1])  # Regularization for stability
#         self.P = np.linalg.inv(A0.T @ A0 + reg)  
#         self.x = (self.P @ (A0.T @ b0)).reshape(-1,1)

#     def update(self, A, b):
#         """
#         Update the RLS estimate using the full matrix A and vector b.

#         Args:
#             A (numpy.ndarray): Matrix A with shape (num_states, num_params).
#             b (numpy.ndarray): Vector b with shape (num_states, 1).
#         """
#         num_states, num_params = A.shape
#         b = b.reshape(num_states, 1)

#         # Compute a priori error
#         e = b - A @ self.x  # Shape (num_states, 1)

#         # Compute Kalman gain
#         P_A = self.P @ A.T  # Shape (num_params, num_states)
#         gain_denominator = self.forgetting_factor * np.eye(A.shape[0]) + A @ P_A  # Shape (num_states, num_states)
#         K = P_A @ np.linalg.inv(gain_denominator)  # Shape (num_params, num_states)

#         # Update parameter estimate
#         self.x += K @ e  # Shape (num_params, 1)

#         # Update covariance matrix (fixed formula)
#         self.P = (self.P - K @ A @ self.P) / self.forgetting_factor

#         # Update power estimates and forgetting factor
#         self.sigma_e2 = self.beta * self.sigma_e2 + (1 - self.beta) * np.linalg.norm(e)**2
#         self.forgetting_factor = 1 - (self.sigma_v2 / self.sigma_e2)

#     def predict(self):
#         """
#         Get the current parameter estimate x.

#         Returns:
#             numpy.ndarray: The current estimate of x, shape (num_params, 1).
#         """
#         return self.x
    
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
        # print(x,y)
        # Innovation covariance
        S = (x @ (self.P * x.T)) + self.R
        # Kalman gain
        K = (self.P * x.T) @ np.linalg.inv(S)  # K is 1x13

        # Update state estimate
        self.theta = self.theta + (K @ y).item()
        # print(self.theta)
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
        return self.x
    
class REK:
    def __init__(self):
        pass
    
    def iterate(self, A, b, x0, num_iter, tol):
        
        m, n = A.shape

        x_k = x0.copy()
        z_k = b.copy()

        A_fro_sq = np.linalg.norm(A, 'fro') ** 2
        row_norms_sq = np.sum(A**2, axis=1)
        col_norms_sq = np.sum(A**2, axis=0)

        iter = 0
        while(True):
            iter += 1

            i_k = np.random.choice(m, p=row_norms_sq / A_fro_sq)
            j_k = np.random.choice(n, p=col_norms_sq / A_fro_sq)

            A_ik = A[i_k, :]
            A_jk = A[:, j_k]

            if np.linalg.norm(A_jk) > 0:
                # import pdb; pdb.set_trace()
                z_k = z_k - ((np.dot(A_jk, z_k) / np.linalg.norm(A_jk)**2) * A_jk).reshape(-1,1)

            if np.linalg.norm(A_ik) > 0:
                x_k = x_k + (((b[i_k] - z_k[i_k] - np.dot(A_ik, x_k)) / np.linalg.norm(A_ik)**2) * A_ik).reshape(-1,1)

            if iter % (8 * min(m,n)) == 0:
                if np.linalg.norm(x_k) == 0:
                    break
                term1 = np.linalg.norm(A @ x_k - (b - z_k)) / (np.linalg.norm(A, 'fro')**2 * np.linalg.norm(x_k))
                term2 = np.linalg.norm(A.T @ z_k) / (np.linalg.norm(A, 'fro')**2 * np.linalg.norm(x_k))

                if term1 <= tol and term2 <= tol:
                    break

            if iter > num_iter:
                print("out of iter")
                break

        return x_k


class DEKA:
    def __init__(self, beta=0):
        """
        Initializes the DEKA solver.

        Args:
            beta (float): Momentum parameter, typically between 0 and 1.
        """
        self.beta = beta

    def iterate(self, A, b, x0, num_iterations, tol=0.01):
        """
        Solves the linear system Ax = b using the DEKA algorithm.

        Args:
            A (np.ndarray): The matrix A (num_states x num_params).
            b (np.ndarray): The vector b (num_states x 1).
            x0 (np.ndarray): The initial guess for x (num_params x 1).
            num_iterations (int): The maximum number of iterations.
            tol (float): The tolerance for convergence.

        Returns:
            np.ndarray: The solution vector x (num_params x 1).
        """

        x_k = x0
        x_prev = x_k.copy()

        # Create a mask to ignore rows where A is all zeros
        row_mask = np.any(A != 0, axis=1)  # True for nonzero rows, False for zero rows

        if not np.any(row_mask):  # If all rows are zero, return x_k immediately
            print("A has only zero rows, returning initial guess")
            return x_k

        if A.shape[0] == 0:
            return x_k
        
        # Apply the mask to A and b
        A = A[row_mask]  # Keep only nonzero rows
        b = b[row_mask]  # Keep corresponding b values

        for k in range(num_iterations):
            # import pdb; pdb.set_trace()
            residual = b - A @ x_k
            residual_norm_sq = np.linalg.norm(residual)**2

            # Compute epsilon_k
            A_row_norms_sq = np.sum(A**2, axis=1) + 1e-10
            # if np.any(A_row_norms_sq == 0):
            #     print("A row norm zero")
            #     return x_k

            if np.linalg.norm(residual) < tol:
                if k < 5:
                    print("DEKA converged in", k, "iterations")
                    print(np.linalg.norm(residual))
                    return x_k*0
                else:
                    print("DEKA converged in", k, "iterations")
                    return x_k 

            max_ratio = np.max(np.abs(residual.flatten())**2 / A_row_norms_sq)
            fro_norm_A_sq = np.linalg.norm(A, 'fro')**2
            epsilon_k = 0.5 * (max_ratio / residual_norm_sq + 1 / fro_norm_A_sq)

            # Determine the index set tau_k
            tau_k = []
            for i in range(A_row_norms_sq.shape[0]):
                if (np.abs(residual[i])**2 / A_row_norms_sq[i]) >= epsilon_k * residual_norm_sq:
                    tau_k.append(i)
            tau_k = np.array(tau_k)

            # Compute eta_k
            if not tau_k.size:
                print("Empty tau_k")
                return x_k

            eta_k = np.zeros_like(b)
            for i in tau_k:
                eta_k[i] = residual[i]

            # Update x_{k+1}
            A_T_eta_k = A.T @ eta_k

            # if np.linalg.norm(A_T_eta_k)**2 == 0:
            #     print("DEKA converged in", k, "iterations")
            #     return x_k

            x_next = x_k + (eta_k.T @ residual) / (np.linalg.norm(A_T_eta_k)**2) * A_T_eta_k + self.beta * (x_k - x_prev)

            # Check for convergence
            # if np.linalg.norm(x_next - x_k) < 0.0001:
            #     print("DEKA converged in", k, "iterations")
            #     return x_next

            # Update variables for the next iteration
            x_prev = x_k
            x_k = x_next

        print("DEKA did not converge within", num_iterations, "iterations")
        return x_k
    
class TestDEKA(unittest.TestCase):
    def test_square_matrix(self):
        A = np.array([[2, 1], [5, 7]], dtype=float)
        b = np.array([[11], [13]], dtype=float)
        x0 = np.array([[1], [1]], dtype=float)
        deka = DEKA()
        x = deka.iterate(A, b, x0, num_iterations=100, tol=1e-6)
        self.assertTrue(np.allclose(A @ x, b, atol=1e-5))

    def test_underdetermined_system(self):
        A = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        b = np.array([[14], [32]], dtype=float)
        x0 = np.array([[0], [0], [0]], dtype=float)
        deka = DEKA()
        x = deka.iterate(A, b, x0, num_iterations=100, tol=1e-6)
        self.assertTrue(np.allclose(A @ x, b, atol=1e-5))

    def test_overdetermined_system(self):
        A = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        b = np.array([[3], [7], [11]], dtype=float)
        x0 = np.array([[0], [0]], dtype=float)
        deka = DEKA()
        x = deka.iterate(A, b, x0, num_iterations=100, tol=1e-6)
        self.assertTrue(np.allclose(A @ x, b, atol=1e-5))

    def test_random_system(self):
        A = np.random.rand(10, 5)
        x = np.random.rand(5, 1)
        b = np.dot(A, x)
        x0 = np.zeros((5, 1))
        deka = DEKA()
        x = deka.iterate(A, b, x0, num_iterations=100, tol=1e-6)
        self.assertTrue(np.allclose(A @ x, b, atol=1e-5))
        
    def test_empty_tau_k(self):
        A = np.array([[1, 0], [0, 1]], dtype=float)
        b = np.array([[0], [0]], dtype=float)  # b is in the null space of A
        x0 = np.array([[1], [1]], dtype=float)  # Initial guess
        deka = DEKA()
        x = deka.iterate(A, b, x0, num_iterations=100, tol=1e-6)
        # In this case, we expect the algorithm to return the initial guess
        # because the residual is zero, leading to an empty tau_k.
        self.assertTrue(np.allclose(x, x0, atol=1e-5))

    # def test_square_matrix(self):
    #     A = np.array([[2, 1], [5, 7]], dtype=float)
    #     b = np.array([[11], [13]], dtype=float)
    #     x0 = np.array([[1], [1]], dtype=float)
    #     rk = RK()
    #     x = rk.iterate(A, b, x0, num_iterations=100, tol=1e-6)
    #     self.assertTrue(np.allclose(A @ x, b, atol=1e-5))

    # def test_underdetermined_system(self):
    #     A = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    #     b = np.array([[14], [32]], dtype=float)
    #     x0 = np.array([[0], [0], [0]], dtype=float)
    #     rk = RK()
    #     x = rk.iterate(A, b, x0, num_iterations=100, tol=1e-6)
    #     self.assertTrue(np.allclose(A @ x, b, atol=1e-5))

    # def test_overdetermined_system(self):
    #     A = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
    #     b = np.array([[3], [7], [11]], dtype=float)
    #     x0 = np.array([[0], [0]], dtype=float)
    #     rk = RK()
    #     x = rk.iterate(A, b, x0, num_iterations=100, tol=1e-6)
    #     self.assertTrue(np.allclose(A @ x, b, atol=1e-5))

    # def test_random_system(self):
    #     A = np.random.rand(10, 5)
    #     x = np.random.rand(5, 1)
    #     b = np.dot(A, x)
    #     x0 = np.zeros((5, 1))
    #     rk = RK()
    #     x = rk.iterate(A, b, x0, num_iterations=100, tol=1e-6)
    #     self.assertTrue(np.allclose(A @ x, b, atol=1e-5))
        
    # def test_empty_tau_k(self):
    #     A = np.array([[1, 0], [0, 1]], dtype=float)
    #     b = np.array([[0], [0]], dtype=float)  # b is in the null space of A
    #     x0 = np.array([[1], [1]], dtype=float)  # Initial guess
    #     rk = RK()
    #     x = rk.iterate(A, b, x0, num_iterations=100, tol=1e-6)
    #     # In this case, we expect the algorithm to return the initial guess
    #     # because the residual is zero, leading to an empty tau_k.
    #     self.assertTrue(np.allclose(x, x0, atol=1e-5))

if __name__ == '__main__':
    unittest.main()