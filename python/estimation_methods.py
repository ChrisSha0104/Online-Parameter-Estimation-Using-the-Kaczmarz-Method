import autograd.numpy as np
import unittest

class LMS:
    """
    Basic least mean squares (LMS) method.

    This is a gradient method and implements both normalized and unnormalized approaches.
    """

    def __init__(self, num_params, theta_hat=None, gain=0.01, normalized=True, bias=1e-6):
        self.n = num_params
        self.gamma = gain
        self.bias = bias
        self.normalized = True
        self.theta_hat = theta_hat if theta_hat is not None else np.zeros((num_params, 1))

    def update(self, A, b):
        Q = np.ones((self.n, 1)) * self.gamma

        if self.normalized:
            Q /= np.linalg.norm(A, axis=1) ** 2 + self.bias

        K = A @ Q
        self.theta_hat += K @ (b - A @ self.theta_hat)

    def predict(self):
        """Note: always call update() first to get updated theta."""
        return self.theta_hat

class RLS:
    """RLS with adaptive forgetting factor (lambda)."""
    def __init__(self, num_params, theta_hat=None, forgetting_factor=0.98, c=1000):
        """
        :param num_params: Number of parameters to estimate.
        :param theta_hat: Initial estimate of parameters, otherwise set to zeros.
        :param forgetting_factor: Forgetting factor (typically referred to as lambda).
        :param c: Constant factor for initial covariance matrix scale.
        """
        self.n = num_params
        self.P = np.eye(num_params) * c
        self.theta_hat = theta_hat if theta_hat is not None else np.zeros((num_params, 1))
        self.lambda_ = forgetting_factor

    def update(self, A, b):
        """
        :param A: Jacobian of system w.r.t. parameters.
        :param b: Measurement vector.
        """
        # import pdb; pdb.set_trace()
        K = self.P @ A.T @ np.linalg.inv(A @ self.P @ A.T + self.lambda_ * np.eye(A.shape[0]))
        self.theta_hat += K @ (b - A @ self.theta_hat)
        self.P = (self.P - K @ A @ self.P) / self.lambda_

    def predict(self):
        """Note: always call update() first to get updated theta."""
        return self.theta_hat


class AdaptiveLambdaRLS:
    """
    RLS with adaptive forgetting factor (lambda).

        lambda = 1 - alpha / (1 + ||b - A * theta||)

    Adjusts lambda such that forgetting is reduced when parameters are stable and increased when changing.
    """

    def __init__(self, num_params, theta_hat=None, forgetting_factor=0.98, alpha=0.01, c=1000):
        """
        :param num_params: Number of parameters to estimate.
        :param theta_hat: Initial estimate of parameters, otherwise set to zeros.
        :param forgetting_factor: Forgetting factor (typically referred to as lambda).
        :param forgetting_adaptation: Tuning factor for adaptive forgetting.
        :param c: Constant factor for initial covariance matrix scale.
        """
        self.n = num_params
        self.P = np.eye(num_params) * c
        self.theta_hat = theta_hat if theta_hat is not None else np.zeros((num_params, 1))
        self.lambda_ = forgetting_factor
        self.alpha = alpha

    def update(self, A, b):
        """
        :param A: Jacobian of system w.r.t. parameters.
        :param b: Measurement vector.
        """
        K = self.P @ A.T @ np.linalg.inv(A @ self.P @ A.T + self.lambda_ * np.eye(A.shape[0]))
        self.theta_hat += K @ (b - A @ self.theta_hat)
        self.P = (self.P - K @ A @ self.P) / self.lambda_
        self.lambda_ = 1 - self.alpha / (1 + np.linalg.norm(b - A @ self.theta_hat))

    def predict(self):
        """Note: always call update() first to get updated theta."""
        return self.theta_hat
    
# class EKF:
#     def __init__(self, process_noise=1e-3, measurement_noise=1e-1):
#         self.theta =0.                    # Initial gravity estimate
#         self.P = 100.0                       # Initial covariance
#         self.Q = process_noise                # Process noise covariance
#         self.R = np.eye(13) * measurement_noise  # Measurement noise covariance

#     def predict(self):
#         # Prediction step
#         self.P = self.P + self.Q

#     def update(self, x, y):
#         """
#         x: 13x1 Jacobian vector (sensitivity of dynamics to gravity)
#         y: 13x1 observation vector (state differences)
#         """
#         # Ensure x and y are column vectors
#         x = np.reshape(x, (13, 1))
#         y = np.reshape(y, (13, 1))
#         # print(x,y)
#         # Innovation covariance
#         S = (x @ (self.P * x.T)) + self.R
#         # Kalman gain
#         K = (self.P * x.T) @ np.linalg.inv(S)  # K is 1x13

#         # Update state estimate
#         self.theta = self.theta + (K @ y).item()
#         # print(self.theta)
#         # Update covariance
#         self.P = self.P - (K @ S @ K.T).item()
#         return self.theta

class EKF:
    """
    Basic Extended Kalman Filter

    Assumes that the process noise and measurement noise are known, and that their Jacobians are I. Additionally, the process model jacobian is assumed to be I.
    """
    def __init__(self, num_params, process_noise, measurement_noise, theta_hat=None, c=1000):
        """
        :param num_params: Number of parameters to estimate (p).
        :param theta_hat: Initial estimate of parameters, otherwise set to zeros (p x 1 vector).
        :param forgetting_factor: Forgetting factor (typically referred to as lambda).
        :param c: Constant factor for initial covariance matrix scale.
        :param P: Estimation error covariance (p x p matrix)
        :param Q: Process noise covariance (d x p matrix)
        :param R: Measurement noise covariance (d x d matrix)
        """
        self.n = num_params
        self.theta_hat = theta_hat if theta_hat is not None else np.zeros((num_params, 1))
        self.P = np.eye(num_params) * c
        self.Q = process_noise
        self.R = measurement_noise

    def update(self, A, b):
        """
        :param A: Jacobian of system w.r.t. parameters (d x p matrix).
        :param b: Measurement vector (d x 1 vector).
        """
        self.P += self.Q
        K = self.P @ A.T @ np.linalg.inv(A @ self.P @ A.T + self.R)
        self.theta_hat += K @ (b - A @ self.theta_hat)
        self.P -= K @ A @ self.P

    def predict(self):
        return self.theta_hat
    
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


class RKAS:
    def __init__(self, alpha=0.99, epsilon=1e-8):
        """
        Randomized Kaczmarz with Adaptive Stepsizes (RKAS).
        :param tol: Convergence tolerance.
        :param max_iter: Maximum number of iterations.
        """
        self.alpha = alpha
        self.epsilon = epsilon

    def solve(self, A, b, x0, max_iter=1000, tol=1e-4):
        """
        Solves the inconsistent system Ax = b using RKAS.
        :param A: (m x n) coefficient matrix.
        :param b: (m x 1) right-hand side vector.
        :return: Approximate solution x.
        """
        m, n = A.shape
        x = x0.copy()  # Initial solution x^0 = 0
        r = -b  # Initial residual r^0 = Ax^0 - b = -b

        # # Compute row selection probabilities
        # row_norms = np.linalg.norm(A, axis=1) ** 2
        # probabilities = row_norms / np.sum(row_norms)  # Pr(i_k = i)

        for k in range(max_iter):
                       # Compute exponential weighting for the rows
            row_norms = np.linalg.norm(A, axis=1)**2

            # Normalize by subtracting the maximum (robust to large values)
            row_norms -= np.max(row_norms)

            # Scale by alpha (can be adjusted for better performance)
            row_norms *= self.alpha

            # Add epsilon to prevent division by zero or log(0)
            row_norms += self.epsilon
            
            #Absolute value of row norm
            row_norms = np.abs(row_norms)

            # Calculate probabilities (should be stable now)
            probabilities = row_norms / np.sum(row_norms)

            # Ensure probabilities sum to 1 (handle potential rounding errors)
            probabilities /= np.sum(probabilities)

            # Step 1: Select row index i_k with probability Pr(i_k = i)
            i_k = np.random.choice(m, p=probabilities)

            # Extract row A_{i_k,:} and corresponding residual component
            A_ik = A[i_k, :].reshape(1, -1)  # Ensure row vector
            AAT_ik = np.dot(A, A_ik.T)  # Compute A * A^T for row i_k

            # Step 2: Compute adaptive step size
            if np.linalg.norm(AAT_ik) > 1e-6:
                alpha_k = np.dot(AAT_ik.T, r) / np.linalg.norm(AAT_ik) ** 2
            else:
                alpha_k = 0

            # Step 3: Update solution and residual
            x = x - alpha_k * A_ik.T
            r = r - alpha_k * AAT_ik

            # Check for convergence
            if np.linalg.norm(r) < tol:
                print(f"Converged in {k} iterations")
                break
        return x

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

            if k % 10 == 0 and np.linalg.norm(b - A @ x_k) < tol:
                print("residual new: ", np.linalg.norm(b - A @ x_k))
                print("residual previous: ", np.linalg.norm(b - A @ x_prev))
                if k < 10:
                    print("DEKA converged in", k, "iterations")
                    return x_k/3
                else:
                    print("DEKA converged in", k, "iterations")
                    return x_k 
                
        print("DEKA did not converge within", num_iterations, "iterations, residual: ", np.linalg.norm(b - A @ x_k))
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
