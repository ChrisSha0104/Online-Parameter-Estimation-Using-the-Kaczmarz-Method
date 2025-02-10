import math
import matplotlib.pyplot as plt
import autograd.numpy as np
import autograd.numpy as sqrt
from autograd.numpy.linalg import norm
from autograd.numpy.linalg import inv
from autograd import jacobian
from autograd.test_util import check_grads
from collections import deque
np.set_printoptions(precision=4, suppress=True)
# Note: autograd does not work with np.block

from quadrotor_dynamics import Quadrotor
from double_pendulum_dynamics import DoublePendulum
from double_pen_LQR import LQRController_pen
from LQR_controller import LQRController
from estimation_methods import RK, RLS, EKF, DEKA, REK
from utilities import visualize_trajectory, visualize_trajectory_with_est

import argparse

class OnlineParamEst:
    def __init__(self):
        # Quadrotor Tasks:
        self.quadrotor = Quadrotor()
        self.quadrotor_controller = LQRController(self.quadrotor.delta_x_quat)
        self.double_pendulum =  DoublePendulum() 
        self.double_pendulum_controller = LQRController_pen() #TODO: use one controller

    def simulate_quadrotor_hover_with_Naive_MPC(self, update_status: str, NSIM: int =200):
        # initialize quadrotor parameters
        theta = np.array([self.quadrotor.mass,self.quadrotor.g])

        # get tasks goals
        x_nom, u_nom = self.quadrotor.get_hover_goals(theta[0], theta[1], self.quadrotor.kt)
        check_grads(self.quadrotor.quad_dynamics_rk4, modes=['rev'], order=2)(x_nom, u_nom, self.quadrotor.mass, self.quadrotor.g)
        Anp, Bnp = self.quadrotor.get_linearized_dynamics(x_nom, u_nom, theta[0], theta[1])
        Q, R = self.quadrotor_controller.get_QR_bryson()
        self.quadrotor_controller.update_linearized_dynamics(Anp, Bnp, Q, R)

        # randomly perturb initial state
        x0 = np.copy(x_nom)
        x0[0:3] += np.array([0.2, 0.2, -0.2])  # disturbed initial position
        x0[3:7] = self.quadrotor.rptoq(np.array([1.0, 0.0, 0.0]))  # disturbed initial attitude #TODO: move math methods to utilities class
        print("Perturbed Intitial State: ")
        print(x0)

        x_all = []
        u_all = []
        theta_all = []

        x_curr = np.copy(x0)
        x_prev = np.copy(x0)

        change_params = False
        changing_steps = np.random.choice(range(20,180),size=3, replace=False)

        # simulate the dynamics with the LQR controller
        for i in range(NSIM):
            # change mass
            if i in changing_steps:
                mass_update_scale = np.random.uniform(1,2)
                gravity_update_scale = np.random.uniform(1,2)
                theta = np.array([self.quadrotor.mass * mass_update_scale,self.quadrotor.g * gravity_update_scale])

            # MPC controller
            if update_status == "immediate_update":
                if i in changing_steps:
                    x_nom, u_nom = self.quadrotor.get_hover_goals(theta[0], theta[1], self.quadrotor.kt)
                    Anp, Bnp = self.quadrotor.get_linearized_dynamics(x_nom, u_nom, theta[0], theta[1])
                    self.quadrotor_controller.update_linearized_dynamics(Anp, Bnp, Q, R)
                u_curr = self.quadrotor_controller.compute(x_curr, x_nom, u_nom)
            elif update_status == "never_update":
                u_curr = self.quadrotor_controller.compute(x_curr, x_nom, u_nom)
            elif update_status == "late_update":
                if i in changing_steps+10: # when to tell controller the right parameters
                    x_nom, u_nom = self.quadrotor.get_hover_goals(theta[0], theta[1], self.quadrotor.kt)
                    Anp, Bnp = self.quadrotor.get_linearized_dynamics(x_nom, u_nom, theta[0], theta[1])
                    self.quadrotor_controller.update_linearized_dynamics(Anp, Bnp, Q, R)
                u_curr = self.quadrotor_controller.compute(x_curr, x_nom, u_nom)
            else:
                print("invalid update status. Choose from 'immediate_update', 'never_update', or 'late_update'.")
                exit()

            print("step: ", i, "\n", 
                  "u_k: ", u_curr, "\n", 
                  "x_k: ", x_prev[:3], "\n", 
                  "theta_k: ", theta, "\n", 
                  )

            # postponing the dynamics model by telling it the correct parameters after several steps
            x_prev = x_curr
            x_curr = self.quadrotor.quad_dynamics_rk4(x_curr, u_curr, theta[0], theta[1])

            x_curr = x_curr.reshape(x_curr.shape[0]).tolist()
            u_curr = u_curr.reshape(u_curr.shape[0]).tolist()

            x_all.append(x_curr)
            u_all.append(u_curr)
            theta_all.append(theta)
        return x_all, u_all, theta_all
    
    def simulate_quadrotor_hover_with_RLS(self, NSIM: int =200): #TODO: add RLS parameters here!
        # initialize quadrotor parameters
        theta = np.array([self.quadrotor.mass,self.quadrotor.g])
        theta_hat = np.copy(theta)

        # get tasks goals
        x_nom, u_nom = self.quadrotor.get_hover_goals(theta[0], theta[1], self.quadrotor.kt)
        check_grads(self.quadrotor.quad_dynamics_rk4, modes=['rev'], order=2)(x_nom, u_nom, self.quadrotor.mass, self.quadrotor.g)
        Anp, Bnp = self.quadrotor.get_linearized_dynamics(x_nom, u_nom, theta[0], theta[1])
        Q, R = self.quadrotor_controller.get_QR_bryson()
        self.quadrotor_controller.update_linearized_dynamics(Anp, Bnp, Q, R)

        # randomly perturb initial state
        x0 = np.copy(x_nom)
        x0[0:3] += np.array([0.2, 0.2, -0.2])  # disturbed initial position
        x0[3:7] = self.quadrotor.rptoq(np.array([1.0, 0.0, 0.0]))  # disturbed initial attitude #TODO: move math methods to utilities class
        print("Perturbed Intitial State: ")
        print(x0)

        # get initial control and state
        u_curr = self.quadrotor_controller.compute(x0, x_nom, u_nom)
        x_curr = np.copy(x0)

        x_all = []
        u_all = []
        theta_all = []
        theta_hat_all = []

        changing_steps = [50]#np.random.choice(range(20,180),size=2, replace=False)
        process_noise_std=np.array([0.00001, 0.0001])

        df_dm = jacobian(self.quadrotor.quad_dynamics_rk4, 2)
        df_dg = jacobian(self.quadrotor.quad_dynamics_rk4, 3)

        rls = RLS(num_params=2)

        # simulate the dynamics with the LQR controller
        for i in range(NSIM):
            # change mass
            if i % 10 == 0:
                theta += np.random.normal(0, process_noise_std, size=theta.shape)
            # update goals
            x_nom, u_nom = self.quadrotor.get_hover_goals(theta_hat[0], theta_hat[1], self.quadrotor.kt) #TODO: use moving ave filter?
            Anp, Bnp = self.quadrotor.get_linearized_dynamics(x_nom, u_nom, theta_hat[0], theta_hat[1])
            self.quadrotor_controller.update_linearized_dynamics(Anp, Bnp, Q, R)

            # compute controls
            
            u_curr = self.quadrotor_controller.compute(x_curr, x_nom, u_nom)                    # at t=k

            # step
            x_prev = x_curr                                                                     # at t=k
            x_curr = self.quadrotor.quad_dynamics_rk4(x_prev, u_curr, theta[0], theta[1])       # at t=k+1

            # estimate parameters
            theta_hat_prev = theta_hat                                                          # at t=k
            A = np.column_stack((df_dm(x_prev, u_curr, theta_hat_prev[0], theta_hat_prev[1]),  # (num_states, num_params)
                                 df_dg(x_prev, u_curr, theta_hat_prev[0], theta_hat_prev[1])))
            f_at_theta_prev = self.quadrotor.quad_dynamics_rk4(x_prev, u_curr, theta_hat_prev[0], theta_hat_prev[1]) # (num_states, 1)
            b = x_curr - f_at_theta_prev # (num_states, 1)

            # print("normed residual at step: ", i, " is " np.linalg.norm(df_dtheta_at_theta_prev @ theta_hat- b))
            # if i == 0:
            #     rls.initialize(df_dtheta_at_theta_prev, b)
            rls.update(A, b)
            delta_theta_hat = np.copy(rls.predict().reshape(2))                                       # at t=k+1
            
            # using LSQ SOLN
            # delta_theta_hat = np.linalg.lstsq(A,b)[0].reshape(2)
            theta_hat = delta_theta_hat + theta_hat_prev

            print("step: ", i, "\n", 
                  "u_k: ", u_curr, "\n", 
                  "x_k: ", x_prev[:3], "\n", 
                  "theta_k: ", theta, "\n", 
                  "theta_hat_k: ", theta_hat, "\n",
                  "delta_theta_hat", delta_theta_hat, "\n",
                #   "x_k+1: ", x_curr[:3], 
                  )
            # print("A: ", A)
            # print("b: ", b)

            x_curr = x_curr.reshape(x_curr.shape[0]).tolist() #TODO: x one step in front of controls
            u_curr = u_curr.reshape(u_curr.shape[0]).tolist()

            x_all.append(x_curr)
            u_all.append(u_curr)
            theta_all.append(theta)
            theta_hat_all.append(theta_hat)

        return x_all, u_all, theta_all, theta_hat_all
    # def simulate_quadrotor_hover_with_RLS(self, NSIM: int =200): #TODO: add RLS parameters here!
    #     # initialize quadrotor parameters
    #     theta = np.array([self.quadrotor.mass,self.quadrotor.g])
    #     theta_hat = np.copy(theta)

    #     # get tasks goals
    #     x_nom, u_nom = self.quadrotor.get_hover_goals(theta[0], theta[1], self.quadrotor.kt)
    #     check_grads(self.quadrotor.quad_dynamics_rk4, modes=['rev'], order=2)(x_nom, u_nom, self.quadrotor.mass, self.quadrotor.g)
    #     Anp, Bnp = self.quadrotor.get_linearized_dynamics(x_nom, u_nom, theta[0], theta[1])
    #     Q, R = self.quadrotor_controller.get_QR_bryson()
    #     self.quadrotor_controller.update_linearized_dynamics(Anp, Bnp, Q, R)

    #     # randomly perturb initial state
    #     x0 = np.copy(x_nom)
    #     x0[0:3] += np.array([0.2, 0.2, -0.2])  # disturbed initial position
    #     x0[3:7] = self.quadrotor.rptoq(np.array([1.0, 0.0, 0.0]))  # disturbed initial attitude #TODO: move math methods to utilities class
    #     print("Perturbed Intitial State: ")
    #     print(x0)

    #     # get initial control and state
    #     u_curr = self.quadrotor_controller.compute(x0, x_nom, u_nom)
    #     x_curr = np.copy(x0)

    #     x_all = []
    #     u_all = []
    #     theta_all = []
    #     theta_hat_all = []

    #     changing_steps = np.random.choice(range(20,180),size=2, replace=False)

    #     df_dm = jacobian(self.quadrotor.quad_dynamics_rk4, 2)
    #     df_dg = jacobian(self.quadrotor.quad_dynamics_rk4, 3)

    #     rls = RLS(num_params=2)

    #     # simulate the dynamics with the LQR controller
    #     for i in range(NSIM):
    #         # change mass
    #         if i in changing_steps:
    #             update_scale = np.random.uniform(1,2) # runs into singular if I change scale to [1/3,4]
    #             theta = np.array([self.quadrotor.mass*update_scale,self.quadrotor.g*update_scale])
    #         # update goals
    #         x_nom, u_nom = self.quadrotor.get_hover_goals(theta_hat[0], theta_hat[1], self.quadrotor.kt)
    #         Anp, Bnp = self.quadrotor.get_linearized_dynamics(x_nom, u_nom, theta_hat[0], theta_hat[1])
    #         self.quadrotor_controller.update_linearized_dynamics(Anp, Bnp, Q, R)

    #         # compute controls
    #         u_curr = self.quadrotor_controller.compute(x_curr, x_nom, u_nom)                    # at t=k

    #         # step
    #         x_prev = x_curr                                                                     # at t=k
    #         x_curr = self.quadrotor.quad_dynamics_rk4(x_prev, u_curr, theta[0], theta[1])       # at t=k+1

    #         # estimate parameters
    #         theta_hat_prev = theta_hat                                                          # at t=k
    #         df_dtheta_at_theta_prev = np.column_stack((df_dm(x_prev, u_curr, theta_hat_prev[0], theta_hat_prev[1]),  # (num_states, num_params)
    #                                                    df_dg(x_prev, u_curr, theta_hat_prev[0], theta_hat_prev[1])))
    #         f_at_theta_prev = self.quadrotor.quad_dynamics_rk4(x_prev, u_curr, theta_hat_prev[0], theta_hat_prev[1]) # (num_states, 1)
    #         b = x_curr - f_at_theta_prev + df_dtheta_at_theta_prev @ theta_hat_prev # (num_states, 1)

    #         # print("normed residual at step: ", i, " is " np.linalg.norm(df_dtheta_at_theta_prev @ theta_hat- b))
    #         # if i == 0:
    #         #     rls.initialize(df_dtheta_at_theta_prev, b)
    #         rls.update(df_dtheta_at_theta_prev, b)
    #         theta_hat = np.copy(rls.predict().reshape(2))                                       # at t=k+1
            
    #         # using LSQ SOLN
    #         # theta_hat = np.linalg.lstsq(df_dtheta_at_theta_prev,b)[0].reshape(2)

    #         print("step: ", i, "\n", 
    #               "u_k: ", u_curr, "\n", 
    #               "x_k: ", x_prev[:3], "\n", 
    #               "theta_k: ", theta, "\n", 
    #               "theta_hat_k: ", theta_hat, "\n",
    #             #   "x_k+1: ", x_curr[:3], 
    #               )
    #         # print("A: ", df_dtheta_at_theta_prev)
    #         # print("b: ", b)

    #         x_curr = x_curr.reshape(x_curr.shape[0]).tolist() #TODO: x one step in front of controls
    #         u_curr = u_curr.reshape(u_curr.shape[0]).tolist()

    #         import pdb; pdb.set_trace()

    #         x_all.append(x_curr)
    #         u_all.append(u_curr)
    #         theta_all.append(theta)
    #         theta_hat_all.append(theta_hat)

    #     return x_all, u_all, theta_all, theta_hat_all
    
    # def simulate_quadrotor_hover_with_RLS(self, NSIM: int =200): #TODO: add RLS parameters here!
    #     # initialize quadrotor parameters
    #     theta = np.array([self.quadrotor.mass,self.quadrotor.g])
    #     theta_hat = np.copy(theta)

    #     # get tasks goals
    #     x_nom, u_nom = self.quadrotor.get_hover_goals(theta[0], theta[1], self.quadrotor.kt)
    #     check_grads(self.quadrotor.quad_dynamics_rk4, modes=['rev'], order=2)(x_nom, u_nom, self.quadrotor.mass, self.quadrotor.g)
    #     Anp, Bnp = self.quadrotor.get_linearized_dynamics(x_nom, u_nom, theta[0], theta[1])
    #     Q, R = self.quadrotor_controller.get_QR_bryson()
    #     self.quadrotor_controller.update_linearized_dynamics(Anp, Bnp, Q, R)

    #     # randomly perturb initial state
    #     x0 = np.copy(x_nom)
    #     x0[0:3] += np.array([0.2, 0.2, -0.2])  # disturbed initial position
    #     x0[3:7] = self.quadrotor.rptoq(np.array([1.0, 0.0, 0.0]))  # disturbed initial attitude #TODO: move math methods to utilities class
    #     print("Perturbed Intitial State: ")
    #     print(x0)

    #     # get initial control and state
    #     u_curr = self.quadrotor_controller.compute(x0, x_nom, u_nom)
    #     x_curr = np.copy(x0)

    #     x_all = []
    #     u_all = []
    #     theta_all = []
    #     theta_hat_all = []

    #     changing_step = 50

    #     df_dm = jacobian(self.quadrotor.quad_dynamics_rk4, 2)
    #     df_dg = jacobian(self.quadrotor.quad_dynamics_rk4, 3)

    #     rls = RLS(num_params=2)

    #     # simulate the dynamics with the LQR controller
    #     for i in range(NSIM):
    #         # change mass
    #         if i == changing_step:
    #             theta = np.array([self.quadrotor.mass*4,self.quadrotor.g*4])
    #         # update goals
    #         x_nom, u_nom = self.quadrotor.get_hover_goals(theta_hat[0], theta_hat[1], self.quadrotor.kt)
    #         Anp, Bnp = self.quadrotor.get_linearized_dynamics(x_nom, u_nom, theta_hat[0], theta_hat[1])
    #         self.quadrotor_controller.update_linearized_dynamics(Anp, Bnp, Q, R)

    #         # compute controls
    #         u_curr = self.quadrotor_controller.compute(x_curr, x_nom, u_nom)

    #         # compute next states -> a step in time
    #         x_prev = x_curr
    #         x_curr = self.quadrotor.quad_dynamics_rk4(x_prev, u_curr, theta[0], theta[1])
    #         print("step: ", i, "\n", "controls: ", u_curr, "\n", "position: ", x_curr[:3], "\n", "theta: ", theta, "\n", "theta_hat: ", theta_hat)

    #         # estimate parameters
    #         theta_hat_prev = theta_hat
    #         df_dtheta_at_theta_prev = np.column_stack((df_dm(x_prev, u_curr, theta_hat_prev[0], theta_hat_prev[1]),  # (num_states, num_params)
    #                                                    df_dg(x_prev, u_curr, theta_hat_prev[0], theta_hat_prev[1])))
    #         f_at_theta_prev = self.quadrotor.quad_dynamics_rk4(x_prev, u_curr, theta_hat_prev[0], theta_hat_prev[1]) # (num_states, 1)
    #         b = x_curr - f_at_theta_prev + df_dtheta_at_theta_prev @ theta_hat_prev # (num_states, 1)

    #         # print("normed residual at step: ", i, " is " np.linalg.norm(df_dtheta_at_theta_prev @ theta_hat- b))
    #         rls.update(df_dtheta_at_theta_prev, b)
    #         theta_hat = np.copy(rls.predict().reshape(2))

    #         # print("A: ", df_dtheta_at_theta_prev)
    #         # print("b: ", b)

    #         x_curr = x_curr.reshape(x_curr.shape[0]).tolist()
    #         u_curr = u_curr.reshape(u_curr.shape[0]).tolist()

    #         x_all.append(x_curr)
    #         u_all.append(u_curr)
    #         theta_all.append(theta)
    #         theta_hat_all.append(theta_hat)

    #     return x_all, u_all, theta_all, theta_hat_all
    
    def simulate_quadrotor_hover_with_KM(self, NSIM: int = 200):
        """
        Simulates a quadrotor hover maneuver with the Kaczmarz method (KM) for parameter estimation.

        Args:
            NSIM (int): Number of simulation steps.

        Returns:
            tuple: x_all, u_all, theta_hat_all, theta_all (lists of states, controls, estimated parameters, and true parameters).
        """

        theta = np.array([self.quadrotor.mass, self.quadrotor.g])
        theta_hat = np.copy(theta)

        # get tasks goals
        x_nom, u_nom = self.quadrotor.get_hover_goals(theta[0], theta[1], self.quadrotor.kt)
        Anp, Bnp = self.quadrotor.get_linearized_dynamics(x_nom, u_nom, theta[0], theta[1])
        Q, R = self.quadrotor_controller.get_QR_bryson()
        self.quadrotor_controller.update_linearized_dynamics(Anp, Bnp, Q, R)

        # randomly perturb initial state
        x0 = np.copy(x_nom)
        x0[0:3] += np.array([0.2, 0.2, -0.2])  # disturbed initial position
        x0[3:7] = self.quadrotor.rptoq(np.array([1.0, 0.0, 0.0]))  # disturbed initial attitude
        print("Perturbed Initial State: ")
        print(x0)

        # get initial control and state
        u_curr = self.quadrotor_controller.compute(x0, x_nom, u_nom)
        x_curr = np.copy(x0)
        
        x_all = []
        u_all = []
        theta_all = []
        theta_hat_all = []

        rk = RK()     

        changing_steps = np.random.choice(range(20, 180), size=3, replace=False) # TODO: avoid frequent/consecutive changes

        debug: bool = False
        debug_history_length = 0

        df_dm = jacobian(self.quadrotor.quad_dynamics_rk4, 2)
        df_dg = jacobian(self.quadrotor.quad_dynamics_rk4, 3)

        # Simulation loop
        for i in range(NSIM):
            # Change system parameters at specific step
            if i in changing_steps:
                mass_update_scale = np.random.uniform(1, 2)
                gravity_update_scale = np.random.uniform(1, 2)
                theta = np.array([self.quadrotor.mass * mass_update_scale, self.quadrotor.g * gravity_update_scale])
                # debug_history_length = 4

            # update goals based on the estimated states
            x_nom, u_nom = self.quadrotor.get_hover_goals(theta_hat[0], theta_hat[1], self.quadrotor.kt)
            Anp, Bnp = self.quadrotor.get_linearized_dynamics(x_nom, u_nom, theta_hat[0], theta_hat[1])
            self.quadrotor_controller.update_linearized_dynamics(Anp, Bnp, Q, R)
            
            # compute controls
            u_curr = self.quadrotor_controller.compute(x_curr, x_nom, u_nom)

            # step
            x_prev = x_curr  # at t=k
            x_curr = self.quadrotor.quad_dynamics_rk4(x_prev, u_curr, theta[0], theta[1])

            # estimate parameters
            theta_hat_prev = theta_hat                                                          # at t=k
            A = np.column_stack((df_dm(x_prev, u_curr, theta_hat_prev[0], theta_hat_prev[1]),  # (num_states, num_params)
                                 df_dg(x_prev, u_curr, theta_hat_prev[0], theta_hat_prev[1])))
            f_at_theta_prev = self.quadrotor.quad_dynamics_rk4(x_prev, u_curr, theta_hat_prev[0], theta_hat_prev[1]) # (num_states, 1)
            b = x_curr - f_at_theta_prev # (num_states, 1)

            theta_km = rk.iterate(A, b, theta_hat_prev.reshape(-1,1), 10000, 0.01).reshape(2) + theta_hat_prev
            delta_theta_hat = np.linalg.lstsq(A, b)[0]
            theta_hat = delta_theta_hat + theta_hat_prev

            if debug_history_length>0:
                print("A: ", A)
                print("b: ", b.T)
                print("gt residual: ", np.linalg.norm(np.dot(A,theta)-b))
                debug_history_length -= 1

            print("step: ", i, "\n",
                  "u_k: ", u_curr, "\n",
                  "x_k: ", x_prev[:3], "\n",
                  "theta_k: ", theta, "\n",
                  "theta_km: ", theta_km, "\n",
                  "theta_lstsq: ", theta_hat
                  )

            x_curr = x_curr.reshape(x_curr.shape[0]).tolist()
            u_curr = u_curr.reshape(u_curr.shape[0]).tolist()

            # Store results
            x_all.append(x_curr)
            u_all.append(u_curr)
            theta_all.append(theta)
            theta_hat_all.append(theta_hat)

        return x_all, u_all, theta_hat_all, theta_all
    
    def entropy_score(self, df_dtheta_window, b_window):
        """
        Compute entropy-based score for a given window of (df_dtheta, b).
        Lower entropy means a more informative motion sample.

        Args:
            df_dtheta_window (list or np.ndarray): List of df_dtheta matrices or a stacked matrix.
            b_window (list or np.ndarray): List of b vectors or a stacked matrix.

        Returns:
            float: Entropy score.
        """

        # Convert lists into proper numpy arrays if necessary
        if isinstance(df_dtheta_window, list):
            df_dtheta_window = np.vstack(df_dtheta_window)  # Shape (num_samples * num_states, num_params)
        if isinstance(b_window, list):
            b_window = np.vstack(b_window)  # Shape (num_samples * num_states, 1)

        # Ensure they are 2D before stacking
        if df_dtheta_window.ndim == 1:
            df_dtheta_window = df_dtheta_window.reshape(-1, 1)
        if b_window.ndim == 1:
            b_window = b_window.reshape(-1, 1)

        # Stack and ensure it's 2D
        stacked_data = np.hstack((df_dtheta_window, b_window))  # Shape (num_samples * num_states, num_params + 1)
        cov_matrix = np.cov(stacked_data, rowvar=False)  # Compute covariance matrix

        # Compute entropy as log determinant of covariance matrix
        entropy = 0.5 * np.linalg.slogdet(cov_matrix)[1]
        return entropy

    
    def simulate_quadrotor_hover_with_DEKA(self, NSIM: int =200):
        theta = np.array([self.quadrotor.mass,self.quadrotor.g])
        theta_hat = np.copy(theta)
        delta_theta_hat = np.array([[0.0,0.0]])

        # get tasks goals
        x_nom, u_nom = self.quadrotor.get_hover_goals(theta[0], theta[1], self.quadrotor.kt)
        # check_grads(self.quadrotor.quad_dynamics_rk4, modes=['rev'], order=2)(x_nom, u_nom, self.quadrotor.mass, self.quadrotor.g)
        Anp, Bnp = self.quadrotor.get_linearized_dynamics(x_nom, u_nom, theta[0], theta[1])
        Q, R = self.quadrotor_controller.get_QR_bryson()
        self.quadrotor_controller.update_linearized_dynamics(Anp, Bnp, Q, R)

        # randomly perturb initial state
        x0 = np.copy(x_nom)
        x0[0:3] += np.array([0.2, 0.2, -0.2])  # disturbed initial position
        x0[3:7] = self.quadrotor.rptoq(np.array([1.0, 0.0, 0.0]))  # disturbed initial attitude #TODO: move math methods to utilities class
        print("Perturbed Intitial State: ")
        print(x0)

        # get initial control and state
        u_curr = self.quadrotor_controller.compute(x0, x_nom, u_nom)
        x_curr = np.copy(x0)

        x_all = []
        u_all = []
        theta_all = []
        theta_hat_all = []

        changing_steps = [30,100]#np.random.choice(range(20,180),size=1, replace=False)
        

        df_dm = jacobian(self.quadrotor.quad_dynamics_rk4, 2)
        df_dg = jacobian(self.quadrotor.quad_dynamics_rk4, 3)

        deka = DEKA()

        # Priority queue for history (fixed size)
        queue_size = 3
        A_queue, b_queue = deque(maxlen=queue_size), deque(maxlen=queue_size)

        # Keep track of scores for removal
        score_queue = deque(maxlen=queue_size)

        # Simulation loop
        for i in range(NSIM):
            # Change system parameters at specific step
            if i in changing_steps:
                mass_update_scale = np.random.uniform(1, 3/2)
                gravity_update_scale = np.random.uniform(1, 3/2)
                theta = np.array([self.quadrotor.mass * mass_update_scale, self.quadrotor.g * gravity_update_scale])

            # update goals
            x_nom, u_nom = self.quadrotor.get_hover_goals(theta_hat[0], theta_hat[1], self.quadrotor.kt)
            Anp, Bnp = self.quadrotor.get_linearized_dynamics(x_nom, u_nom, theta_hat[0], theta_hat[1])
            self.quadrotor_controller.update_linearized_dynamics(Anp, Bnp, Q, R)

            # compute controls
            u_curr = self.quadrotor_controller.compute(x_curr, x_nom, u_nom) 

            # step
            x_prev = x_curr                                                                     # at t=k
            x_curr = self.quadrotor.quad_dynamics_rk4(x_prev, u_curr, theta[0], theta[1]) 

            # estimate parameters
            theta_hat_prev = theta_hat                                                          # at t=k
            A = np.column_stack((df_dm(x_prev, u_curr, theta_hat_prev[0], theta_hat_prev[1]),  # (num_states, num_params)
                                 df_dg(x_prev, u_curr, theta_hat_prev[0], theta_hat_prev[1])))
            f_at_theta_prev = self.quadrotor.quad_dynamics_rk4(x_prev, u_curr, theta_hat_prev[0], theta_hat_prev[1]) # (num_states, 1)
            b = x_curr - f_at_theta_prev # (num_states, 1)

            # # Add to history queue
            # A_queue.append(A)
            # b_queue.append(b)

            # if i == 30 or i == 100:
            #     A_queue.clear()
            #     b_queue.clear() 

            # lsq_delta = np.linalg.lstsq(A, b)[0]
            # if len(A_queue) == queue_size:
            #     A_hist = np.vstack(A_queue)  # Shape (13*3, 2)
            #     b_hist = np.vstack(b_queue)  # Shape (13*3, 1)

            delta_theta_hat_prev = delta_theta_hat
            delta_theta_hat = deka.iterate(A.reshape(-1,2), b.reshape(-1,1), delta_theta_hat_prev.reshape(-1,1), 1000, 1e-4).reshape(2)
            theta_hat = delta_theta_hat + theta_hat_prev

            print("step: ", i, "\n",
                  "u_k: ", u_curr, "\n",
                  "x_k: ", x_prev[:3], "\n",
                  "theta_k: ", theta, "\n",
                #   "theta_km: ", theta_km, "\n",
                  "theta_lstsq: ", theta_hat, "\n",
                  )
            
            x_curr = x_curr.reshape(x_curr.shape[0]).tolist()
            u_curr = u_curr.reshape(u_curr.shape[0]).tolist()
            
            # Store results
            x_all.append(x_curr)
            u_all.append(u_curr)
            theta_all.append(theta)
            theta_hat_all.append(theta_hat)

        return x_all, u_all, theta_hat_all, theta_all
    
    def simulate_quadrotor_hover_with_REK(self, NSIM: int =200):
        theta = np.array([self.quadrotor.mass,self.quadrotor.g])
        theta_hat = np.copy(theta)
        delta_theta_hat = np.array([[0.0,0.0]])

        # get tasks goals
        x_nom, u_nom = self.quadrotor.get_hover_goals(theta[0], theta[1], self.quadrotor.kt)
        # check_grads(self.quadrotor.quad_dynamics_rk4, modes=['rev'], order=2)(x_nom, u_nom, self.quadrotor.mass, self.quadrotor.g)
        Anp, Bnp = self.quadrotor.get_linearized_dynamics(x_nom, u_nom, theta[0], theta[1])
        Q, R = self.quadrotor_controller.get_QR_bryson()
        self.quadrotor_controller.update_linearized_dynamics(Anp, Bnp, Q, R)

        # randomly perturb initial state
        x0 = np.copy(x_nom)
        x0[0:3] += np.array([0.2, 0.2, -0.2])  # disturbed initial position
        x0[3:7] = self.quadrotor.rptoq(np.array([1.0, 0.0, 0.0]))  # disturbed initial attitude #TODO: move math methods to utilities class
        print("Perturbed Intitial State: ")
        print(x0)

        # get initial control and state
        u_curr = self.quadrotor_controller.compute(x0, x_nom, u_nom)
        x_curr = np.copy(x0)

        x_all = []
        u_all = []
        theta_all = []
        theta_hat_all = []

        changing_steps = [30,100]#np.random.choice(range(20,180),size=1, replace=False)
        

        df_dm = jacobian(self.quadrotor.quad_dynamics_rk4, 2)
        df_dg = jacobian(self.quadrotor.quad_dynamics_rk4, 3)

        rek = REK()

        # Priority queue for history (fixed size)
        queue_size = 3
        A_queue, b_queue = deque(maxlen=queue_size), deque(maxlen=queue_size)

        # Keep track of scores for removal
        score_queue = deque(maxlen=queue_size)

        # Simulation loop
        for i in range(NSIM):
            # Change system parameters at specific step
            if i in changing_steps:
                mass_update_scale = np.random.uniform(1, 3/2)
                gravity_update_scale = np.random.uniform(1, 3/2)
                theta = np.array([self.quadrotor.mass * mass_update_scale, self.quadrotor.g * gravity_update_scale])

            # update goals
            x_nom, u_nom = self.quadrotor.get_hover_goals(theta_hat[0], theta_hat[1], self.quadrotor.kt)
            Anp, Bnp = self.quadrotor.get_linearized_dynamics(x_nom, u_nom, theta_hat[0], theta_hat[1])
            self.quadrotor_controller.update_linearized_dynamics(Anp, Bnp, Q, R)

            # compute controls
            u_curr = self.quadrotor_controller.compute(x_curr, x_nom, u_nom) 

            # step
            x_prev = x_curr                                                                     # at t=k
            x_curr = self.quadrotor.quad_dynamics_rk4(x_prev, u_curr, theta[0], theta[1]) 

            # estimate parameters
            theta_hat_prev = theta_hat                                                          # at t=k
            A = np.column_stack((df_dm(x_prev, u_curr, theta_hat_prev[0], theta_hat_prev[1]),  # (num_states, num_params)
                                 df_dg(x_prev, u_curr, theta_hat_prev[0], theta_hat_prev[1])))
            f_at_theta_prev = self.quadrotor.quad_dynamics_rk4(x_prev, u_curr, theta_hat_prev[0], theta_hat_prev[1]) # (num_states, 1)
            b = x_curr - f_at_theta_prev # (num_states, 1)

            # # Add to history queue
            # A_queue.append(A)
            # b_queue.append(b)

            # if i == 30 or i == 100:
            #     A_queue.clear()
            #     b_queue.clear() 

            # lsq_delta = np.linalg.lstsq(A, b)[0]
            # if len(A_queue) == queue_size:
            #     A_hist = np.vstack(A_queue)  # Shape (13*3, 2)
            #     b_hist = np.vstack(b_queue)  # Shape (13*3, 1)

            delta_theta_hat_prev = delta_theta_hat
            delta_theta_hat = rek.iterate(A.reshape(-1,2), b.reshape(-1,1), delta_theta_hat_prev.reshape(-1,1), 1000, 1e-5).reshape(2)
            theta_hat = delta_theta_hat + theta_hat_prev

            print("step: ", i, "\n",
                  "u_k: ", u_curr, "\n",
                  "x_k: ", x_prev[:3], "\n",
                  "theta_k: ", theta, "\n",
                  "theta_km: ", theta_hat, "\n",
                  )
            
            x_curr = x_curr.reshape(x_curr.shape[0]).tolist()
            u_curr = u_curr.reshape(u_curr.shape[0]).tolist()
            
            # Store results
            x_all.append(x_curr)
            u_all.append(u_curr)
            theta_all.append(theta)
            theta_hat_all.append(theta_hat)

        return x_all, u_all, theta_hat_all, theta_all

    def simulate_double_pendulum_control_with_Naive_MPC(self, update_status: str, NSIM: int =400):
        # initialize quadrotor parameters
        theta = np.array([self.double_pendulum.m1,self.double_pendulum.g])
        #TODO continue to modify double pendulum
        # get tasks goals
        x_nom, u_nom = self.quadrotor.get_hover_goals(theta[0], theta[1], self.quadrotor.kt)
        check_grads(self.quadrotor.quad_dynamics_rk4, modes=['rev'], order=2)(x_nom, u_nom, self.quadrotor.mass, self.quadrotor.g)
        Anp, Bnp = self.quadrotor.get_linearized_dynamics(x_nom, u_nom, theta[0], theta[1])
        Q, R = self.quadrotor_controller.get_QR_bryson()
        self.quadrotor_controller.update_linearized_dynamics(Anp, Bnp, Q, R)

        # randomly perturb initial state
        x0 = np.copy(x_nom)
        x0[0:3] += np.array([0.2, 0.2, -0.2])  # disturbed initial position
        x0[3:7] = self.quadrotor.rptoq(np.array([1.0, 0.0, 0.0]))  # disturbed initial attitude #TODO: move math methods to utilities class
        print("Perturbed Intitial State: ")
        print(x0)

        x_all = []
        u_all = []
        theta_all = []

        x_curr = np.copy(x0)

        change_params = False
        changing_step = 50

        # simulate the dynamics with the LQR controller
        for i in range(NSIM):
            # change mass
            if i == changing_step:
                change_params = True
                theta = np.array([self.quadrotor.mass * 4,self.quadrotor.g * 2])

            # MPC controller
            if update_status == "immediate_update":
                if i == changing_step:
                    x_nom, u_nom = self.quadrotor.get_hover_goals(theta[0], theta[1], self.quadrotor.kt)
                    Anp, Bnp = self.quadrotor.get_linearized_dynamics(x_nom, u_nom, theta[0], theta[1])
                    self.quadrotor_controller.update_linearized_dynamics(Anp, Bnp, Q, R)
                u_curr = self.quadrotor_controller.compute(x_curr, x_nom, u_nom)
            elif update_status == "never_update":
                u_curr = self.quadrotor_controller.compute(x_curr, x_nom, u_nom)
            elif update_status == "late_update":
                if i == changing_step+10: # when to tell controller the right parameters
                    x_nom, u_nom = self.quadrotor.get_hover_goals(theta[0], theta[1], self.quadrotor.kt)
                    Anp, Bnp = self.quadrotor.get_linearized_dynamics(x_nom, u_nom, theta[0], theta[1])
                    self.quadrotor_controller.update_linearized_dynamics(Anp, Bnp, Q, R)
                u_curr = self.quadrotor_controller.compute(x_curr, x_nom, u_nom)
            else:
                print("invalid update status. Choose from 'immediate_update', 'never_update', or 'late_update'.")
                exit()

            print("step: ", i, "\n", "controls: ", u_curr, "\n", "position: ", x_curr[:3])

            # postponing the dynamics model by telling it the correct parameters after several steps
            if change_params:
                x_curr = self.quadrotor.quad_dynamics_rk4(x_curr, u_curr, theta[0], theta[1])
            else:
                x_curr = self.quadrotor.quad_dynamics_rk4(x_curr, u_curr, theta[0], theta[1])

            x_curr = x_curr.reshape(x_curr.shape[0]).tolist()
            u_curr = u_curr.reshape(u_curr.shape[0]).tolist()

            x_all.append(x_curr)
            u_all.append(u_curr)
            theta_all.append(theta)
        return x_all, u_all, theta_all

    def main(): 
        parser = argparse.ArgumentParser(description="An online parameter estimation interface with different robots, control tasks, estimation methods.")
        parser.add_argument("--robot", type=str, required=True, help="choose from quadrotor, cartpole, double_pendulum")
        parser.add_argument("--task", type=str, required=True, help="choose from hover, tracking, control") 
        parser.add_argument("--method", type=str, required=True, help="choose from Naive_MPC, KM, RLS, EKF, DEKA")  
        parser.add_argument("--update_type", type=str, required=False, help="choose from immediate_update, late_update, never_update") 
        args = parser.parse_args()

        method_name = f"simulate_{args.robot}_{args.task}_with_{args.method}"
        ParamEst = OnlineParamEst()

        if hasattr(ParamEst, method_name):
            if args.method == "Naive_MPC":              # without parameter estimation
                update_type = args.update_type
                x_history, u_history, theta_history = getattr(ParamEst, method_name)(update_type)
                title = f"{args.robot} {args.task} with {args.method} and update type: " + update_type 
                visualize_trajectory(x_history, u_history, theta_history, title) #TODO: different state space for different robots!
            else:                                       # with parameter estimation 
                x_history, u_history, theta_history, theta_hat_history = getattr(ParamEst, method_name)()
                title = f"Online Parameter Estimation for {args.robot} {args.task} with {args.method}"
                visualize_trajectory_with_est(x_history, u_history, theta_history, theta_hat_history, title)
        else: 
            print(f"Error: Method {method_name} not found. Use --help to see available option.")

        # if args.method == "Naive_MPC":
        #     ParamEst = OnlineParamEst()
        #     update_status = "immediate_update"
        #     x_history, u_history, theta_history = ParamEst.simulate_quadrotor_hover_with_MPC(update_status)
        #     title = "Online Parameter Estimation with Naive MPC; update status - " + update_status 
        #     visualize_trajectory_hover(x_history, u_history, theta_history, title)
        # elif args.method == "RLS":
        #     ParamEst = OnlineParamEst()
        #     # x_history, u_history, theta_history, theta_hat_history = ParamEst.simulate_quadrotor_hover_with_RLS()
        #     x_history, u_history, theta_history, theta_hat_history = ParamEst.simulate_quadrotor_hover_with_RLS_dynamical()
        #     title = "Online Parameter Estimation with RLS"
        #     visualize_trajectory_hover_with_est(x_history, u_history, theta_history, theta_hat_history, title)
        # elif args.method == "KM":
        #     ParamEst = OnlineParamEst()
        #     # x_history, u_history, theta_history, theta_hat_history = ParamEst.simulate_quadrotor_hover_with_RLS()
        #     x_history, u_history, theta_history, theta_hat_history = ParamEst.simulate_quadrotor_hover_with_KM()
        #     title = "Online Parameter Estimation with KM"
        #     visualize_trajectory_hover_with_est(x_history, u_history, theta_history, theta_hat_history, title)
        # elif args.method == "DEKA":
        #     ParamEst = OnlineParamEst()
        #     # x_history, u_history, theta_history, theta_hat_history = ParamEst.simulate_quadrotor_hover_with_RLS()
        #     x_history, u_history, theta_history, theta_hat_history = ParamEst.simulate_quadrotor_hover_with_DEKA()
        #     title = "Online Parameter Estimation with DEKA"
        #     visualize_trajectory_hover_with_est(x_history, u_history, theta_history, theta_hat_history, title)
        # else:
        #     print("Invalid method. Choose from Naive_MPC, KM, RLS, EKF")

if __name__ == "__main__":
    OnlineParamEst.main()