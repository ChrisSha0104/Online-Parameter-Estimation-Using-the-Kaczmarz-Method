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
from estimation_methods import RK, RLS, EKF
from utilities import visualize_trajectory, visualize_trajectory_hover_with_est, visualize_trajectory_hover

import argparse

class OnlineParamEst:
    def __init__(self):
        # Quadrotor Tasks:
        self.quadrotor = Quadrotor()
        self.quadrotor_controller = LQRController(self.quadrotor.delta_x_quat)
        self.double_pendulum =  DoublePendulum() 
        self.double_pendulum_controller = LQRController_pen()

    def simulate_quadrotor_hover_with_MPC(self, update_status: str, NSIM: int =400):
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
    
    def simulate_quadrotor_hover_with_RLS_dynamical(self, NSIM: int =200): #TODO: add RLS parameters here!
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

        changing_steps = np.random.choice(range(20,180),size=2, replace=False)

        df_dm = jacobian(self.quadrotor.quad_dynamics_rk4, 2)
        df_dg = jacobian(self.quadrotor.quad_dynamics_rk4, 3)

        rls = RLS(num_params=2)

        # simulate the dynamics with the LQR controller
        for i in range(NSIM):
            # change mass
            if i in changing_steps:
                update_scale = np.random.uniform(1/2,3) # runs into singular if I change scale to [1/3,4]
                theta = np.array([self.quadrotor.mass*update_scale,self.quadrotor.g*update_scale])
            # update goals
            x_nom, u_nom = self.quadrotor.get_hover_goals(theta_hat[0], theta_hat[1], self.quadrotor.kt)
            Anp, Bnp = self.quadrotor.get_linearized_dynamics(x_nom, u_nom, theta_hat[0], theta_hat[1])
            self.quadrotor_controller.update_linearized_dynamics(Anp, Bnp, Q, R)

            # compute controls
            u_curr = self.quadrotor_controller.compute(x_curr, x_nom, u_nom)                    # at t=k

            # step
            x_prev = x_curr                                                                     # at t=k
            x_curr = self.quadrotor.quad_dynamics_rk4(x_prev, u_curr, theta[0], theta[1])       # at t=k+1

            # estimate parameters
            theta_hat_prev = theta_hat                                                          # at t=k
            df_dtheta_at_theta_prev = np.column_stack((df_dm(x_prev, u_curr, theta_hat_prev[0], theta_hat_prev[1]),  # (num_states, num_params)
                                                       df_dg(x_prev, u_curr, theta_hat_prev[0], theta_hat_prev[1])))
            f_at_theta_prev = self.quadrotor.quad_dynamics_rk4(x_prev, u_curr, theta_hat_prev[0], theta_hat_prev[1]) # (num_states, 1)
            b = x_curr - f_at_theta_prev + df_dtheta_at_theta_prev @ theta_hat_prev # (num_states, 1)

            # print("normed residual at step: ", i, " is " np.linalg.norm(df_dtheta_at_theta_prev @ theta_hat- b))
            # if i == 0:
            #     rls.initialize(df_dtheta_at_theta_prev, b)
            rls.update(df_dtheta_at_theta_prev, b)
            theta_hat = np.copy(rls.predict().reshape(2))                                       # at t=k+1
            
            # using LSQ SOLN
            # theta_hat = np.linalg.lstsq(df_dtheta_at_theta_prev,b)[0].reshape(2)

            print("step: ", i, "\n", 
                  "u_k: ", u_curr, "\n", 
                  "x_k: ", x_prev[:3], "\n", 
                  "theta_k: ", theta, "\n", 
                  "theta_hat_k: ", theta_hat, "\n",
                  "x_k+1: ", x_curr[:3], 
                  )
            # print("A: ", df_dtheta_at_theta_prev)
            # print("b: ", b)

            x_curr = x_curr.reshape(x_curr.shape[0]).tolist() #TODO: x one step in front of controls
            u_curr = u_curr.reshape(u_curr.shape[0]).tolist()

            x_all.append(x_curr)
            u_all.append(u_curr)
            theta_all.append(theta)
            theta_hat_all.append(theta_hat)

        return x_all, u_all, theta_all, theta_hat_all
    
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

        changing_step = 50

        df_dm = jacobian(self.quadrotor.quad_dynamics_rk4, 2)
        df_dg = jacobian(self.quadrotor.quad_dynamics_rk4, 3)

        rls = RLS(num_params=2)

        # simulate the dynamics with the LQR controller
        for i in range(NSIM):
            # change mass
            if i == changing_step:
                theta = np.array([self.quadrotor.mass*4,self.quadrotor.g*4])
            # update goals
            x_nom, u_nom = self.quadrotor.get_hover_goals(theta_hat[0], theta_hat[1], self.quadrotor.kt)
            Anp, Bnp = self.quadrotor.get_linearized_dynamics(x_nom, u_nom, theta_hat[0], theta_hat[1])
            self.quadrotor_controller.update_linearized_dynamics(Anp, Bnp, Q, R)

            # compute controls
            u_curr = self.quadrotor_controller.compute(x_curr, x_nom, u_nom)

            # compute next states -> a step in time
            x_prev = x_curr
            x_curr = self.quadrotor.quad_dynamics_rk4(x_prev, u_curr, theta[0], theta[1])
            print("step: ", i, "\n", "controls: ", u_curr, "\n", "position: ", x_curr[:3], "\n", "theta: ", theta, "\n", "theta_hat: ", theta_hat)

            # estimate parameters
            theta_hat_prev = theta_hat
            df_dtheta_at_theta_prev = np.column_stack((df_dm(x_prev, u_curr, theta_hat_prev[0], theta_hat_prev[1]),  # (num_states, num_params)
                                                       df_dg(x_prev, u_curr, theta_hat_prev[0], theta_hat_prev[1])))
            f_at_theta_prev = self.quadrotor.quad_dynamics_rk4(x_prev, u_curr, theta_hat_prev[0], theta_hat_prev[1]) # (num_states, 1)
            b = x_curr - f_at_theta_prev + df_dtheta_at_theta_prev @ theta_hat_prev # (num_states, 1)

            # print("normed residual at step: ", i, " is " np.linalg.norm(df_dtheta_at_theta_prev @ theta_hat- b))
            rls.update(df_dtheta_at_theta_prev, b)
            theta_hat = np.copy(rls.predict().reshape(2))

            # print("A: ", df_dtheta_at_theta_prev)
            # print("b: ", b)

            x_curr = x_curr.reshape(x_curr.shape[0]).tolist()
            u_curr = u_curr.reshape(u_curr.shape[0]).tolist()

            x_all.append(x_curr)
            u_all.append(u_curr)
            theta_all.append(theta)
            theta_hat_all.append(theta_hat)

        return x_all, u_all, theta_all, theta_hat_all
    
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

        changing_steps = np.random.choice(range(20, 180), size=5, replace=False) # TODO: avoid frequent/consecutive changes

        debug: bool = False
        debug_history_length = 0

        # Priority queue for history (fixed size)
        queue_size = 20
        df_dtheta_queue, b_queue = deque(maxlen=queue_size), deque(maxlen=queue_size)

        df_dm = jacobian(self.quadrotor.quad_dynamics_rk4, 2)
        df_dg = jacobian(self.quadrotor.quad_dynamics_rk4, 3)

        # Simulation loop
        for i in range(NSIM):
            # Change system parameters at specific step
            if i in changing_steps:
                mass_update_scale = np.random.uniform(1, 3)
                gravity_update_scale = np.random.uniform(1, 3)
                theta = np.array([self.quadrotor.mass * mass_update_scale, self.quadrotor.g * gravity_update_scale])
                theta_all.append(theta)
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
            theta_hat_prev = theta_hat
            df_dtheta_at_theta_prev = np.column_stack((df_dm(x_prev, u_curr, theta_hat_prev[0], theta_hat_prev[1]), # = A
                                                    df_dg(x_prev, u_curr, theta_hat_prev[0], theta_hat_prev[1])))
            f_at_theta_prev = self.quadrotor.quad_dynamics_rk4(x_prev, u_curr, theta_hat_prev[0], theta_hat_prev[1])
            b = x_curr - f_at_theta_prev + df_dtheta_at_theta_prev @ theta_hat_prev

            if i == 0:
                theta_hat_prev = theta_hat
                theta_km = theta_hat
            else:
                # Add new data to the queues
                # df_dtheta_queue.append(df_dtheta_at_theta_prev)
                # b_queue.append(b.reshape(-1,1))

                # Use the regular Kaczmarz method (no entropy-based selection)
                theta_hat_prev = theta_hat
                # theta_hat = rk.iterate(df_dtheta_queue, b_queue, theta_hat_prev.reshape(-1,1), 100, 0.01).reshape(2)
                theta_km = rk.iterate(df_dtheta_at_theta_prev, b, theta_hat_prev.reshape(-1,1), 1000, 0.001).reshape(2)
                theta_hat = np.linalg.lstsq(df_dtheta_at_theta_prev, b)[0]

                if debug_history_length>0:
                    print("A: ", df_dtheta_at_theta_prev)
                    print("b: ", b.T)
                    print("gt residual: ", np.linalg.norm(np.dot(df_dtheta_at_theta_prev,theta)-b))
                    debug_history_length -= 1

            print("step: ", i, "\n",
                #   "u_k: ", u_curr, "\n",
                #   "x_k: ", x_prev[:3], "\n",
                  "theta_k: ", theta, "\n",
                  "theta_km: ", theta_km, "\n",
                  "theta_lstsq: ", theta_hat
                  )

            x_curr = x_curr.reshape(x_curr.shape[0]).tolist()
            u_curr = u_curr.reshape(u_curr.shape[0]).tolist()

            # Store results
            x_all.append(x_curr)
            u_all.append(u_curr)
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

        changing_steps = np.random.choice(range(20,180),size=3, replace=False)
        

        df_dm = jacobian(self.quadrotor.quad_dynamics_rk4, 2)
        df_dg = jacobian(self.quadrotor.quad_dynamics_rk4, 3)

        rk = RK()

        # Priority queue for history (fixed size)
        queue_size = 20
        df_dtheta_queue, b_queue = deque(maxlen=queue_size), deque(maxlen=queue_size)

        # Keep track of scores for removal
        score_queue = deque(maxlen=queue_size)

        # Simulation loop
        for i in range(NSIM):
            # Change system parameters at specific step
            if i in changing_steps:
                update_scale = np.random.uniform(1/2,3) # runs into singular if I change scale to [1/3,4]
                theta = np.array([self.quadrotor.mass*update_scale,self.quadrotor.g*update_scale])
                theta_all.append(theta)

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
            theta_hat_prev = theta_hat    
            df_dtheta_at_theta_prev = np.column_stack((df_dm(x_prev, u_curr, theta_hat_prev[0], theta_hat_prev[1]),  # (num_states, num_params)
                                                       df_dg(x_prev, u_curr, theta_hat_prev[0], theta_hat_prev[1])))
            f_at_theta_prev = self.quadrotor.quad_dynamics_rk4(x_prev, u_curr, theta_hat_prev[0], theta_hat_prev[1]) # (num_states, 1)
            b = x_curr - f_at_theta_prev + df_dtheta_at_theta_prev @ theta_hat_prev # (num_states, 1)

            if i == 0:
                theta_hat_prev = theta_hat
            else:
                # Reshape b to be (13, 1) before passing to entropy_score
                b = b.reshape(-1, 1)  # -1 automatically calculates the appropriate dimension  
    
                # Calculate score for new data point
                print("Shape of df_dtheta_at_theta_prev:", df_dtheta_at_theta_prev.shape)
                print("Shape of b:", b.shape)
                new_score = self.entropy_score([df_dtheta_at_theta_prev], [b])

                # Add new data to the queues
                df_dtheta_queue.append(df_dtheta_at_theta_prev)
                b_queue.append(b)
                score_queue.append(new_score)

                # Remove oldest data point if queue exceeds maxlen and criteria met
                if len(df_dtheta_queue) > queue_size:
                    # Remove the least informative element based on score if not drastic
                    worst_index = np.argmax(score_queue)

                    del df_dtheta_queue[worst_index]
                    del b_queue[worst_index]
                    del score_queue[worst_index]

                theta_hat_prev = theta_hat
                theta_hat = rk.iterate(np.vstack(df_dtheta_queue), np.vstack(b_queue), theta_hat_prev.reshape(-1,1), 1000, 0.01).reshape(2) 

            print("step: ", i, "\n", 
                  "u_k: ", u_curr, "\n", 
                  "x_k: ", x_prev[:3], "\n", 
                  "theta_k: ", theta, "\n", 
                  "theta_hat_k: ", theta_hat, "\n",
                  "x_k+1: ", x_curr[:3], 
                  )
            
            x_curr = x_curr.reshape(x_curr.shape[0]).tolist()
            u_curr = u_curr.reshape(u_curr.shape[0]).tolist()
            
            # Store results
            x_all.append(x_curr)
            u_all.append(u_curr)
            theta_hat_all.append(theta_hat)

        return x_all, u_all, theta_hat_all, theta_all
    def simulate_pendulum_with_MPC(self, update_status: str, NSIM: int =400):
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
        parser = argparse.ArgumentParser(description="Example script with command-line arguments")
        parser.add_argument("--robot", type=str, required=False, help="choose from quadrotor, cartpole, doublependulum")
        parser.add_argument("--task", type=str, required=False, help="choose from hover, tracking, control") 
        parser.add_argument("--method", type=str, required=True, help="choose from Naive_MPC, KM, RLS, EKF")  
        args = parser.parse_args()

        if args.method == "Naive_MPC":
            ParamEst = OnlineParamEst()
            update_status = "immediate_update"
            x_history, u_history, theta_history = ParamEst.simulate_quadrotor_hover_with_MPC(update_status)
            title = "Online Parameter Estimation with Naive MPC; update status - " + update_status 
            visualize_trajectory_hover(x_history, u_history, theta_history, title)
        elif args.method == "RLS":
            ParamEst = OnlineParamEst()
            # x_history, u_history, theta_history, theta_hat_history = ParamEst.simulate_quadrotor_hover_with_RLS()
            x_history, u_history, theta_history, theta_hat_history = ParamEst.simulate_quadrotor_hover_with_RLS_dynamical()
            title = "Online Parameter Estimation with RLS"
            visualize_trajectory_hover_with_est(x_history, u_history, theta_history, theta_hat_history, title)
        elif args.method == "KM":
            ParamEst = OnlineParamEst()
            # x_history, u_history, theta_history, theta_hat_history = ParamEst.simulate_quadrotor_hover_with_RLS()
            x_history, u_history, theta_history, theta_hat_history = ParamEst.simulate_quadrotor_hover_with_KM()
            title = "Online Parameter Estimation with KM"
            visualize_trajectory_hover_with_est(x_history, u_history, theta_history, theta_hat_history, title)
        elif args.method == "DEKA":
            ParamEst = OnlineParamEst()
            # x_history, u_history, theta_history, theta_hat_history = ParamEst.simulate_quadrotor_hover_with_RLS()
            x_history, u_history, theta_history, theta_hat_history = ParamEst.simulate_quadrotor_hover_with_DEKA()
            title = "Online Parameter Estimation with DEKA"
            visualize_trajectory_hover_with_est(x_history, u_history, theta_history, theta_hat_history, title)
        else:
            print("Invalid method. Choose from Naive_MPC, KM, RLS, EKF")

if __name__ == "__main__":
    OnlineParamEst.main()