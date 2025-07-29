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

from python.quadrotor.quadrotor_dynamics_old import Quadrotor
from python.double_pendulum.double_pendulum_dynamics import DoublePendulum
from python.double_pendulum.double_pen_LQR import LQRController_pen
from python.common.LQR_controller import LQRController
from python.common.estimation_methods_old import *
from python.common.utilities import *

import argparse


class OnlineParamEst:
    def __init__(self):
        # Quadrotor Tasks:
        self.quadrotor = Quadrotor()
        self.quadrotor_controller = LQRController(self.quadrotor.delta_x_quat)
        self.double_pendulum =  DoublePendulum() 
        self.double_pendulum_controller = LQRController_pen() #TODO: use one controller

    def simulate_quadrotor_hover_with_Naive_MPC(self, update_status: str, NSIM: int =300):
        
        np.set_printoptions(suppress=False, precision=10)
        # initialize quadrotor parameters
        Ixx, Iyy, Izz = self.quadrotor.J[0,0], self.quadrotor.J[1,1], self.quadrotor.J[2,2]
        Ixy, Ixz, Iyz = self.quadrotor.J[0,1], self.quadrotor.J[0,2], self.quadrotor.J[1,2]
        theta = np.array([self.quadrotor.mass,Ixx,Ixy,Ixz,Iyy,Iyz,Izz])
        
        # get tasks goals
        x_nom, u_nom = self.quadrotor.get_hover_goals(theta)
        Anp, Bnp = self.quadrotor.get_linearized_dynamics(x_nom, u_nom, theta)
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
        changing_steps = [100] #np.random.choice(range(20,180),size=3, replace=False)

        # simulate the dynamics with the LQR controller
        for i in range(NSIM):
            # change mass
            if i in changing_steps:
                update_scale = np.random.uniform(1,2)
                theta *= update_scale

            # MPC controller
            if update_status == "immediate_update":
                if i in changing_steps:
                    x_nom, u_nom = self.quadrotor.get_hover_goals(theta)
                    Anp, Bnp = self.quadrotor.get_linearized_dynamics(x_nom, u_nom, theta)
                    self.quadrotor_controller.update_linearized_dynamics(Anp, Bnp, Q, R)
                u_curr = self.quadrotor_controller.compute(x_curr, x_nom, u_nom)
            elif update_status == "never_update":
                u_curr = self.quadrotor_controller.compute(x_curr, x_nom, u_nom)
            elif update_status == "late_update":
                if i == 120: # TODO: hardcoded
                    x_nom, u_nom = self.quadrotor.get_hover_goals(theta)
                    Anp, Bnp = self.quadrotor.get_linearized_dynamics(x_nom, u_nom, theta)
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
            x_curr = self.quadrotor.quad_dynamics_rk4(x_curr, u_curr, theta)

            x_curr = x_curr.reshape(x_curr.shape[0]).tolist()
            u_curr = u_curr.reshape(u_curr.shape[0]).tolist()

            x_all.append(x_curr)
            u_all.append(u_curr)
            theta_all.append(theta.copy())
        return x_all, u_all, theta_all 
    
    def simulate_quadrotor_hover_with_RLS(self, NSIM: int =300): #TODO: add RLS parameters here!
        # np.random.seed(42)
        self.quadrotor = Quadrotor()
        self.quadrotor_controller = LQRController(self.quadrotor.delta_x_quat)
        # initialize quadrotor parameters
        np.set_printoptions(suppress=False, precision=10)
        Ixx, Iyy, Izz = self.quadrotor.J[0,0], self.quadrotor.J[1,1], self.quadrotor.J[2,2]
        Ixy, Ixz, Iyz = self.quadrotor.J[0,1], self.quadrotor.J[0,2], self.quadrotor.J[1,2]
        # theta = np.array([self.quadrotor.mass,Ixx,Ixy,Ixz,Iyy,Iyz,Izz])
        theta = np.array([self.quadrotor.mass, Ixx,Ixy,Ixz,Iyy,Iyz,Izz])
        theta_hat = theta.copy()
       # print("RLS theta: ", theta)
        
        # get tasks goals
        x_nom, u_nom = self.quadrotor.get_hover_goals(theta_hat)
        Anp, Bnp = self.quadrotor.get_linearized_dynamics(x_nom, u_nom, theta_hat)
        Q, R = self.quadrotor_controller.get_QR_bryson()
        self.quadrotor_controller.update_linearized_dynamics(Anp, Bnp, Q, R)

        # randomly perturb initial state
        x0 = np.copy(x_nom)
        x0[0:3] += np.array([0.1, -0.15, -0.1])  # disturbed initial position
        x0[3:7] = self.quadrotor.rptoq(np.array([0.3, 0.4, 0.2]))  # disturbed initial attitude #TODO: move math methods to utilities class
        # print("Perturbed Intitial State: ")
        # print(x0)

        # get initial control and state
        u_curr = self.quadrotor_controller.compute(x0, x_nom, u_nom)
        x_curr = np.copy(x0)

        x_all = []
        u_all = []
        theta_all = []
        theta_hat_all = []

        changing_steps = [100]#np.random.choice(range(20,180),size=2, replace=False)
 
        rls = RLS(num_params=7, forgetting_factor=0.85, theta_hat=theta.copy().reshape(-1,1))
        n = 10
        A_tot = np.zeros((6*n,7))
        b_tot = np.zeros((6*n,1))

        # simulate the dynamics with the LQR controller
        for i in range(NSIM):
            # noising the parameters
            # if i % 10 == 0:
            #     theta += np.random.normal(0, process_noise_std, size=theta.shape)

            # Change system parameters at specific step
            if i in changing_steps:
                update_scale = np.random.uniform(2, 3)
                theta[0] *= update_scale
    
            # update goals
            x_nom, u_nom = self.quadrotor.get_hover_goals(theta_hat.copy())
            Anp, Bnp = self.quadrotor.get_linearized_dynamics(x_nom, u_nom, theta_hat.copy())
            self.quadrotor_controller.update_linearized_dynamics(Anp, Bnp, Q, R)

            # compute controls
            u_curr = self.quadrotor_controller.compute(x_curr, x_nom, u_nom)                    # at t=k

            # step
            if i > 0:
                process_noise_std = 0.02 * np.abs(theta[0])  # Process noise
                theta[0] += np.random.normal(0, process_noise_std)

            x_curr = self.quadrotor.quad_dynamics_rk4(x_curr, u_curr, theta)       # at t=k+1
            # formulate measurement model
            A = self.quadrotor.get_data_matrix(x_curr, self.quadrotor.quad_dynamics(x_curr, u_curr, theta))
            b = self.get_force_vector(x_curr, u_curr, theta)
            measurement_noise_std = 0.05 * np.abs(b)
            b += np.random.normal(0, measurement_noise_std, size = b.shape)

            A_tot = np.roll(A_tot, shift=3, axis=0)
            A_tot[:6] = A
            b_tot = np.roll(b_tot, shift=3, axis=0)
            b_tot[:6] = b

            if i % 5 == 0 and i > 0:
                theta_hat = rls.iterate(A_tot,b_tot).reshape(-1,)
            
            # print(A@(theta.reshape(-1,1)))
            # print(b)

            # print("step: ", i, "\n", 
                # "prediction_err: ", np.linalg.norm(theta-theta_hat)/7, "\n"
                #   )


            x_curr = x_curr.reshape(x_curr.shape[0]).tolist() #TODO: x one step in front of controls
            u_curr = u_curr.reshape(u_curr.shape[0]).tolist()

            x_all.append(x_curr)
            u_all.append(u_curr)
            theta_all.append(theta.copy())
            theta_hat_all.append(theta_hat.copy())

        return x_all, u_all, theta_all, theta_hat_all
    
    def simulate_quadrotor_hover_with_KF(self, Q_noise, R_noise, NSIM: int =300): #TODO: add RLS parameters here!
        # np.random.seed(42)
        self.quadrotor = Quadrotor()
        self.quadrotor_controller = LQRController(self.quadrotor.delta_x_quat)
        # initialize quadrotor parameters
        np.set_printoptions(suppress=False, precision=10)
        Ixx, Iyy, Izz = self.quadrotor.J[0,0], self.quadrotor.J[1,1], self.quadrotor.J[2,2]
        Ixy, Ixz, Iyz = self.quadrotor.J[0,1], self.quadrotor.J[0,2], self.quadrotor.J[1,2]
        # theta = np.array([self.quadrotor.mass,Ixx,Ixy,Ixz,Iyy,Iyz,Izz])
      #  print(Ixx)
        theta = np.array([self.quadrotor.mass, Ixx,Ixy,Ixz,Iyy,Iyz,Izz])
        theta_hat = theta.copy()
       # print("KF theta: ", theta)
        
        # get tasks goals
        x_nom, u_nom = self.quadrotor.get_hover_goals(theta_hat)
        Anp, Bnp = self.quadrotor.get_linearized_dynamics(x_nom, u_nom, theta_hat)
        Q, R = self.quadrotor_controller.get_QR_bryson()
        self.quadrotor_controller.update_linearized_dynamics(Anp, Bnp, Q, R)

        # randomly perturb initial state
        x0 = np.copy(x_nom)
        x0[0:3] += np.array([0.1, -0.15, -0.1])  # disturbed initial position
        x0[3:7] = self.quadrotor.rptoq(np.array([0.3, 0.4, 0.2]))  # disturbed initial attitude #TODO: move math methods to utilities class
       # print("Perturbed Intitial State: ")
        #print(x0)

        # get initial control and state
        u_curr = self.quadrotor_controller.compute(x0, x_nom, u_nom)
        x_curr = np.copy(x0)

        x_all = []
        u_all = []
        theta_all = []
        theta_hat_all = []

        changing_steps = [100]#np.random.choice(range(20,180),size=2, replace=False)

        n = 10
        A_tot = np.zeros((6*n,7))
        b_tot = np.zeros((6*n,1))
        Q_ekf = 1e-5 * np.eye(len(theta))
        R_ekf = 1e-5 * np.eye(6*n)

        kf = EKF(num_params=7, process_noise=Q_noise, measurement_noise=R_noise)#, theta_hat=theta.copy().reshape(-1,1))

        # simulate the dynamics with the LQR controller
        for i in range(NSIM):
            # noising the parameters
            # if i % 10 == 0:
            #     theta += np.random.normal(0, process_noise_std, size=theta.shape)

            # Change system parameters at specific step
            if i in changing_steps:
                update_scale = np.random.uniform(2, 3)
                theta[0] *= update_scale

            # update goals
            x_nom, u_nom = self.quadrotor.get_hover_goals(theta_hat.copy())
            Anp, Bnp = self.quadrotor.get_linearized_dynamics(x_nom, u_nom, theta_hat.copy())
            self.quadrotor_controller.update_linearized_dynamics(Anp, Bnp, Q, R)

            # compute controls
            u_curr = self.quadrotor_controller.compute(x_curr, x_nom, u_nom)                    # at t=k

            # step
            if i > 0 and i % 15 == 0:
                process_noise_std = 0.1 * np.abs(theta[0])  # Process noise
                theta[0] += np.random.normal(0, process_noise_std)
            x_curr = self.quadrotor.quad_dynamics_rk4(x_curr, u_curr, theta)      # at t=k+1
            # formulate measurement model
            A = self.quadrotor.get_data_matrix(x_curr, self.quadrotor.quad_dynamics(x_curr, u_curr, theta))
            b = self.quadrotor.get_force_vector(x_curr, u_curr, theta)
            measurement_noise_std = 0.05 * np.abs(b)
            b += np.random.normal(0, measurement_noise_std, size = b.shape)

            A_tot = np.roll(A_tot, shift=3, axis=0)
            A_tot[:6] = A
            b_tot = np.roll(b_tot, shift=3, axis=0)
            b_tot[:6] = b

            if i % 5 == 0 and i > 0:
                theta_hat = kf.iterate(A_tot,b_tot).reshape(-1,)
            
            # print(A@(theta.reshape(-1,1)))
            # print(b)

            # print("step: ", i, "\n", 
                # "prediction_err: ", np.linalg.norm(theta-theta_hat)/7, "\n"
                #   )

            x_curr = x_curr.reshape(x_curr.shape[0]).tolist() #TODO: x one step in front of controls
            u_curr = u_curr.reshape(u_curr.shape[0]).tolist()

            x_all.append(x_curr)
            u_all.append(u_curr)
            theta_all.append(theta.copy())
            theta_hat_all.append(theta_hat.copy())

        return x_all, u_all, theta_all, theta_hat_all

    def simulate_quadrotor_hover_with_DEKA(self, NSIM: int =300): #TODO: add RLS parameters here!
        # np.random.seed(42)
        self.quadrotor = Quadrotor()
        self.quadrotor_controller = LQRController(self.quadrotor.delta_x_quat)

        # initialize quadrotor parameters
        np.set_printoptions(suppress=False, precision=10)
        Ixx, Iyy, Izz = self.quadrotor.J[0,0], self.quadrotor.J[1,1], self.quadrotor.J[2,2]
        Ixy, Ixz, Iyz = self.quadrotor.J[0,1], self.quadrotor.J[0,2], self.quadrotor.J[1,2]
        # theta = np.array([self.quadrotor.mass,Ixx,Ixy,Ixz,Iyy,Iyz,Izz])
      #  print(Ixx)
        mass = np.copy(self.quadrotor.mass)
        theta = np.array([mass,Ixx,Ixy,Ixz,Iyy,Iyz,Izz])
        theta_hat = theta.copy()
       # print("DEKA theta: ", theta)
        # get tasks goals
        x_nom, u_nom = self.quadrotor.get_hover_goals(theta.copy())
        Anp, Bnp = self.quadrotor.get_linearized_dynamics(x_nom, u_nom, theta_hat)
        Q, R = self.quadrotor_controller.get_QR_bryson()
        self.quadrotor_controller.update_linearized_dynamics(Anp, Bnp, Q, R)

        # randomly perturb initial state
        x0 = np.copy(x_nom)
        x0[0:3] += np.array([0.1, -0.15, -0.1])  # disturbed initial position
        x0[3:7] = self.quadrotor.rptoq(np.array([0.3, 0.4, 0.2]))  # disturbed initial attitude #TODO: move math methods to utilities class
       # print("Perturbed Intitial State: ")
      #  print(x0)

        # get initial control and state
        u_curr = self.quadrotor_controller.compute(x0, x_nom, u_nom)
        x_curr = np.copy(x0)

        x_all = []
        u_all = []
        theta_all = []
        theta_hat_all = []

        changing_steps = [100]#np.random.choice(range(20,180),size=2, replace=False)
 
        deka = DEKA(num_params=7, x0=theta.reshape(-1,1), damping=0.7, regularization=1e-15, smoothing_factor=0.2, tol_max=1e-5, tol_min=0.01)

        n = 10

        A_tot = np.zeros((6*n,7))
        b_tot = np.zeros((6*n,1))

        # simulate the dynamics with the LQR controller
        for i in range(NSIM):
            # noising the parameters
            # if i % 10 == 0:
            #     theta += np.random.normal(0, process_noise_std, size=theta.shape)

            # Change system parameters at specific step
            if i in changing_steps:
                update_scale = np.random.uniform(2, 3)
                theta = np.array([mass*update_scale,Ixx,Ixy,Ixz,Iyy,Iyz,Izz])

            # update goals
            x_nom, u_nom = self.quadrotor.get_hover_goals(theta_hat.copy())
            Anp, Bnp = self.quadrotor.get_linearized_dynamics(x_nom, u_nom, theta_hat.copy())
            self.quadrotor_controller.update_linearized_dynamics(Anp, Bnp, Q, R)

            # compute controls
            u_curr = self.quadrotor_controller.compute(x_curr, x_nom, u_nom)                    # at t=k

            # step
            if i > 0 and i % 15 == 0:
                process_noise_std = 0.1 * np.abs(theta[0])  # Process noise
                theta[0] += np.random.normal(0, process_noise_std)
            x_curr = self.quadrotor.quad_dynamics_rk4(x_curr, u_curr, theta)    # at t=k+1
            # formulate measurement model
            A = self.quadrotor.get_data_matrix(x_curr, self.quadrotor.quad_dynamics(x_curr, u_curr, theta))
            b = self.quadrotor.get_force_vector(x_curr, u_curr, theta)
            measurement_noise_std = 0.05 * np.abs(b)
            b += np.random.normal(0, measurement_noise_std, size = b.shape)

            # print(A@(theta.reshape(-1,1)))
            # print(b)
            # import pdb; pdb.set_trace()

            A_tot = np.roll(A_tot, shift=3, axis=0)
            A_tot[:6] = A
            b_tot = np.roll(b_tot, shift=3, axis=0)
            b_tot[:6] = b

            if i % 5 == 0 and i > 0:
                # print("step", i)
                theta_hat_prev = theta_hat
                theta_hat = deka.iterate(A_tot,b_tot, x_0=theta_hat_prev.reshape(-1,1),num_iterations=int(9/6*(n**2)))[0].reshape(-1,)
                
                if (np.linalg.norm(theta_hat - theta_hat_prev) / np.linalg.norm(theta_hat)) > 0.2:
                    theta_hat = 0.8 * theta_hat + 0.2 * theta_hat_prev
                    # print(f"smoothed at step {i} with value {(np.linalg.norm(theta_hat - theta_hat_prev) / np.linalg.norm(theta_hat))}: ")
                # print("theta: ", theta, "\n"
                #     "theta_hat", theta_hat, "\n",
                #     )
            # print(A@(theta.reshape(-1,1)))
            # print(b)




            x_curr = x_curr.reshape(x_curr.shape[0]).tolist() #TODO: x one step in front of controls
            u_curr = u_curr.reshape(u_curr.shape[0]).tolist()

            x_all.append(x_curr)
            u_all.append(u_curr)
            theta_all.append(theta.copy())
            theta_hat_all.append(theta_hat.copy())

        return x_all, u_all, theta_all, theta_hat_all

    def main(): 
        np.set_printoptions(suppress=False, precision=10)

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
                visualize_trajectory_inertia(x_history, u_history, theta_history, title) #TODO: different state space for different robots!
            else:                                       # with parameter estimation 
                x_history, u_history, theta_history, theta_hat_history = getattr(ParamEst, method_name)()
                title = f"Online Parameter Estimation for {args.robot} {args.task} with {args.method}"
                visualize_trajectory_with_est(x_history, u_history, theta_history, theta_hat_history, title)
                # theta_hat_deka_array = np.array(theta_hat_history)
                # theta_deka_array = np.array(theta_history)
                # visualize_all_parameter(theta_deka_array, theta_hat_deka_array)

        else: 
            print(f"Error: Method {method_name} not found. Use --help to see available option.")


if __name__ == "__main__":
    OnlineParamEst.main()