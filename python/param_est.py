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
from LQR_controller import LQRController
from estimation_methods import RK, RLS, EKF
from utilities import visualize_trajectory, visualize_trajectory_hover_with_est, visualize_trajectory_hover

import argparse

class OnlineParamEst:
    def __init__(self):
        # Quadrotor Tasks:
        self.quadrotor = Quadrotor()
        self.quadrotor_controller = LQRController(self.quadrotor.delta_x_quat)

    def simulate_quadrotor_hover_with_MPC(self, update_status: str, NSIM: int =200):
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
                theta = np.array([self.quadrotor.mass + 2,self.quadrotor.g * 2])

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
    
    def simulate_quadrotor_hover_with_RLS(self, NSIM: int =100): #TODO: add RLS parameters here!
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
                theta = np.array([self.quadrotor.mass + 2,self.quadrotor.g * 2])
            # MPC controller
            x_nom, u_nom = self.quadrotor.get_hover_goals(theta_hat[0], theta_hat[1], self.quadrotor.kt)
            Anp, Bnp = self.quadrotor.get_linearized_dynamics(x_nom, u_nom, theta_hat[0], theta_hat[1])
            self.quadrotor_controller.update_linearized_dynamics(Anp, Bnp, Q, R)
            u_prev = u_curr
            u_curr = self.quadrotor_controller.compute(x_curr, x_nom, u_nom)

            x_prev = x_curr
            x_curr = self.quadrotor.quad_dynamics_rk4(x_curr, u_curr, theta[0], theta[1])
            print("step: ", i, "\n", "controls: ", u_curr, "\n", "position: ", x_curr[:3], "\n", "theta: ", theta, "\n", "theta_hat: ", theta_hat)

            # estimate parameters
            theta_hat_prev = theta_hat
            df_dtheta_at_theta_prev = np.column_stack((df_dm(x_prev, u_prev, theta_hat_prev[0], theta_hat_prev[1]),  # (num_states, num_params)
                                                       df_dg(x_prev, u_prev, theta_hat_prev[0], theta_hat_prev[1])))
            f_at_theta_prev = self.quadrotor.quad_dynamics_rk4(x_prev, u_prev, theta_hat_prev[0], theta_hat_prev[1]) # (num_states, 1)
            b = x_curr - f_at_theta_prev + df_dtheta_at_theta_prev @ theta_hat_prev # (num_states, 1)

            # import pdb; pdb.set_trace()
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

    def main(): 
        parser = argparse.ArgumentParser(description="Example script with command-line arguments")
        parser.add_argument("--robot", type=str, required=False, help="choose from quadrotor, cartpole, doublependulum")
        parser.add_argument("--task", type=str, required=False, help="choose from hover, tracking, control") 
        parser.add_argument("--method", type=str, required=True, help="choose from Naive_MPC, KM, RLS, EKF")  
        args = parser.parse_args()

        if args.method == "Naive_MPC":
            ParamEst = OnlineParamEst()
            update_status = "late_update"
            x_history, u_history, theta_history = ParamEst.simulate_quadrotor_hover_with_MPC(update_status)
            title = "g*=2, m+=2 at t=50 Naive MPC with update status: " + update_status
            visualize_trajectory_hover(x_history, u_history, theta_history, title)
        if args.method == "RLS":
            ParamEst = OnlineParamEst()
            x_history, u_history, theta_history, theta_hat_history = ParamEst.simulate_quadrotor_hover_with_RLS()
            title = "g*=2, m+=2 at t=50 with RLS"
            visualize_trajectory_hover_with_est(x_history, u_history, theta_history, theta_hat_history, title)

if __name__ == "__main__":
    OnlineParamEst.main()