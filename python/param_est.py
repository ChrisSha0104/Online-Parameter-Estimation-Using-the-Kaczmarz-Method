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
from utilities import visualize_trajectory, visualize_trajectory_with_theta, visualize_trajectory_hover

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
    
    def simulate_quadrotor_hover_with_controller_RK(self, x0, x_nom, u_nom, NSIM = 200):
        x_all = []
        u_all = []
        theta_all = []
        theta_hat_all = []
        x_curr = np.copy(x0)

        # initialize
        mass = self.quadrotor.mass
        g0 = self.quadrotor.g
        g_hat = self.quadrotor.g

        change_params = False
        u_curr = self.controller(x0, x_nom, u_nom)

        rk = RK()

        # simulate the dynamics with the LQR controller
        for i in range(NSIM):
            # change mass
            if i == 50:
                change_params = True
                # mass_new = 2*mass
                g_new = 2*g0

            # MPC controller
            u_nom = (mass*g_hat/self.quadrotor.kt/4)*np.ones(4)
            u_prev = u_curr
            u_curr = self.controller(x_curr, x_nom, u_nom)

            # print("step: ", i, "\n", "controls: ", u_curr, "\n", "position: ", x_curr[:3])

            if change_params:
                x_prev = x_curr
                x_curr = self.quadrotor.quad_dynamics_rk4(x_curr, u_curr, mass, g_new)
                theta_all.append(g_new)
                x0 = g_hat
            else:
                x_prev = x_curr
                x_curr = self.quadrotor.quad_dynamics_rk4(x_curr, u_curr, mass, g0)
                theta_all.append(g0)
                x0 = g_hat

            # taylor exp
            df_dg = jacobian(self.quadrotor.quad_dynamics_rk4, 3)

            df_dg_at_g0 = df_dg(x_prev, u_prev, mass, g0)
            f_at_g0 = self.quadrotor.quad_dynamics_rk4(x_prev, u_prev, mass, g0)
            x_curr_hat = f_at_g0 + df_dg_at_g0 * (g_hat - g0)
            b = (x_curr - f_at_g0 + df_dg_at_g0 * g0)

            if i == 0 or i == 60:
                b_all = b.reshape(13,1)
                df_dg_all = df_dg_at_g0.reshape(13,1)
            if i >= 1:
                # import pdb
                # pdb.set_trace()
                b_all = np.vstack((b_all, b.reshape(13,1)))
                df_dg_all = np.vstack((df_dg_all, df_dg_at_g0.reshape(13,1)))
                g_hat = rk.iterate(df_dg_all, b_all, x0, 1000).item()

            # print("A: ", df_dg_at_g0)
            # print("b: ", b)
            if change_params:
                print("residual: ", np.abs(df_dg_all * g_new - b_all).sum())
            else:
                print("residual: ", np.abs(df_dg_all * g0 - b_all).sum())

            print("A:" , df_dg_at_g0)
            print("b: ", b)

            if change_params:
                print("step: ", i, "guess: ", g_hat, "gt: ", g_new)
            else:
                print("step: ", i, "guess: ", g_hat, "gt: ", g0)

            x_curr = x_curr.reshape(x_curr.shape[0]).tolist()
            u_curr = u_curr.reshape(u_curr.shape[0]).tolist()

            x_all.append(x_curr)
            u_all.append(u_curr)
            theta_hat_all.append(g_hat)
        return x_all, u_all, theta_hat_all, theta_all
    
    def simulate_with_controller_RLS(self, x0, x_nom, u_nom, NSIM = 300):
        # TODO: change variables, add comments
        x_all = []
        u_all = []
        theta_all = []
        theta_hat_all = []
        x_curr = np.copy(x0)

        # initialize
        mass = self.quadrotor.mass
        g0 = self.quadrotor.g
        g_hat = self.quadrotor.g

        change_params = False
        u_curr = np.array([1.2837, 1.0766, 0.1413, 0.152])

        rls = RLS(num_params=1)

        # simulate the dynamics with the LQR controller
        for i in range(NSIM):
            # change mass
            if i == 50:
                change_params = True
                # mass_new = 2*mass
                g_new = 2*g0

            # MPC controller
            u_nom = (mass*g_hat/self.quadrotor.kt/4)*np.ones(4)

            u_prev = u_curr
            u_curr = self.controller(x_curr, x_nom, u_nom)

            # print("step: ", i, "\n", "controls: ", u_curr, "\n", "position: ", x_curr[:3])

            if change_params:
                x_prev = x_curr
                x_curr = self.quadrotor.quad_dynamics_rk4(x_curr, u_curr, mass, g_new)
                theta_all.append(g_new)
            else:
                x_prev = x_curr
                x_curr = self.quadrotor.quad_dynamics_rk4(x_curr, u_curr, mass, g0)
                theta_all.append(g0)

            df_dg = jacobian(self.quadrotor.quad_dynamics_rk4)
            df_dg_at_g0 = df_dg(x_prev, u_prev, mass, g0)

            f_at_g0 = self.quadrotor.quad_dynamics_rk4(x_prev, u_prev, mass, g0)

            x_curr_hat = f_at_g0 + df_dg_at_g0 * (g_hat - g0)

            b = x_curr - f_at_g0 + df_dg_at_g0 * g0

            print("A: ", df_dg_at_g0)
            print("b: ", b)
            print("g_hat", g_hat)

            # print("Ax-b: ", df_dg_at_g0 * g_hat - b)

            g_hat = rls.update(df_dg_at_g0, b).item()

            # if change_params:
            #     print("step: ", i, "guess: ", g_hat, "gt: ", g_new)
            # else:
            #     print("step: ", i, "guess: ", g_hat, "gt: ", g0)

            x_curr = x_curr.reshape(x_curr.shape[0]).tolist()
            u_curr = u_curr.reshape(u_curr.shape[0]).tolist()

            x_all.append(x_curr)
            u_all.append(u_curr)
            theta_hat_all.append(g_hat)
        return x_all, u_all, theta_hat_all, theta_all

    def main(): 
        parser = argparse.ArgumentParser(description="Example script with command-line arguments")
        parser.add_argument("--robot", type=str, required=True, help="choose from quadrotor, cartpole, doublependulum")
        parser.add_argument("--task", type=str, required=True, help="choose from hover, tracking, control") 
        parser.add_argument("--method", type=str, required=True, help="choose from Naive_MPC, KM, RLS, EKF")  
        args = parser.parse_args()

        if args.method == "Naive_MPC":
            ParamEst = OnlineParamEst()
            update_status = "late_update"
            x_history, u_history, theta_history = ParamEst.simulate_quadrotor_hover_with_MPC(update_status)
            title = "doubling parameter at t=50 with update status: " + update_status
            visualize_trajectory_hover(x_history, u_history, theta_history, title)
        if args.method == "RLS":
            ParamEst = OnlineParamEst()
            x_history, u_history, theta_history = ParamEst.simulate_quadrotor_hover_with_RLS()
            title = "doubling parameter at t=50 with update status: " + update_status
            visualize_trajectory_hover(x_history, u_history, theta_history, title)

if __name__ == "__main__":
    OnlineParamEst.main()