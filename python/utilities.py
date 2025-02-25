import matplotlib.pyplot as plt
import autograd.numpy as np

def visualize_trajectory(x_all, u_all, theta_all, title):
    # Set up the figure and axis for plotting
    fig, ax = plt.subplots(3, 1)

    # Plot the trajectory
    x_all = np.array(x_all)
    nsteps = len(x_all)
    steps = np.arange(nsteps)
    ax[0].plot(steps, x_all[:, 0], label="x", linewidth=2)
    ax[0].plot(steps, x_all[:, 1], label="y", linewidth=2)
    ax[0].plot(steps, x_all[:, 2], label="z", linewidth=2)
    ax[0].legend()
    ax[0].title.set_text("Position")

    theta_all = np.array(theta_all)
    nsteps = len(theta_all)
    steps = np.arange(nsteps)
    ax[1].plot(steps, theta_all[:, 0], label="mass", linewidth=2)
    ax[1].plot(steps, theta_all[:, 1], label="gravity", linewidth=2)
    ax[1].legend()
    ax[1].title.set_text("Parameters: Mass & Gravity")
   
    u_all = np.array(u_all)
    nsteps = len(u_all)
    steps = np.arange(nsteps)
    ax[2].plot(steps, u_all[:, 0], label="u1", linewidth=2)
    ax[2].plot(steps, u_all[:, 1], label="u2", linewidth=2)
    ax[2].plot(steps, u_all[:, 2], label="u3", linewidth=2)
    ax[2].plot(steps, u_all[:, 3], label="u4", linewidth=2)
    ax[2].legend()
    ax[2].title.set_text("Controls")

    plt.suptitle(title)
    plt.show()

def visualize_trajectory_with_est(x_all, u_all, theta_all, theta_hat_all, title):
    # Set up the figure and axis for plotting
    fig, ax = plt.subplots(4, 1)

    # Plot the trajectory
    x_all = np.array(x_all)
    nsteps = len(x_all)
    steps = np.arange(nsteps)
    ax[0].plot(steps, x_all[:, 0], label="x", linewidth=1)
    ax[0].plot(steps, x_all[:, 1], label="y", linewidth=1)
    ax[0].plot(steps, x_all[:, 2], label="z", linewidth=1)
    ax[0].legend()
    ax[0].title.set_text("Position")

    theta_all = np.array(theta_all)
    theta_hat_all = np.array(theta_hat_all)
    nsteps = len(theta_all)
    steps = np.arange(nsteps)
    ax[1].plot(steps, theta_all[:,0], label="m", linewidth=1)
    ax[1].plot(steps, theta_hat_all[:,0], label="m_hat", linewidth=1)
    ax[1].legend()
    ax[1].title.set_text("Param: Mass")

    ax[2].plot(steps, theta_all[:,1], label="g", linewidth=1)
    ax[2].plot(steps, theta_hat_all[:,1], label="g_hat", linewidth=1)
    ax[2].legend()
    ax[2].title.set_text("Param: Gravity")

    u_all = np.array(u_all)
    nsteps = len(u_all)
    steps = np.arange(nsteps)
    ax[3].plot(steps, u_all[:, 0], label="u1", linewidth=1)
    ax[3].plot(steps, u_all[:, 1], label="u2", linewidth=1)
    ax[3].plot(steps, u_all[:, 2], label="u3", linewidth=1)
    ax[3].plot(steps, u_all[:, 3], label="u4", linewidth=1)
    ax[3].legend()
    ax[3].title.set_text("Controls")

    plt.tight_layout()
    plt.suptitle(title)
    plt.show()

def plot_trajectory_with_est(x_all, u_all, theta_all, theta_hat_all, lsq_solns, title="Quadrotor Simulation"):
        """
        Plots quadrotor state trajectory, control inputs, and estimated parameters.
        """
        time = range(len(x_all))  # 0,1,2,... up to NSIM-1
        
        # Extract position
        pos_x = [state[0] for state in x_all]
        pos_y = [state[1] for state in x_all]
        pos_z = [state[2] for state in x_all]

        # Extract controls
        u1 = [ctrl[0] for ctrl in u_all]
        u2 = [ctrl[1] for ctrl in u_all]
        u3 = [ctrl[2] for ctrl in u_all]
        u4 = [ctrl[3] for ctrl in u_all]

        # Extract ground truth parameters
        mass, Jx, Jy, Jz = [], [], [], []
        for theta in theta_all:
            mass.append(theta[0].item())
            Jx.append(theta[1].item())
            Jy.append(theta[2].item())
            Jz.append(theta[3].item())

        # Extract estimated parameters
        mass_est, Jx_est, Jy_est, Jz_est = [], [], [], []
        for theta_hat in theta_hat_all:
            mass_est.append(theta_hat[0].item())
            Jx_est.append(theta_hat[1].item())
            Jy_est.append(theta_hat[2].item())
            Jz_est.append(theta_hat[3].item())

        # Extract LSQ solutions
        mass_lsq, Jx_lsq, Jy_lsq, Jz_lsq = [], [], [], []
        for lsq in lsq_solns:
            mass_lsq.append(lsq[0].item())
            Jx_lsq.append(lsq[1].item())
            Jy_lsq.append(lsq[2].item())
            Jz_lsq.append(lsq[3].item())

        # Plot everything
        plt.figure(figsize=(10, 14))

        # Position
        plt.subplot(6,1,1)
        plt.plot(time, pos_x, label='x-position')
        plt.plot(time, pos_y, label='y-position')
        plt.plot(time, pos_z, label='z-position')
        plt.title(title)
        plt.ylabel("Position (m)")
        plt.legend()
        plt.grid(True)

        # Control Inputs
        plt.subplot(6,1,2)
        plt.plot(time, u1, label='u1')
        plt.plot(time, u2, label='u2')
        plt.plot(time, u3, label='u3')
        plt.plot(time, u4, label='u4')
        plt.ylabel("Control Inputs")
        plt.legend()
        plt.grid(True)

        # Jx
        plt.subplot(6,1,3)
        plt.plot(time, Jx, label='Jx_true')
        plt.plot(time, Jx_est, label='Jx_est')
        plt.plot(time, Jx_lsq, label='Jx_lsq')
        plt.ylabel("Theta")
        plt.legend()
        plt.grid(True)

        # Jy
        plt.subplot(6,1,4)
        plt.plot(time, Jy, label='Jy_true')
        plt.plot(time, Jy_est, label='Jy_est')
        plt.plot(time, Jy_lsq, label='Jy_lsq')
        plt.ylabel("Theta")
        plt.legend()
        plt.grid(True)

        # Jz
        plt.subplot(6,1,5)
        plt.plot(time, Jz, label='Jz_true')
        plt.plot(time, Jz_est, label='Jz_est')
        plt.plot(time, Jz_lsq, label='Jz_lsq')
        plt.ylabel("Theta")
        plt.legend()
        plt.grid(True)

        # Mass
        plt.subplot(6,1,6)
        plt.plot(time, mass, label='mass_true')
        plt.plot(time, mass_est, label='mass_est')
        plt.plot(time, mass_lsq, label='mass_lsq')
        plt.xlabel("Time Step")
        plt.ylabel("Theta")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

def plot_trajectory(x_all, u_all, theta_all, title="Quadrotor Simulation"):
        """
        Plots quadrotor state trajectory, control inputs, and estimated parameters.
        """
        time = range(len(x_all))  # 0,1,2,... up to NSIM-1
        
        # Extract position
        pos_x = [state[0] for state in x_all]
        pos_y = [state[1] for state in x_all]
        pos_z = [state[2] for state in x_all]

        # Extract controls
        u1 = [ctrl[0] for ctrl in u_all]
        u2 = [ctrl[1] for ctrl in u_all]
        u3 = [ctrl[2] for ctrl in u_all]
        u4 = [ctrl[3] for ctrl in u_all]

        # Extract ground truth parameters
        mass, Jx, Jy, Jz = [], [], [], []
        for theta in theta_all:
            mass.append(theta[0].item())
            Jx.append(theta[1].item())
            Jy.append(theta[2].item())
            Jz.append(theta[3].item())    

        # Plot everything
        plt.figure(figsize=(10, 14))

        # Position
        plt.subplot(6,1,1)
        plt.plot(time, pos_x, label='x-position')
        plt.plot(time, pos_y, label='y-position')
        plt.plot(time, pos_z, label='z-position')
        plt.title(title)
        plt.ylabel("Position (m)")
        plt.legend()
        plt.grid(True)

        # Control Inputs
        plt.subplot(6,1,2)
        plt.plot(time, u1, label='u1')
        plt.plot(time, u2, label='u2')
        plt.plot(time, u3, label='u3')
        plt.plot(time, u4, label='u4')
        plt.ylabel("Control Inputs")
        plt.legend()
        plt.grid(True)

        # Jx
        plt.subplot(6,1,3)
        plt.plot(time, Jx, label='Jx_true')
        plt.ylabel("Theta")
        plt.legend()
        plt.grid(True)

        # Jy
        plt.subplot(6,1,4)
        plt.plot(time, Jy, label='Jy_true')
        plt.ylabel("Theta")
        plt.legend()
        plt.grid(True)

        # Jz
        plt.subplot(6,1,5)
        plt.plot(time, Jz, label='Jz_true')
        plt.ylabel("Theta")
        plt.legend()
        plt.grid(True)

        # Mass
        plt.subplot(6,1,6)
        plt.plot(time, mass, label='mass_true')
        plt.xlabel("Time Step")
        plt.ylabel("Theta")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

# @staticmethod
# def visualize_trajectory(x_all, u_all, title):
#     # Set up the figure and axis for plotting
#     fig, ax = plt.subplots(2, 1)

#     # Plot the trajectory
#     x_all = np.array(x_all)
#     nsteps = len(x_all)
#     steps = np.arange(nsteps)
#     ax[0].plot(steps, x_all[:, 0], label="x", linewidth=2)
#     ax[0].plot(steps, x_all[:, 1], label="y", linewidth=2)
#     ax[0].plot(steps, x_all[:, 2], label="z", linewidth=2)
#     ax[0].legend()
#     ax[0].title.set_text("Position")

#     u_all = np.array(u_all)
#     nsteps = len(u_all)
#     steps = np.arange(nsteps)
#     ax[1].plot(steps, u_all[:, 0], label="u1", linewidth=2)
#     ax[1].plot(steps, u_all[:, 1], label="u2", linewidth=2)
#     ax[1].plot(steps, u_all[:, 2], label="u3", linewidth=2)
#     ax[1].plot(steps, u_all[:, 3], label="u4", linewidth=2)
#     # ax[1].legend()
#     ax[1].title.set_text("Controls")

#     plt.suptitle(title)
#     plt.show()