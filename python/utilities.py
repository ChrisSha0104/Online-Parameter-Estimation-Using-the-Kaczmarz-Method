import matplotlib.pyplot as plt
import autograd.numpy as np

@staticmethod
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

@staticmethod
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