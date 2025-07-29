import numpy as np
from scipy.spatial.transform import Rotation as R

def apply_noise(x, pos_std=3e-4, rot_std_deg=1.0, vel_std=1e-4, angvel_std_deg=0.5):
    """
    Apply physically meaningful noise to a 13D state vector.
    
    Args:
        x (np.ndarray): State vector of shape (13,)
        pos_std (float): Position noise std dev in meters (default: 1 mm)
        rot_std_deg (float): Orientation noise in degrees (default: 2 deg)
        vel_std (float): Linear velocity noise in m/s (default: 1 cm/s)
        angvel_std_deg (float): Angular velocity noise in deg/s (default: 1 deg/s)

    Returns:
        x_noisy (np.ndarray): Noisy state vector
    """
    x_noisy = x.copy()

    # 1. Position noise (Euclidean)
    x_noisy[0:3] += np.random.normal(0, pos_std, 3)

    # 2. Rotation noise (SO(3)) via axis-angle
    rot_std_rad = np.deg2rad(rot_std_deg)
    delta_rotvec = np.random.normal(0, rot_std_rad, 3)  # axis-angle vector
    delta_rot = R.from_rotvec(delta_rotvec)
    
    q_wxyz = x_noisy[3:7]  # quaternion in w, x, y, z order
    q_xyzw = np.roll(q_wxyz, -1)  # convert to x, y, z, w order for scipy
    q_orig = R.from_quat(q_xyzw)  # [x, y, z, w]
    q_noisy = (delta_rot * q_orig).as_quat()  # apply left multiplication
    
    q_out = q_noisy / np.linalg.norm(q_noisy)  # normalize for safety
    x_noisy[3:7] = np.roll(q_out, 1)  # convert back to w, x, y, z order

    # 3. Linear velocity noise (Euclidean)
    x_noisy[7:10] += np.random.normal(0, vel_std, 3)

    # 4. Angular velocity noise (Euclidean)
    angvel_std_rad = np.deg2rad(angvel_std_deg)
    x_noisy[10:13] += np.random.normal(0, angvel_std_rad, 3)

    return x_noisy

class AWGNNoise:
    """
    Additive White Gaussian Noise (AWGN)
    Uncorrelated zero-mean Gaussian noise.
    """

    def __init__(self, sigma: float):
        """
        Args:
            sigma: Standard deviation of the noise.
        """
        self.sigma = sigma

    def reset(self):
        """No internal state to reset for white noise."""
        pass

    def sample(self, shape=()):
        """
        Generate AWGN samples.
        
        Args:
            shape: Tuple indicating the output shape (default scalar).
        
        Returns:
            np.ndarray or float: Noise sample(s).
        """
        return np.random.normal(0.0, self.sigma, size=shape)


class OUNoise:
    """
    Ornstein–Uhlenbeck (OU) process for temporally correlated noise.
    x_{t+1} = x_t + theta*(mu - x_t)*dt + sigma*sqrt(dt)*N(0,1)
    """

    def __init__(self, theta: float, mu: float, sigma: float, dt: float = 0.01):
        """
        Args:
            theta: Rate of mean reversion.
            mu: Long-term mean.
            sigma: Noise intensity.
            dt: Time step.
        """
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.state = None
        self.reset()

    def reset(self):
        """Reset the process to its mean."""
        self.state = np.array(self.mu, dtype=float)

    def sample(self):
        """
        Advance one time step and return OU noise.
        
        Returns:
            float: New noise value.
        """
        dx = self.theta * (self.mu - self.state) * self.dt
        dx += self.sigma * np.sqrt(self.dt) * np.random.randn()
        self.state += dx
        return self.state


class RandomWalkNoise:
    """
    Random walk noise: b_{t+1} = b_t + sigma * sqrt(dt) * N(0,1)
    Models slowly drifting biases.
    """

    def __init__(self, sigma: float, dt: float = 0.01):
        """
        Args:
            sigma: Standard deviation of increments.
            dt: Time step.
        """
        self.sigma = sigma
        self.dt = dt
        self.state = None
        self.reset()

    def reset(self):
        """Reset the bias to zero."""
        self.state = 0.0

    def sample(self):
        """
        Advance one time step and return random-walk bias.
        
        Returns:
            float: New bias value.
        """
        step = self.sigma * np.sqrt(self.dt) * np.random.randn()
        self.state += step
        return self.state
