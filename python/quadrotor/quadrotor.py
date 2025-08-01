import autograd.numpy as np
from autograd.numpy.linalg import norm, inv
from autograd import jacobian
import math

class QuadrotorDynamics:
    def __init__(self, r_off=None):
        # ————————————————————————————————
        # Ground‑truth parameters (the “real” quadrotor)
        # ————————————————————————————————
        self._mass_true  = 0.035
        self._J_true     = np.array([[1.66e-5, 0.83e-6, 0.72e-6],
                                     [0.83e-6, 1.66e-5, 1.8e-6],
                                     [0.72e-6, 1.8e-6, 2.93e-5]])
        self.g           = 9.81
        self.thrustToTorque = 0.0008
        self.el          = 0.046 / math.sqrt(2)
        self.scale       = 65535
        self.kt          = 2.245365e-6 * self.scale
        self.km          = self.kt * self.thrustToTorque

        # COM offset (first moment) for the true plant
        self._r_off_true = np.zeros(3) if r_off is None else np.array(r_off)

        # True per‑motor hover thrust
        self.hover_thrust_true = (self._mass_true * self.g / self.kt / 4) * np.ones(4)

        # ————————————————————————————————
        # Estimated parameters (what the controller “believes”)
        # ————————————————————————————————
        # initialize to truth
        self.mass_est   = self._mass_true
        self.J_est      = self._J_true.copy()
        self.r_off_est  = self._r_off_true.copy()

        # Dimensions & timing
        self.freq = 50.0
        self.dt   = 1.0 / self.freq
        self.nx   = 12
        self.nu   = 4

    # ──────────────────────────────────────────────────────────
    # Setters / getters for true vs. estimated inertial params
    # ──────────────────────────────────────────────────────────

    def set_true_inertial_params(self, theta):
        """Change the *ground‑truth* mass/J/r_off (e.g. attach payload)."""
        m, cx, cy, cz, Ixx, Iyy, Izz, Ixy, Ixz, Iyz = theta
        self._mass_true  = m
        self._r_off_true = np.array([cx, cy, cz]) / m
        self._J_true     = np.array([[Ixx, Ixy, Ixz],
                                     [Ixy, Iyy, Iyz],
                                     [Ixz, Iyz, Izz]])
        # update true hover thrust too:
        self.hover_thrust_true = (m * self.g / self.kt / 4) * np.ones(4) #TODO: check

    def get_true_inertial_params(self):
        c = self._mass_true * self._r_off_true
        I = self._J_true
        return np.array([
            self._mass_true,
            *c,
            I[0,0], I[1,1], I[2,2],
            I[0,1], I[0,2], I[1,2]
        ])

    def set_estimated_inertial_params(self, theta):
        """Update the *estimated* mass/J/r_off (used by controller)."""
        m, cx, cy, cz, Ixx, Iyy, Izz, Ixy, Ixz, Iyz = theta
        self.mass_est  = m
        self.r_off_est = np.array([cx, cy, cz]) / m
        self.J_est     = np.array([[Ixx, Ixy, Ixz],
                                   [Ixy, Iyy, Iyz],
                                   [Ixz, Iyz, Izz]])

    def get_estimated_inertial_params(self):
        c = self.mass_est * self.r_off_est
        I = self.J_est
        return np.array([
            self.mass_est,
            *c,
            I[0,0], I[1,1], I[2,2],
            I[0,1], I[0,2], I[1,2]
        ])

    # ──────────────────────────────────────────────────────────
    # Hover thrust for the controller (uses estimated mass)
    # ──────────────────────────────────────────────────────────

    def get_hover_thrust_est(self):
        """Per‑motor hover thrust computed from the *estimated* mass."""
        return (self.mass_est * self.g / self.kt / 4) * np.ones(4)

    def get_hover_thrust_true(self):
        """Per‑motor hover thrust computed from the *true* mass."""
        return (self._mass_true * self.g / self.kt / 4) * np.ones(4)

    # ──────────────────────────────────────────────────────────
    # Core dynamics, parameterized by a chosen mass/J/r_off
    # ──────────────────────────────────────────────────────────

    def _dynamics_param(self, x, u, mass, J, r_off, wind_vec=None):
        # unpack
        r    = x[0:3]
        q    = x[3:7] / norm(x[3:7])
        v    = x[7:10]
        omg  = x[10:13]
        Qmat = self.qtoQ(q)

        # dr, dq
        dr = v
        dq = 0.5 * self.L(q) @ self.H @ omg

        # dv_base
        F_th = np.array([[0,0,0,0],
                         [0,0,0,0],
                         [self.kt, self.kt, self.kt, self.kt]]) @ u
        dv_base = np.array([0,0,-self.g]) + (1/mass) * Qmat @ F_th
        if wind_vec is not None:
            dv_base += wind_vec

        # COM coupling
        c = mass * r_off
        # rotational torque + COM coupling
        tau = np.array([
            [-self.el*self.kt, -self.el*self.kt,  self.el*self.kt,  self.el*self.kt],
            [-self.el*self.kt,  self.el*self.kt,  self.el*self.kt, -self.el*self.kt],
            [-self.km,           self.km,          -self.km,          self.km]
        ]) @ u

        # print("J:", J)
        # print("omg:", omg)
        # import pdb; pdb.set_trace()  # Debugging breakpoint
        domg = inv(J) @ (
            -self.hat(omg) @ J @ omg
            + tau
            - self.hat(c) @ dv_base
        )

        # translational COM coupling
        dv_c = np.cross(r_off, domg) + np.cross(omg, np.cross(r_off, omg))
        dv   = dv_base + dv_c

        return np.hstack([dr, dq, dv, domg])

    def _rk4(self, dyn, x, u, dt, wind_vec):
        f1 = dyn(x, u, wind_vec)
        f2 = dyn(x + 0.5*dt*f1, u, wind_vec)
        f3 = dyn(x + 0.5*dt*f2, u, wind_vec)
        f4 = dyn(x +    dt*f3, u, wind_vec)
        xn = x + (dt/6.0)*(f1 + 2*f2 + 2*f3 + f4)
        # renormalize quaternion
        qn = xn[3:7] / norm(xn[3:7])
        return np.hstack([xn[0:3], qn, xn[7:13]])

    # ──────────────────────────────────────────────────────────
    # Public plant vs. controller‐model dynamics
    # ──────────────────────────────────────────────────────────

    def dynamics_true(self, x, u, wind_vec=None):
        return self._dynamics_param(x, u,
                                    self._mass_true,
                                    self._J_true,
                                    self._r_off_true,
                                    wind_vec)

    def dynamics_rk4_true(self, x, u, dt=None, wind_vec=None):
        if dt is None: dt = self.dt
        return self._rk4(self.dynamics_true, x, u, dt, wind_vec)

    def dynamics_est(self, x, u, wind_vec=None):
        return self._dynamics_param(x, u,
                                    self.mass_est,
                                    self.J_est,
                                    self.r_off_est,
                                    wind_vec)

    def dynamics_rk4_est(self, x, u, dt=None, wind_vec=None):
        if dt is None: dt = self.dt
        return self._rk4(self.dynamics_est, x, u, dt, wind_vec)

    # ──────────────────────────────────────────────────────────
    # Linearization about (x_ref, u_ref)
    # ──────────────────────────────────────────────────────────

    def get_linearized_true(self, x_ref, u_ref):
        A_j = jacobian(self.dynamics_rk4_true, 0)
        B_j = jacobian(self.dynamics_rk4_true, 1)
        A = A_j(x_ref, u_ref)
        B = B_j(x_ref, u_ref)
        return self.E(x_ref[3:7]).T @ A @ self.E(x_ref[3:7]), \
               self.E(x_ref[3:7]).T @ B

    def get_linearized_est(self, x_ref, u_ref):
        A_j = jacobian(self.dynamics_rk4_est, 0)
        B_j = jacobian(self.dynamics_rk4_est, 1)
        A = A_j(x_ref, u_ref)
        B = B_j(x_ref, u_ref)
        return self.E(x_ref[3:7]).T @ A @ self.E(x_ref[3:7]), \
               self.E(x_ref[3:7]).T @ B

    # def get_data_matrix(self, x_curr, dx):
    #     a_x, a_y, a_z = dx[7:10]
    #     a_p, a_q, a_r = dx[10:13]
    #     w_x, w_y, w_z = x_curr[10:13]

    #     # First moment terms: r_off = [x_c, y_c, z_c] -> m*x_c, m*y_c, m*z_c as variables
    #     A = np.array([
    #         [a_x, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #         [a_y, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #         [a_z, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #         [0, a_p, a_q - w_x*w_z, a_r + w_x*w_y, -w_y*w_z, w_y**2 + w_z**2, w_y*w_z, 0, -a_z, a_y],
    #         [0, w_x*w_z, a_p + w_y*w_z, w_z**2 - w_x**2, a_q, a_r - w_x*w_y, -w_x*w_z, a_z, 0, -a_x],
    #         [0, -w_x*w_y, w_x**2 - w_y**2, a_p - w_y*w_z, w_x*w_y, a_q + w_x*w_z, a_r, -a_y, a_x, 0]
    #     ], dtype=np.float64)
    #     return A
    
    # def get_force_vector(self, x_curr, u_curr):
    #     R = self.qtoQ(x_curr[3:7])
    #     m = self._mass_true
    #     c = self._mass_true * self._r_off_true

    #     F_b = np.array([[0], [0], [np.sum(u_curr) * self.kt]])
    #     F_w = R @ F_b
    #     F_w[2] -= m * self.g

    #     tau_b = np.array([
    #         [self.el * self.kt * (-u_curr[0] - u_curr[1] + u_curr[2] + u_curr[3])],
    #         [self.el * self.kt * (-u_curr[0] + u_curr[1] + u_curr[2] - u_curr[3])],
    #         [self.km * (-u_curr[0] + u_curr[1] - u_curr[2] + u_curr[3])]
    #     ])

    #     tau_b += np.cross(c.reshape(3), F_b.reshape(3)).reshape(3,1)
    #     return np.vstack([F_w, tau_b]).flatten()
    def get_force_vector(self, x_curr: np.ndarray,
                            dx:     np.ndarray,
                            u_curr: np.ndarray) -> np.ndarray:
        """
        Build the 6×1 force/torque vector b = [F; τ] such that A·θ = b
        with θ = [m, m x_c, m y_c, m z_c, Ixx, Iyy, Izz, Ixy, Ixz, Iyz].

        Args:
        x_curr: state (12,) = [r, q, v, ω]
        dx:     state derivative (12,) = [v, q̇, a, α]
        u_curr: motor inputs (4,)

        Returns:
        b: (6,) = [F_world; τ_body]
        """
        # normalize quaternion & build rotation
        q = x_curr[3:7]
        q = q / np.linalg.norm(q)
        R = self.qtoQ(q)            # body → world

        # physical constants
        m   = self._mass_true
        r_c = self._r_off_true      # COM offset in body frame
        c   = m * r_c               # first moment vector

        # 1) THRUST in body & world
        F_b = np.array([0.0, 0.0, np.sum(u_curr) * self.kt])  # total thrust in body-z
        F_w = R @ F_b                                        # rotate to world
        F_w[2] -= m * self.g                                 # subtract weight

        # 2) TRANSLATIONAL COM‐COUPLING
        ω     = x_curr[10:13]        # body‐frame angular vel
        α     = dx[10:13]            # body‐frame angular accel
        # dv_c in body: ω×(ω×r_c) + r_c×α
        dv_c_b = np.cross(ω, np.cross(ω, r_c)) + np.cross(r_c, α)
        dv_c_w = R @ dv_c_b          # into world frame
        F_coup = m * dv_c_w          # mass × coupling accel

        # total force (world)
        b_trans = F_w + F_coup       # = m*(dv_base + dv_c)

        # 3) MOTOR TORQUES (body)
        tau_motors = np.array([
            [-self.el*self.kt, -self.el*self.kt,  self.el*self.kt,  self.el*self.kt],
            [-self.el*self.kt,  self.el*self.kt,  self.el*self.kt, -self.el*self.kt],
            [-self.km,           self.km,          -self.km,          self.km]
        ]) @ u_curr

        # 4) ROTATIONAL COM‐COUPLING TORQUE
        # dv_base in body = F_b/m + Rᵀ [0,0,−g]
        dv_base_b = F_b/m + R.T @ np.array([0.0, 0.0, -self.g])
        tau_coup  = - np.cross(c, dv_base_b)

        b_rot = tau_motors + tau_coup

        # stack into a flat 6-vector
        return np.hstack([b_trans, b_rot])
    
    def get_data_matrix(self, x, dx):
        # --- Translational rows (world) ---
        a_x, a_y, a_z = dx[7:10]     # world accel
        A = np.zeros((6,10))
        A[0,0] = a_x
        A[1,0] = a_y
        A[2,0] = a_z

        # --- Rotational rows (body) ---
        # first extract body-frame accel:
        q = x[3:7] / np.linalg.norm(x[3:7])
        R = self.qtoQ(q)
        a_b = R.T @ np.array([a_x, a_y, a_z])             # drop gravity since it's orthogonal
        omega = x[10:13]
        alpha = dx[10:13]  # body-frame angular accel

        # row for τ_x = Ixx*α_x + Ixy*α_y + Ixz*α_z
        #           + (Izz−Iyy) ω_y ω_z
        #           −m( y_c a_{b,z} − z_c a_{b,y} )
        A[3,1] = 0                    # m·x_c term
        A[3,2] = -a_b[2]              # m·y_c term
        A[3,3] = +a_b[1]              # m·z_c term
        A[3,4] = alpha[0]             # Ixx
        A[3,5] =      0               # Iyy (off-diags only)
        A[3,6] =      0               # Izz
        A[3,7] = alpha[1]             # Ixy
        A[3,8] = alpha[2]             # Ixz
        A[3,9] =      0               # Iyz
        # plus you’d add the (Izz−Iyy)ω_yω_z term into columns 5 & 6
        # and similarly fill rows 4 and 5 for τ_y and τ_z.

        return A
    
    # parameter change events
    def attach_payload(self, m_p, delta_r_p):
        # delta_r_p: vector in body frame where payload attaches
        m_old = self._mass_true
        r_old = self._r_off_true
        J_old = self._J_true

        m_new = m_old + m_p
        # update COM
        r_new = (m_old*r_old + m_p*delta_r_p) / m_new
        # update J via parallel‐axis
        d = delta_r_p
        Jp = m_p * (np.dot(d,d)*np.eye(3) - np.outer(d,d))
        J_new = J_old + Jp

        self._mass_true  = m_new
        self._r_off_true  = r_new
        self._J_true      = J_new

    def aero_added_inertia(self, wind_speed):
        # scale factor k
        k = 1e-6  
        deltaJ = k * wind_speed**2 * np.eye(3)
        return self._J_true + deltaJ

    @staticmethod
    def hat(v: np.ndarray) -> np.ndarray:
        """
        Convert a 3-vector into a 3×3 skew-symmetric matrix so that
        hat(u) @ v == u × v.

        Parameters:
            v (np.ndarray): length-3 array

        Returns:
            np.ndarray: 3×3 skew-symmetric matrix
        """
        return np.array([
            [    0,    -v[2],   v[1]],
            [ v[2],       0,   -v[0]],
            [-v[1],    v[0],      0]
        ])
    @staticmethod
    def L(q):
        """
        Build the 4×4 left-quaternion multiplication matrix such that
        L(q) @ p == q ⊗ p.

        Parameters:
            q (np.ndarray): length-4 unit quaternion [q0, q1, q2, q3]

        Returns:
            np.ndarray: 4×4 Hamilton product matrix
        """
        s = q[0]
        v = q[1:4]
        return np.vstack([
            np.hstack([s, -v]),
            np.hstack([v.reshape(3,1), s*np.eye(3) + QuadrotorDynamics.hat(v)])
        ])

    # Constants
    T = np.diag([1.0, -1, -1, -1])
    H = np.vstack([np.zeros((1,3)), np.eye(3)])

    @classmethod
    def qtoQ(cls, q):
        """
        Compute the 3×3 rotation matrix corresponding to quaternion q.

        Parameters:
            q (np.ndarray): length-4 unit quaternion

        Returns:
            np.ndarray: 3×3 rotation matrix in SO(3)
        """
        return cls.H.T @ cls.T @ cls.L(q) @ cls.T @ cls.L(q) @ cls.H

    @classmethod
    def G(cls, q):
        return cls.L(q) @ cls.H

    @staticmethod
    def rptoq(phi):
        """
        Convert Rodrigues parameters (3-vector) into a unit quaternion:
            q = [1; phi] / sqrt(1 + phiᵀ phi)

        Parameters:
            phi (np.ndarray): length-3 vector

        Returns:
            np.ndarray: length-4 unit quaternion
        """
        return (1./math.sqrt(1 + phi.T @ phi)) * np.hstack([1, phi])

    @staticmethod
    def qtorp(q):
        """
        Extract Rodrigues parameters from a quaternion:
            phi = q_v / q0

        Parameters:
            q (np.ndarray): length-4 unit quaternion

        Returns:
            np.ndarray: length-3 Rodrigues vector
        """
        return q[1:4]/q[0]

    @classmethod
    def E(cls, q):
        """
        Build a block-diagonal embedding for an extended state that
        linearizes the quaternion part via G(q), leaving other states untouched.

        Parameters:
            q (np.ndarray): length-4 unit quaternion

        Returns:
            np.ndarray: (3+4+6)×(3+4+6) selection/linearization matrix
        """
        return np.vstack([
            np.hstack([np.eye(3), np.zeros((3,3)), np.zeros((3,6))]),
            np.hstack([np.zeros((4,3)), cls.G(q), np.zeros((4,6))]),
            np.hstack([np.zeros((6,3)), np.zeros((6,3)), np.eye(6)])
        ])

   