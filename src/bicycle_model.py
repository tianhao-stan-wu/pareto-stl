# kinematic_bicycle.py
"""
State x = [px, py, theta, v]
Control u = [a, beta]

Continuous dynamics:
    px_dot   = v*cos(theta) - v*sin(theta)*beta
    py_dot   = v*sin(theta) + v*cos(theta)*beta
    theta_dot= (v / lr) * beta
    v_dot    = a

Discrete (forward Euler):
    x_next = x + dt * xdot(x, u)

Features:
- numpy-based step and rollout
- linearize(x,u) returns discrete-time A, B (analytic Jacobians)
- optional CasADi function generator (if casadi is installed)
"""

from typing import Tuple, Optional
import numpy as np

try:
    import casadi as ca  # type: ignore
    _HAS_CASADI = True
except Exception:
    _HAS_CASADI = False


class KinematicBicycle:
    def __init__(self, lr: float = 1.5, dt: float = 0.05,
                 v_min: Optional[float] = None, v_max: Optional[float] = None):
        """
        Parameters
        ----------
        lr : float
            distance from center of mass to rear axle (m)
        dt : float
            discrete timestep (s)
        v_min, v_max : optional floats
            optional velocity clipping bounds used in step() to keep numeric stability
        """
        self.lr = float(lr)
        self.dt = float(dt)
        self.v_min = None if v_min is None else float(v_min)
        self.v_max = None if v_max is None else float(v_max)

    # -----------------------
    # Core continuous dynamics
    # -----------------------
    def _xdot(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Continuous-time derivative x_dot = f(x,u) under small-slip approx.
        x shape: (4,)  -> [px, py, theta, v]
        u shape: (2,)  -> [a, beta]
        returns xdot shape (4,)
        """
        px, py, theta, v = x
        a, beta = u

        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        px_dot = v * cos_t - v * sin_t * beta
        py_dot = v * sin_t + v * cos_t * beta
        theta_dot = (v / self.lr) * beta
        v_dot = a

        return np.array([px_dot, py_dot, theta_dot, v_dot], dtype=float)

    # -----------------------
    # Discrete step / rollout
    # -----------------------
    def step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        One-step forward Euler integration: x_next = x + dt * xdot

        Parameters
        ----------
        x : array-like (4,)
        u : array-like (2,)

        Returns
        -------
        x_next : ndarray (4,)
        """
        x = np.asarray(x, dtype=float).reshape(4,)
        u = np.asarray(u, dtype=float).reshape(2,)
        xdot = self._xdot(x, u)
        x_next = x + self.dt * xdot

        # optional velocity clipping for numerical stability
        if self.v_min is not None or self.v_max is not None:
            v = x_next[3]
            if self.v_min is not None:
                v = max(v, self.v_min)
            if self.v_max is not None:
                v = min(v, self.v_max)
            x_next[3] = v

        return x_next

    def rollout(self, x0: np.ndarray, U: np.ndarray) -> np.ndarray:
        """
        Propagate a sequence of controls.

        Parameters
        ----------
        x0 : array-like (4,)
            initial state
        U : array-like (T, 2)
            control sequence

        Returns
        -------
        X : ndarray (T+1, 4)
            states along the trajectory (includes x0 as first row)
        """
        x = np.asarray(x0, dtype=float).reshape(4,)
        U = np.asarray(U, dtype=float)
        T = U.shape[0]
        X = np.zeros((T + 1, 4), dtype=float)
        X[0] = x
        for k in range(T):
            x = self.step(x, U[k])
            X[k + 1] = x
        return X

    # -----------------------
    # Linearization (discrete A, B)
    # -----------------------
    def linearize(self, x: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Linearize the discretized dynamics around (x, u) using analytic Jacobians.

        Returns discrete-time Jacobians A, B such that:
            x_next ≈ A @ x + B @ u + c  (c can be computed if needed)
        where A = I + dt * df/dx, B = dt * df/du for the forward Euler integrator.

        Parameters
        ----------
        x : array-like (4,)
        u : array-like (2,)

        Returns
        -------
        A : ndarray (4,4)
        B : ndarray (4,2)
        """
        x = np.asarray(x, dtype=float).reshape(4,)
        u = np.asarray(u, dtype=float).reshape(2,)
        _, _, theta, v = x
        a, beta = u

        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        # continuous-time Jacobians df/dx and df/du for xdot = f(x,u)
        # xdot = [v*cos - v*sin*beta,
        #         v*sin + v*cos*beta,
        #         v/lr * beta,
        #         a]
        df_dx = np.zeros((4, 4), dtype=float)
        # d/dtheta
        df_dx[0, 2] = -v * sin_t - v * cos_t * beta  # d(px_dot)/dtheta
        df_dx[1, 2] = v * cos_t - v * sin_t * beta   # d(py_dot)/dtheta
        df_dx[2, 2] = 0.0                             # d(theta_dot)/dtheta = 0
        # d/dv
        df_dx[0, 3] = cos_t - sin_t * beta           # d(px_dot)/dv
        df_dx[1, 3] = sin_t + cos_t * beta           # d(py_dot)/dv
        df_dx[2, 3] = beta / self.lr                  # d(theta_dot)/dv
        # df_dx[3, :] are zeros (v_dot does not depend on state)

        # df/du
        df_du = np.zeros((4, 2), dtype=float)
        # d(v_dot)/da = 1
        df_du[3, 0] = 1.0
        # d(px_dot)/dbeta = -v * sin(theta)
        df_du[0, 1] = -v * sin_t
        # d(py_dot)/dbeta = v * cos(theta)
        df_du[1, 1] = v * cos_t
        # d(theta_dot)/dbeta = v / lr
        df_du[2, 1] = v / self.lr

        # discrete-time A,B via forward Euler
        A = np.eye(4) + self.dt * df_dx
        B = self.dt * df_du
        return A, B

    # -----------------------
    # CasADi function (optional)
    # -----------------------
    def casadi_function(self):
        """
        Return a CasADi function f_casadi(x, u) -> x_next if CasADi is available.

        Raises
        ------
        ImportError if casadi is not installed.
        """
        if not _HAS_CASADI:
            raise ImportError("casadi is not installed or failed to import")

        x_sym = ca.SX.sym("x", 4)
        u_sym = ca.SX.sym("u", 2)
        px, py, theta, v = x_sym[0], x_sym[1], x_sym[2], x_sym[3]
        a_sym, beta_sym = u_sym[0], u_sym[1]

        cos_t = ca.cos(theta)
        sin_t = ca.sin(theta)

        px_dot = v * cos_t - v * sin_t * beta_sym
        py_dot = v * sin_t + v * cos_t * beta_sym
        theta_dot = (v / self.lr) * beta_sym
        v_dot = a_sym

        xdot = ca.vertcat(px_dot, py_dot, theta_dot, v_dot)
        x_next = x_sym + self.dt * xdot
        f = ca.Function("bicycle_step", [x_sym, u_sym], [x_next], ["x", "u"], ["x_next"])
        return f

    # -----------------------
    # Utilities
    # -----------------------
    @staticmethod
    def wrap_theta(theta: float) -> float:
        """Wrap angle to [-pi, pi)."""
        return (theta + np.pi) % (2 * np.pi) - np.pi


