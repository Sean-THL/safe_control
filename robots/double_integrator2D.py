import numpy as np
import casadi as ca

"""
Created on July 15th, 2024
@author: Taekyung Kim

@description: 
Double Integrator model for CBF-QP and MPC-CBF (casadi) with separated position and attitude states
"""


def angle_normalize(x):
    if isinstance(x, (np.ndarray, float, int)):
        # NumPy implementation
        return (((x + np.pi) % (2 * np.pi)) - np.pi)
    elif isinstance(x, (ca.SX, ca.MX, ca.DM)):
        # CasADi implementation
        return ca.fmod(x + ca.pi, 2 * ca.pi) - ca.pi
    else:
        raise TypeError(f"Unsupported input type: {type(x)}")


class DoubleIntegrator2D:

    def __init__(self, dt, robot_spec):
        '''
            X: [x, y, vx, vy]
            theta: yaw angle
            U: [ax, ay]
            U_attitude: [yaw_rate]
            cbf: h(x) = ||x-x_obs||^2 - beta*d_min^2
            relative degree: 2
        '''
        self.dt = dt
        self.robot_spec = robot_spec
        
        self.robot_spec.setdefault('model', 'DoubleIntegrator2D')

        self.robot_spec.setdefault('a_max', 1.0)
        self.robot_spec.setdefault('v_max', 1.0)
        self.robot_spec.setdefault('ax_max', self.robot_spec['a_max'])
        self.robot_spec.setdefault('ay_max', self.robot_spec['a_max'])
        self.robot_spec.setdefault('w_max', 0.5)

    def f(self, X, casadi=False):
        if casadi:
            return ca.vertcat(
                X[2, 0],
                X[3, 0],
                0,
                0
            )
        else:
            return np.array([X[2, 0],
                             X[3, 0],
                             0,
                             0]).reshape(-1, 1)

    def df_dx(self, X):
        return np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])

    def g(self, X, casadi=False):
        if casadi:
            return ca.DM([
                [0, 0],
                [0, 0],
                [1, 0],
                [0, 1]
            ])
        else:
            return np.array([[0, 0], [0, 0], [1, 0], [0, 1]])

    def step(self, X, U):
        """One-step discrete-time dynamics.

        Supports both NumPy (simulation) and CasADi SX/MX/DM (optimization).
        """
        is_casadi = isinstance(X, (ca.SX, ca.MX, ca.DM)) or isinstance(U, (ca.SX, ca.MX, ca.DM))

        # Use the appropriate backend for f and g
        fX = self.f(X, casadi=is_casadi)
        gX = self.g(X, casadi=is_casadi)
        Xn = X + (fX + gX @ U) * self.dt

        # Enforce velocity limits if specified
        v_max = self.robot_spec.get('v_max')
        if v_max is not None:
            vx, vy = Xn[2, 0], Xn[3, 0]
            if is_casadi:
                v_mag = ca.sqrt(vx * vx + vy * vy)
                eps = 1e-9
                scale = ca.if_else(v_mag > v_max, v_max / (v_mag + eps), 1.0)
                Xn[2, 0] = vx * scale
                Xn[3, 0] = vy * scale
            else:
                v_mag = np.sqrt(vx**2 + vy**2)
                if v_mag > v_max:
                    scale = v_max / v_mag
                    Xn[2, 0] *= scale
                    Xn[3, 0] *= scale

        return Xn

    def step_rotate(self, theta, U_attitude):
        theta = angle_normalize(theta + U_attitude[0, 0] * self.dt)
        return theta

    def nominal_input(self, X, G, d_min=0.05, k_v=1.0, k_a=1.0):
        '''
        nominal input for CBF-QP (position control)
        '''
        G = np.copy(G.reshape(-1, 1))  # goal state
        v_max = self.robot_spec['v_max']  # Maximum velocity (x+y)
        a_max = self.robot_spec['a_max']  # Maximum acceleration

        pos_errors = G[0:2, 0] - X[0:2, 0]
        pos_errors = np.sign(pos_errors) * \
            np.maximum(np.abs(pos_errors) - d_min, 0.0)

        # Compute desired velocities for x and y
        v_des = k_v * pos_errors
        v_mag = np.linalg.norm(v_des)
        if v_mag > v_max:
            v_des = v_des * v_max / v_mag

        # Compute accelerations
        current_v = X[2:4, 0]
        a = k_a * (v_des - current_v)
        a_mag = np.linalg.norm(a)
        if a_mag > a_max:
            a = a * a_max / a_mag

        return a.reshape(-1, 1)

    def nominal_attitude_input(self, theta, theta_des, k_theta=1.0):
        '''
        nominal input for attitude control
        '''
        error_theta = angle_normalize(theta_des - theta)
        yaw_rate = k_theta * error_theta
        return np.array([yaw_rate]).reshape(-1, 1)

    def stop(self, X, k_a=1.0):
        # Set desired velocity to zero
        vx_des, vy_des = 0.0, 0.0
        ax = k_a * (vx_des - X[2, 0])
        ay = k_a * (vy_des - X[3, 0])
        return np.array([ax, ay]).reshape(-1, 1)

    def has_stopped(self, X, tol=0.05):
        return np.linalg.norm(X[2:4, 0]) < tol

    def rotate_to(self, theta, theta_des, k_omega=2.0):
        error_theta = angle_normalize(theta_des - theta)
        yaw_rate = k_omega * error_theta
        yaw_rate = np.clip(yaw_rate, -self.robot_spec['w_max'], self.robot_spec['w_max'])
        return np.array([yaw_rate]).reshape(-1, 1)

    def agent_barrier(self, X, obs, robot_radius, beta=1.01):
        '''Continuous Time High Order CBF'''
        h = 0
        h_dot = 0
        dh_dot_dx = 0
        if obs[-1] == 0:
            obsX = obs[0:2].reshape(-1, 1)
            d_min = obs[2] + robot_radius  # obs radius + robot radius

            h = np.linalg.norm(X[0:2] - obsX[0:2])**2 - beta*d_min**2
            # Lgh is zero => relative degree is 2, f(x)[0:2] actually equals to X[2:4]
            h_dot = 2 * (X[0:2] - obsX[0:2]).T @ (self.f(X)[0:2])

            # these two options are the same
            # df_dx = self.df_dx(X)
            # dh_dot_dx = np.append( ( 2 * self.f(X)[0:2] ).T, np.array([[0,0]]), axis = 1 ) + 2 * ( X[0:2] - obsX[0:2] ).T @ df_dx[0:2,:]
            dh_dot_dx = np.append(2 * X[2:4].T, 2 * (X[0:2] - obsX[0:2]).T, axis=1)
        elif obs[-1] == 1:
            ox = obs[0]
            oy = obs[1]
            a = obs[2]
            b = obs[3]
            e = obs[4]
            theta = obs[5]

            pox_prime = np.cos(theta)*(X[0, 0]-ox) + np.sin(theta)*(X[1, 0]-oy)
            poy_prime = -np.sin(theta)*(X[0, 0]-ox) + np.cos(theta)*(X[1, 0]-oy)

            h = (pox_prime/(a + robot_radius))**(e) + (poy_prime/(b + robot_radius))**(e) - 1

            dh_dx = np.array([
                e*(pox_prime**(e-1))*(np.cos(theta)/(a + robot_radius)**e) + e*(poy_prime**(e-1))*(-np.sin(theta)/(b + robot_radius)**e),
                e*(pox_prime**(e-1))*(np.sin(theta)/(a + robot_radius)**e) + e*(poy_prime**(e-1))*(np.cos(theta)/(b + robot_radius)**e),
                0,
                0
            ]).reshape(1, -1)

            h_dot = dh_dx @ (self.f(X))

            dh_dot_dx = np.array([
                ((e * (e - 1) / ((a + robot_radius)**e)) * (pox_prime**(e - 2)) * np.cos(theta)**2 + (e * (e - 1) / ((b + robot_radius)**e)) * (poy_prime**(e - 2)) * np.sin(theta)**2) * X[2, 0] 
                + (((e * (e - 1) / ((a + robot_radius)**e)) * (pox_prime**(e - 2)) - (e * (e - 1) / ((b + robot_radius)**e)) * (poy_prime**(e - 2))) * np.cos(theta) * np.sin(theta)) * X[3, 0],
            
            
                (((e * (e - 1) / ((a + robot_radius)**e)) * (pox_prime**(e - 2)) - (e * (e - 1) / ((b + robot_radius)**e)) * (poy_prime**(e - 2))) * np.cos(theta) * np.sin(theta)) * X[2, 0] 
                + ((e * (e - 1) / ((a + robot_radius)**e)) * (pox_prime**(e - 2)) * np.sin(theta)**2 + (e * (e - 1) / ((b + robot_radius)**e)) * (poy_prime**(e - 2)) * np.cos(theta)**2) * X[3, 0],
            
                e * (pox_prime**(e - 1))*(np.cos(theta) / (a + robot_radius)**e) + e * (poy_prime**(e - 1)) * (-np.sin(theta) / (b + robot_radius)**e),
            
                e * (pox_prime**(e - 1))*(np.sin(theta) / (a + robot_radius)**e) + e * (poy_prime**(e - 1)) * (np.cos(theta) / (b + robot_radius)**e)
            ]).reshape(1, -1)


        return h, h_dot, dh_dot_dx

    def agent_barrier_dt(self, x_k, u_k, obs, robot_radius, beta=1.01):
        '''Discrete Time High Order CBF'''
        # Dynamics equations for the next states
        x_k1 = self.step(x_k, u_k)
        x_k2 = self.step(x_k1, u_k)

        def _h_circle(x, obs, robot_radius, beta):
            """CBF for circular obstacle: ||p-p_obs||^2 - beta*d_min^2"""
            x_obs = obs[0]
            y_obs = obs[1]
            r_obs = obs[2]
            d_min = robot_radius + r_obs
            return (x[0, 0] - x_obs) ** 2 + (x[1, 0] - y_obs) ** 2 - beta * d_min ** 2

        def _h_superellipsoid(x, obs, robot_radius, beta):
            """CBF for rotated super-ellipsoid obstacle."""
            ox = obs[0]
            oy = obs[1]
            a = obs[2]
            b = obs[3]
            e = obs[4]
            theta = obs[5]

            # Use CasADi trig if x is symbolic; otherwise use NumPy
            if isinstance(x, (ca.SX, ca.MX, ca.DM)):
                c = ca.cos(theta)
                s = ca.sin(theta)
            else:
                c = np.cos(theta)
                s = np.sin(theta)

            pox_prime = c * (x[0, 0] - ox) + s * (x[1, 0] - oy)
            poy_prime = -s * (x[0, 0] - ox) + c * (x[1, 0] - oy)

            return (pox_prime / (a + robot_radius)) ** e + (poy_prime / (b + robot_radius)) ** e - 1

        def h(x, obs, robot_radius, beta=1.01):
            # obs[6] encodes obstacle type: 0 for circle, 1 for super-ellipsoid
            is_circle = obs[6] < 0.5
            if isinstance(x, (ca.SX, ca.MX, ca.DM)):
                return ca.if_else(is_circle,
                                  _h_circle(x, obs, robot_radius, beta),
                                  _h_superellipsoid(x, obs, robot_radius, beta))
            else:
                return _h_circle(x, obs, robot_radius, beta) if is_circle else _h_superellipsoid(x, obs, robot_radius, beta)

        h_k2 = h(x_k2, obs, robot_radius, beta)
        h_k1 = h(x_k1, obs, robot_radius, beta)
        h_k = h(x_k, obs, robot_radius, beta)

        d_h = h_k1 - h_k
        dd_h = h_k2 - 2 * h_k1 + h_k
        # hocbf_2nd_order = h_ddot + (gamma1 + gamma2) * h_dot + (gamma1 * gamma2) * h_k

        return h_k, d_h, dd_h