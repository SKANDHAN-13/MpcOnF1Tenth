#!/usr/bin/env python3
import math
import os
import time

from dataclasses import dataclass, field
import csv

import numpy as np
import rclpy

from geometry_msgs.msg import Point
from ackermann_msgs.msg import AckermannDrive, AckermannDriveStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from builtin_interfaces.msg import Duration
from visualization_msgs.msg import Marker


########################################################################################################
@dataclass
class mpc_config:
    NXK: int = 4  # length of kinematic state vector: z = [x, y, v, yaw]
    NU: int = 2   # length of input vector: u = [accel, steering_angle]
    TK: int = 4   # finite time horizon length

    Rk: list = field(
        default_factory=lambda: np.diag([
            3.0,   
            5.0,   
        ])
    )

    Rdk: list = field(
        default_factory=lambda: np.diag([
            3.0,   
            5.0,   
        ])
    )

    Qk: list = field(
        default_factory=lambda: np.diag([
            8.0,    
            8.0,    
            20.0,   
            20.0,   
        ])
    )

    Qfk: list = field(
        default_factory=lambda: np.diag([
            8.0,    
            8.0,    
            20.0,   
            20.0,   
        ])
    )

    N_IND_SEARCH: int = 20
    DTK: float  = 0.1    # time step [s]
    dlk: float  = 0.03   # dist step [m]
    LENGTH: float = 0.58
    WIDTH: float  = 0.31
    WB: float     = 0.33
    MIN_STEER: float  = -0.4189
    MAX_STEER: float  =  0.4189
    MAX_DSTEER: float = np.deg2rad(180.0)
    MAX_SPEED: float  = 5.0
    MIN_SPEED: float  = 0.0
    MAX_ACCEL: float  = 3.0


#######################################################################################################
@dataclass
class State:
    x: float = 0.0
    y: float = 0.0
    delta: float = 0.0
    v: float = 0.0
    yaw: float = 0.0
    yawrate: float = 0.0
    beta: float = 0.0


##################################################################################################
def calc_speed_profile(cx, cy, max_speed: float = 5.0, max_accel: float = 3.0, ds: float = 0.03):
    """
    Curvature-based speed profile with forward-backward acceleration passes.
    Without the passes, the reference drops instantaneously at corner entry.
    The MPC ignores an impossible step-change and the car enters too fast.

    Args:
        cx, cy     : path x,y arrays (uniformly spaced at ds)
        max_speed  : straight-line speed cap [m/s]
        max_accel  : peak accel AND decel [m/s2], must match config.MAX_ACCEL
        ds         : waypoint spacing [m], must match config.dlk
    """
    min_speed = max_speed * 0.6

    x_ext = np.concatenate((cx[-2:], cx, cx[:2]))
    y_ext = np.concatenate((cy[-2:], cy, cy[:2]))
    dx  = np.gradient(x_ext, edge_order=2)
    dy  = np.gradient(y_ext, edge_order=2)
    d2x = np.gradient(dx,    edge_order=2)
    d2y = np.gradient(dy,    edge_order=2)
    denom = (dx**2 + dy**2) ** 1
    denom = np.where(denom < 1e-9, 1e-9, denom)
    kappa = np.abs((dx * d2y - d2x * dy) / denom)[2:-2]

    kappa_max = np.percentile(kappa, 95)
    if kappa_max < 1e-6:
        return np.full(len(cx), max_speed)
    t  = np.clip(kappa / kappa_max, 0.0, 1.0)
    sp = max_speed - t * (max_speed - min_speed)
    sp = np.clip(sp, min_speed, max_speed)

    # Backward pass - braking ramp into slow corners
    for i in range(len(sp) - 2, -1, -1):
        v_can_reach = math.sqrt(sp[(i + 1) % len(sp)] ** 2 + 2.0 * max_accel * ds)
        sp[i] = min(sp[i], v_can_reach)

    # Forward pass - acceleration ramp out of corners
    for i in range(1, len(sp)):
        v_can_reach = math.sqrt(sp[i - 1] ** 2 + 2.0 * max_accel * ds)
        sp[i] = min(sp[i], v_can_reach)

    return sp


######################################################################################################
def wrap_angle(a):
    """Wrap angle argument into [-pi, pi]."""
    return np.arctan2(np.sin(a), np.cos(a))


#######################################################################################################
class MPC(Node):
    def __init__(self):
        super().__init__('mpc_node')

        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.pose_sub  = self.create_subscription(
            Odometry, '/ego_racecar/odom', self.pose_callback, 10)

        self.waypoints = self.load_waypoints("waypoints.csv")
        self._ref_v    = None

        self.target_ind             = 0
        self.target_ind_initialized = False
        self.prev_odom_yaw          = None
        self.yaw_offset             = 0.0

        self.config      = mpc_config()
        self.odelta_v    = None
        self.oa          = None
        self._prev_xk    = None
        self._prev_uk    = None
        self._solve_ok   = False

        # CSV logging
        self._session_start = time.time()
        self._csv_path   = os.path.join(os.getcwd(), 'lmpc_performance.csv')
        self._csv_file   = open(self._csv_path, 'w', newline='')
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow([
            'time_s', 'label', 'x', 'y', 'v', 'yaw',
            'cte', 'ref_x', 'ref_y', 'ref_v', 'yaw_diff'
        ])
        self._csv_file.flush()

        # Visualization
        self.ref_pub   = self.create_publisher(Marker, '/ego_racecar/mpc_ref_traj', 10)
        self.ref_timer = self.create_timer(0.5, self.visualize_ref_path)
        self.traj_pub  = self.create_publisher(Marker, '/ego_racecar/driven_traj', 10)

        self.traj_marker = Marker()
        self.traj_marker.header.frame_id = "ego_racecar/odom"
        self.traj_marker.ns    = "driven_path"
        self.traj_marker.id    = 100
        self.traj_marker.type  = Marker.LINE_STRIP
        self.traj_marker.action = Marker.ADD
        self.traj_marker.scale.x = 0.05
        self.traj_marker.color.r = 0.0
        self.traj_marker.color.g = 1.0
        self.traj_marker.color.b = 0.0
        self.traj_marker.color.a = 1.0
        self.traj_marker.pose.orientation.w = 1.0
        self.traj_marker.lifetime = Duration(sec=0)
        self.traj_marker.points   = []
        self.last_traj_x = None
        self.last_traj_y = None

        self._lmpc_prob_init()
        self.get_logger().info(f'LMPC (OSQP, equality-only) ready. CSV: {self._csv_path}')

    ###############################################################################
    # Visualization
    ###############################################################################

    def visualize_ref_path(self):
        marker = Marker()
        marker.header.frame_id = "ego_racecar/odom"
        marker.header.stamp    = self.get_clock().now().to_msg()
        marker.ns    = "ref_path"
        marker.id    = 0
        marker.type  = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.05
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        marker.pose.orientation.w = 1.0
        marker.lifetime = Duration(sec=0)
        marker.points   = []
        for i in range(len(self.waypoints)):
            p = Point()
            p.x = float(self.waypoints[i, 0])
            p.y = float(self.waypoints[i, 1])
            p.z = 0.05
            marker.points.append(p)
        self.ref_pub.publish(marker)

    ###############################################################################
    # Waypoint loading
    ###############################################################################

    def load_waypoints(self, path):
        data = []
        with open(path, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                x, y, yaw = map(float, row)
                data.append([x, y, yaw])

        wp = np.array(data)
        wp[:, 2] = np.unwrap(wp[:, 2])

        x0, y0, _ = wp[0]
        x1, y1, _ = wp[-1]
        gap = np.hypot(x1 - x0, y1 - y0)

        if gap > 0.05:
            N_interp = max(int(gap / 0.03), 2)
            t = np.linspace(0.0, 1.0, N_interp + 2)[1:-1]
            bridge = np.vstack([
                x1 + t * (x0 - x1),
                y1 + t * (y0 - y1),
                np.zeros(len(t))
            ]).T
            wp = np.vstack([wp, bridge])

        diffs = np.diff(wp[:, :2], axis=0)
        seg_lengths = np.hypot(diffs[:, 0], diffs[:, 1])
        s = np.concatenate([[0.0], np.cumsum(seg_lengths)])
        total_length = s[-1]

        s_new  = np.arange(0.0, total_length, 0.03)
        wp_x   = np.interp(s_new, s, wp[:, 0])
        wp_y   = np.interp(s_new, s, wp[:, 1])
        wp_yaw = np.interp(s_new, s, wp[:, 2])

        return np.vstack([wp_x, wp_y, wp_yaw]).T

    ###############################################################################
    # Pose callback
    ###############################################################################

    def pose_callback(self, pose_msg):
        x = pose_msg.pose.pose.position.x
        y = pose_msg.pose.pose.position.y
        q = pose_msg.pose.pose.orientation
        t3  = 2.0 * (q.w * q.z + q.x * q.y)
        t4  = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw_raw = math.atan2(t3, t4)

        if self.prev_odom_yaw is None:
            self.prev_odom_yaw = yaw_raw
        dyaw = yaw_raw - self.prev_odom_yaw
        if dyaw >  np.pi: self.yaw_offset -= 2 * np.pi
        elif dyaw < -np.pi: self.yaw_offset += 2 * np.pi
        yaw_z = yaw_raw + self.yaw_offset
        self.prev_odom_yaw = yaw_raw

        vx = pose_msg.twist.twist.linear.x
        vy = pose_msg.twist.twist.linear.y
        v  = math.sqrt(vx**2 + vy**2)
        vehicle_state = State(x=x, y=y, v=v, yaw=yaw_z)

        # Update driven trajectory trail
        if self.last_traj_x is None:
            self.last_traj_x, self.last_traj_y = x, y
        if math.hypot(x - self.last_traj_x, y - self.last_traj_y) > 0.05:
            p = Point(); p.x = float(x); p.y = float(y); p.z = 0.02
            self.traj_marker.points.append(p)
            self.last_traj_x, self.last_traj_y = x, y
            self.traj_marker.header.stamp = self.get_clock().now().to_msg()
            self.traj_pub.publish(self.traj_marker)

        ref_x   = self.waypoints[:, 0]
        ref_y   = self.waypoints[:, 1]
        ref_yaw = self.waypoints[:, 2]
        if self._ref_v is None:
            self._ref_v = calc_speed_profile(ref_x, ref_y,
                                             max_speed=self.config.MAX_SPEED,
                                             max_accel=self.config.MAX_ACCEL,
                                             ds=self.config.dlk)
        ref_v = self._ref_v

        ref_path = self.calc_ref_trajectory(vehicle_state, ref_x, ref_y, ref_yaw, ref_v)
        x0 = [vehicle_state.x, vehicle_state.y, vehicle_state.v, vehicle_state.yaw]

        self.oa, self.odelta_v, ox, oy, oyaw = self._lmpc_control(ref_path, x0)

        self.log_cte(vehicle_state,
                     ref_x[self.target_ind], ref_y[self.target_ind],
                     ref_v[self.target_ind], ref_yaw[self.target_ind])

        if self.oa is None or self.odelta_v is None:
            drive = AckermannDriveStamped()
            drive.drive.steering_angle = 0.0
            drive.drive.speed          = vehicle_state.v
            try: self.drive_pub.publish(drive)
            except Exception: pass
            return

        steer_output = float(self.odelta_v[0])
        if self._solve_ok:
            raw_speed    = vehicle_state.v + float(self.oa[0]) * self.config.DTK
            floor        = max(self.config.MIN_SPEED,
                               float(ref_v[self.target_ind]) - self.config.MAX_ACCEL * self.config.DTK)
            speed_output = float(np.clip(raw_speed, floor, self.config.MAX_SPEED))
        else:
            speed_output = float(np.clip(ref_v[self.target_ind],
                                         self.config.MIN_SPEED, self.config.MAX_SPEED))

        drive = AckermannDriveStamped()
        drive.drive.steering_angle = steer_output
        drive.drive.speed          = speed_output
        self.visualize_ref_path()
        try: self.drive_pub.publish(drive)
        except Exception: pass

    ###############################################################################
    # Reference trajectory
    ###############################################################################

    def calc_ref_trajectory(self, state, cx, cy, cyaw, sp):
        ref_traj = np.zeros((self.config.NXK, self.config.TK + 1))
        ncourse  = len(cx)

        if not self.target_ind_initialized:
            dists = (cx - state.x)**2 + (cy - state.y)**2
            self.target_ind = int(np.argmin(dists))
            self.target_ind_initialized = True

        search_indices = [(self.target_ind + i) % ncourse
                          for i in range(self.config.N_IND_SEARCH)]
        local_cx = np.array([cx[i] for i in search_indices])
        local_cy = np.array([cy[i] for i in search_indices])
        dists    = (local_cx - state.x)**2 + (local_cy - state.y)**2
        local_best = np.argmin(dists)
        if local_best > 0:
            self.target_ind = search_indices[local_best]

        ind = self.target_ind
        ref_traj[0, 0] = cx[ind]
        ref_traj[1, 0] = cy[ind]
        ref_traj[2, 0] = sp[ind]
        ref_traj[3, 0] = cyaw[ind]

        travel   = max(abs(state.v), self.config.MAX_SPEED * 0.5) * self.config.DTK
        dind     = travel / self.config.dlk
        ind_list = (int(ind) + np.insert(
            np.cumsum(np.repeat(dind, self.config.TK)), 0, 0).astype(int)) % ncourse

        ref_traj[0, :] = cx[ind_list]
        ref_traj[1, :] = cy[ind_list]
        ref_traj[2, :] = sp[ind_list]
        for i, idx in enumerate(ind_list):
            ref_traj[3, i] = cyaw[idx]
        return ref_traj

    ###############################################################################

    def update_state(self, state, a, delta):
        delta = np.clip(delta, -self.config.MAX_STEER, self.config.MAX_STEER)
        state.x   += state.v * math.cos(state.yaw) * self.config.DTK
        state.y   += state.v * math.sin(state.yaw) * self.config.DTK
        state.yaw += (state.v / self.config.WB) * math.tan(delta) * self.config.DTK
        state.v    = np.clip(state.v + a * self.config.DTK,
                             self.config.MIN_SPEED, self.config.MAX_SPEED)
        return state
    ###############################################################################
    def _get_model_matrix(self, v, phi, delta):
        """Linearised discrete-time bicycle model matrices (A, B, C)."""
        dt = self.config.DTK
        WB = self.config.WB

        A = np.eye(self.config.NXK)
        A[0, 2] =  dt * math.cos(phi)
        A[0, 3] = -dt * v * math.sin(phi)
        A[1, 2] =  dt * math.sin(phi)
        A[1, 3] =  dt * v * math.cos(phi)
        A[3, 2] =  dt * math.tan(delta) / WB

        B = np.zeros((self.config.NXK, self.config.NU))
        B[2, 0] = dt
        B[3, 1] = dt * v / (WB * math.cos(delta)**2)

        C = np.zeros(self.config.NXK)
        C[0] =  dt * v * math.sin(phi) * phi
        C[1] = -dt * v * math.cos(phi) * phi
        C[3] = -dt * v * delta / (WB * math.cos(delta)**2)
        return A, B, C
    ################################################################################
    def _predict_motion(self, x0, oa, od, xref):
        path_predict = xref * 0.0
        for i in range(len(x0)):
            path_predict[i, 0] = x0[i]
        state = State(x=x0[0], y=x0[1], v=x0[2], yaw=x0[3])
        for i in range(1, self.config.TK + 1):
            state = self.update_state(state, oa[i - 1], od[i - 1])
            path_predict[0, i] = state.x
            path_predict[1, i] = state.y
            path_predict[2, i] = state.v
            path_predict[3, i] = state.yaw
        return path_predict

    ###############################################################################
    # LMPC - OSQP, equality constraints only
    ###############################################################################

    def _lmpc_prob_init(self):
        """
        OSQP solves:  min  (1/2) z' P z + q' z
                      s.t. l <= A z <= u

        Decision vector z (column-major):
            z = [ x_0 ... x_T  |  u_0 ... u_{T-1} ]

        P is fixed (only weight matrices, no reference).
        q is rebuilt each tick (contains -2*Q*x_ref terms).
        A contains only:
            - Initial-state equality:  x_0 = x0
            - Dynamics equalities:     x_{t+1} = A_t x_t + B_t u_t + C_t
        
        """
        from scipy.sparse import csc_matrix as _csc

        NX = self.config.NXK
        NU = self.config.NU
        T  = self.config.TK

        self._l_NX = NX
        self._l_NU = NU
        self._l_T  = T
        self._l_nz = NX * (T + 1) + NU * T

        # Symmetrized weights 
        Qk  = (self.config.Qk  + self.config.Qk.T)  / 2
        Qfk = (self.config.Qfk + self.config.Qfk.T) / 2
        Rk  = (self.config.Rk  + self.config.Rk.T)  / 2
        Rdk = (self.config.Rdk + self.config.Rdk.T)  / 2
        self._l_Qk  = Qk
        self._l_Qfk = Qfk
        self._l_Rk  = Rk
        self._l_Rdk = Rdk

        nz = self._l_nz
        def xi(t): return NX * t
        def ui(t): return NX * (T + 1) + NU * t

        # Build P - tracking + input + rate penalty Hessian
        P = np.zeros((nz, nz))
        for t in range(T):
            s = xi(t)
            P[s:s+NX, s:s+NX] += 2 * Qk
        s = xi(T)
        P[s:s+NX, s:s+NX] += 2 * Qfk
        for t in range(T):
            s = ui(t)
            P[s:s+NU, s:s+NU] += 2 * Rk

        # Rate penalty tridiagonal blocks
        for t in range(T - 1):
            s0, s1 = ui(t), ui(t + 1)
            P[s0:s0+NU, s0:s0+NU] += 2 * Rdk
            P[s1:s1+NU, s1:s1+NU] += 2 * Rdk
            P[s0:s0+NU, s1:s1+NU] -= 2 * Rdk   # upper triangle only

        self._l_P_csc = _csc(np.triu(P))
        self._osqp_y_ws = None
    ####################################################################################
    def _lmpc_prob_solve(self, ref_traj, path_predict, x0):
        """
        Build the equality-only QP for the current tick and solve with OSQP.

        Constraint matrix A:
          r0_init  [NX]   : x_0 = x0                              (equality)
          r0_dyn   [NX*T] : I x_{t+1} - A_t x_t - B_t u_t = C_t  (equality)
          r0_vspd  [T+1]  : MIN_SPEED <= v_t <= MAX_SPEED          (inequality)
          r0_accel [T]    : |a_t| <= MAX_ACCEL                     (inequality)
          r0_steer [T]    : |delta_t| <= MAX_STEER                 (inequality)
          r0_srate [T-1]  : |delta_{t+1}-delta_t| <= MAX_DSTEER*DTK (inequality)

        """
        import osqp as _osqp
        from scipy.sparse import lil_matrix, csc_matrix as _csc

        NX, NU, T = self._l_NX, self._l_NU, self._l_T
        nz        = self._l_nz

        def xi(t): return NX * t
        def ui(t): return NX * (T + 1) + NU * t

        # Yaw continuity
        ref_traj = ref_traj.copy()
        ref_traj[3, :] = x0[3] + wrap_angle(ref_traj[3, :] - x0[3])
        for i in range(1, ref_traj.shape[1]):
            d = ref_traj[3, i] - ref_traj[3, i - 1]
            if d >  np.pi: ref_traj[3, i] -= 2 * np.pi
            if d < -np.pi: ref_traj[3, i] += 2 * np.pi

        # Linear cost q = -2 * Q * x_ref  (reference-dependent, rebuilt each tick)
        q = np.zeros(nz)
        for t in range(T):
            q[xi(t):xi(t)+NX] = -2 * self._l_Qk @ ref_traj[:, t]
        q[xi(T):xi(T)+NX] = -2 * self._l_Qfk @ ref_traj[:, T]

        # Linearise dynamics along predicted trajectory
        A_mats, B_mats, C_vecs = [], [], []
        for t in range(T):
            delta_bar = float(self.odelta_v[t]) if self.odelta_v is not None else 0.0
            A, B, C = self._get_model_matrix(
                float(path_predict[2, t]), float(path_predict[3, t]), delta_bar)
            A_mats.append(A); B_mats.append(B); C_vecs.append(C)

        # Constraint matrix row layout:
        #   r0_init  [NX]   : x_0 = x0
        #   r0_dyn   [NX*T] : I x_{t+1} - A_t x_t - B_t u_t = C_t
        #   r0_vspd  [T+1]  : MIN_SPEED <= v_t <= MAX_SPEED
        #   r0_accel [T]    : |a_t| <= MAX_ACCEL
        #   r0_steer [T]    : |delta_t| <= MAX_STEER
        #   r0_srate [T-1]  : |delta_{t+1} - delta_t| <= MAX_DSTEER * DTK
        r0_init  = 0
        r0_dyn   = r0_init  + NX
        r0_vspd  = r0_dyn   + NX * T
        r0_accel = r0_vspd  + (T + 1)
        r0_steer = r0_accel + T
        r0_srate = r0_steer + T
        n_rows   = r0_srate + (T - 1)

        A_sp = lil_matrix((n_rows, nz))
        l_v  = np.full(n_rows, -np.inf)
        u_v  = np.full(n_rows,  np.inf)

        # Initial state pin
        for j in range(NX):
            A_sp[r0_init + j, xi(0) + j] = 1.0
        l_v[r0_init:r0_init+NX] = x0
        u_v[r0_init:r0_init+NX] = x0

        # Dynamics equalities
        for t in range(T):
            rb = r0_dyn + NX * t
            for i in range(NX):
                A_sp[rb + i, xi(t + 1) + i] = 1.0
                for j in range(NX):
                    A_sp[rb + i, xi(t) + j] = -A_mats[t][i, j]
                for j in range(NU):
                    A_sp[rb + i, ui(t) + j] = -B_mats[t][i, j]
            l_v[rb:rb+NX] = C_vecs[t]
            u_v[rb:rb+NX] = C_vecs[t]

        # Speed bounds
        for t in range(T + 1):
            A_sp[r0_vspd + t, xi(t) + 2] = 1.0
            l_v[r0_vspd + t] = self.config.MIN_SPEED
            u_v[r0_vspd + t] = self.config.MAX_SPEED

        # Accel magnitude bounds
        for t in range(T):
            A_sp[r0_accel + t, ui(t)] = 1.0
            l_v[r0_accel + t] = -self.config.MAX_ACCEL
            u_v[r0_accel + t] =  self.config.MAX_ACCEL

        # Steer magnitude bounds
        for t in range(T):
            A_sp[r0_steer + t, ui(t) + 1] = 1.0
            l_v[r0_steer + t] = -self.config.MAX_STEER
            u_v[r0_steer + t] =  self.config.MAX_STEER

        # Steer rate based bounds
        max_ds = self.config.MAX_DSTEER * self.config.DTK
        for t in range(T - 1):
            A_sp[r0_srate + t, ui(t + 1) + 1] =  1.0
            A_sp[r0_srate + t, ui(t)     + 1] = -1.0
            l_v[r0_srate + t] = -max_ds
            u_v[r0_srate + t] =  max_ds

        A_csc = _csc(A_sp)

        # Warm-start: shift previous solution by one step
        if self._prev_xk is not None and self._prev_uk is not None:
            x_ws         = np.zeros((NX, T + 1))
            u_ws         = np.zeros((NU, T))
            x_ws[:, :-1] = self._prev_xk[:, 1:]
            x_ws[:, -1]  = self._prev_xk[:, -1]
            u_ws[:, :-1] = self._prev_uk[:, 1:]
            u_ws[:, -1]  = self._prev_uk[:, -1]
            z_ws = np.concatenate([x_ws.flatten(order='F'), u_ws.flatten(order='F')])
        else:
            z_ws = np.concatenate([ref_traj.flatten(order='F'), np.zeros(NU * T)])

        oa     = self.oa       if self.oa       is not None else np.zeros(T)
        odelta = self.odelta_v if self.odelta_v is not None else np.zeros(T)
        ox = oy = oyaw = None

        try:
            prob = _osqp.OSQP()
            prob.setup(
                self._l_P_csc, q, A_csc, l_v, u_v,
                warm_starting     = False,
                eps_abs           = 1e-3,
                eps_rel           = 1e-3,
                max_iter          = 8000,
                polish            = True,
                verbose           = False,
                adaptive_rho      = True,
                check_termination = 10,
            )
            if self._osqp_y_ws is not None and len(self._osqp_y_ws) == n_rows:
                prob.warm_start(x=z_ws, y=self._osqp_y_ws)
            else:
                prob.warm_start(x=z_ws)

            res = prob.solve()

            if res.x is not None and not np.any(np.isnan(res.x)):
                xk_sol = res.x[:NX*(T+1)].reshape(NX, T+1, order='F')
                uk_sol = res.x[NX*(T+1):].reshape(NU, T,   order='F')
                ox     = xk_sol[0, :]
                oy     = xk_sol[1, :]
                oyaw   = xk_sol[3, :]
                oa     = uk_sol[0, :]
                odelta = uk_sol[1, :]
                self._prev_xk   = xk_sol.copy()
                self._prev_uk   = uk_sol.copy()
                self._osqp_y_ws = res.y.copy() if res.y is not None else None
                self._solve_ok  = True
                print(f"LMPC : {res.info.status}")
            else:
                self._solve_ok  = False
                self._osqp_y_ws = None
                self._prev_xk   = None
                self._prev_uk   = None
                print(f"LMPC : no valid iterate - {res.info.status}")

        except Exception as e:
            self._solve_ok  = False
            self._osqp_y_ws = None
            print(f"LMPC : exception - {e}")

        return oa, odelta, ox, oy, oyaw

    #####################################################################################
    def _lmpc_control(self, ref_path, x0):
        """Iterative-linearisation outer loop: predict -> solve -> return."""
        oa = self.oa       if self.oa       is not None else [0.0] * self.config.TK
        od = self.odelta_v if self.odelta_v is not None else [0.0] * self.config.TK

        x0_pred = list(x0)
        if abs(x0_pred[2]) < 0.5:
            x0_pred[2] = float(ref_path[2, 1])

        path_predict = self._predict_motion(x0_pred, oa, od, ref_path)
        return self._lmpc_prob_solve(ref_path, path_predict, x0)

    ###############################################################################
    # Logging
    ###############################################################################

    def log_cte(self, state, intended_x, intended_y, intended_v, intended_yaw):
        ind  = self.target_ind
        cx   = self.waypoints[ind, 0]
        cy   = self.waypoints[ind, 1]
        cyaw = self.waypoints[ind, 2]
        cte      = -(state.x - cx) * math.sin(cyaw) + (state.y - cy) * math.cos(cyaw)
        yaw_diff = wrap_angle(state.yaw - float(intended_yaw))
        self._csv_writer.writerow([
            round(time.time() - self._session_start, 4),
            'LMPC',
            round(state.x,   4), round(state.y,   4),
            round(state.v,   4), round(state.yaw, 4),
            round(abs(cte),  4),
            round(float(intended_x), 4), round(float(intended_y), 4),
            round(float(intended_v), 4), round(yaw_diff,          4),
        ])
        self._csv_file.flush()
    #######################################################################################

    def destroy_node(self):
        try:
            self._csv_file.flush()
            self._csv_file.close()
            self.get_logger().info(f'CSV saved: {self._csv_path}')
        except Exception:
            pass
        super().destroy_node()


#######################################################################################
def main(args=None):
    rclpy.init(args=args)
    print("MPC (LMPC / OSQP, equality-only) Initialized")
    mpc_node = MPC()
    try:
        rclpy.spin(mpc_node)
    except (KeyboardInterrupt, SystemExit, Exception):
        pass
    finally:
        mpc_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
    
