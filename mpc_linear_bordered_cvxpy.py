#!/usr/bin/env python3
import math
import os
import time

from dataclasses import dataclass, field
import csv

import cvxpy as cp
import numpy as np
from scipy.optimize import minimize, Bounds, NonlinearConstraint
import rclpy

from geometry_msgs.msg import Point
from ackermann_msgs.msg import AckermannDrive, AckermannDriveStamped
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from scipy.sparse import block_diag
from sensor_msgs.msg import LaserScan
from utils import nearest_point
from builtin_interfaces.msg import Duration
from visualization_msgs.msg import Marker


########################################################################################################
@dataclass
class mpc_config:
    NXK: int = 4  # length of kinematic state vector: z = [x, y, v, yaw]
    NU: int = 2  # length of input vector: u = = [steering speed, acceleration]
    TK: int = 2 # finite time horizon length kinematic (2 × 0.2 s = 0.4 s)
    
    
    Rk: list = field(
    default_factory=lambda: np.diag([
        1.2, 6.0,   
    ])
    )

    Rdk: list = field(
    default_factory=lambda: np.diag([
        2.0,   
        320.0, 
    ])
    )

    Qk: list = field(
    default_factory=lambda: np.diag([
        260.0,  
        260.0,  
        420.0, 
        900.0, 
    ])
    )

    Qfk: list = field(
    default_factory=lambda: np.diag([
        1400.0, 
        1400.0, 
        1400.0, 
        4200.0, 
    ])
    )

    N_IND_SEARCH: int = 20  # Search index number
    DTK: float = 0.2  # time step [s] 
    dlk: float = 0.03  # dist step [m]
    LENGTH: float = 0.58  # Length of the vehicle [m]
    WIDTH: float = 0.31  # Width of the vehicle [m]
    WB: float = 0.33  # Wheelbase [m]
    MIN_STEER: float = -0.4189  # maximum steering angle [rad]
    MAX_STEER: float = 0.4189  # maximum steering angle [rad]
    MAX_DSTEER: float = np.deg2rad(180.0)  # maximum steering speed [rad/s]
    MAX_SPEED: float = 10.0  # maximum speed [m/s]  
    MIN_SPEED: float = 0.0  # minimum backward speed [m/s]
    MAX_ACCEL: float = 3.0  # maximum acceleration [m/s*s]

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
        max_accel  : peak accel AND decel [m/s²], must match config.MAX_ACCEL
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

    # Backward pass - Braking ramp into slow corners
    # If sp[i+1] < sp[i] and the gap exceeds what max_accel can achieve in
    # one ds step, pull sp[i] down so the car arrives at sp[i+1] in time.
    for i in range(len(sp) - 2, -1, -1):
        v_can_reach = math.sqrt(sp[(i + 1) % len(sp)] ** 2 + 2.0 * max_accel * ds)
        sp[i] = min(sp[i], v_can_reach)

    # Forward pass - Acceleration ramp out of corners
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
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10) #Publisher for drive commands
        self.pose_sub  = self.create_subscription(
            Odometry, '/ego_racecar/odom', self.pose_callback, 10) #Subscriber for odometry data

        self.waypoints = self.load_waypoints("waypoints.csv") # Waypoints are loaded from a CSV file
        self._ref_v    = None #Empty first callback

        #Target indicator for calculating reference trajectory forward in time for MPC to optimize for
        self.target_ind             = 0
        self.target_ind_initialized = False

        self.prev_odom_yaw          = None
        self.yaw_offset             = 0.0

        # MPC configuration parameters
        self.config      = mpc_config()
        self.odelta_v    = None
        self.odelta      = None
        self.oa          = None
        self.init_flag   = 0
        self._prev_xk    = None
        self._prev_uk    = None
        self._lmpc_solve_ok = False

        # CSV logging
        self._session_start = time.time()
        self._csv_path      = None
        self._csv_file      = None
        self._csv_writer    = None

        # Reference trajectory marker
        self.ref_pub   = self.create_publisher(Marker, '/ego_racecar/mpc_ref_traj', 10)
        self.ref_id    = 0
        self.ref_timer = self.create_timer(0.5, self.visualize_ref_path)
        self.traj_pub  = self.create_publisher(Marker, '/ego_racecar/driven_traj', 10)

        # Marker for visualizing the driven trajectory
        self.traj_marker = Marker()
        self.traj_marker.header.frame_id = "ego_racecar/odom"
        self.traj_marker.ns              = "driven_path"
        self.traj_marker.id              = 100
        self.traj_marker.type            = Marker.LINE_STRIP
        self.traj_marker.action          = Marker.ADD
        self.traj_marker.scale.x         = 0.05

        #BRIGHT GREEN
        self.traj_marker.color.r         = 0.0
        self.traj_marker.color.g         = 1.0
        self.traj_marker.color.b         = 0.0
        self.traj_marker.color.a         = 1.0
        self.traj_marker.pose.orientation.w = 1.0
        self.traj_marker.lifetime        = Duration(sec=0)
        self.traj_marker.points          = []
        self.last_traj_x = None
        self.last_traj_y = None
        

        # Left wall  = Reference + d_max * normal  (cyan)
        # Right wall = Reference - d_max * normal  (orange)
        self.corridor_pub_left  = self.create_publisher(
            Marker, '/ego_racecar/corridor_left',  10)
        self.corridor_pub_right = self.create_publisher(
            Marker, '/ego_racecar/corridor_right', 10)

        self.USE_NMPC = False

        solver_label     = 'nmpc' if self.USE_NMPC else 'lmpc'
        self._csv_path   = os.path.join(os.getcwd(), f'{solver_label}_cvxpy_performance.csv')
        self._csv_file   = open(self._csv_path, 'w', newline='')
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow([
            'time_s', 'label', 'x', 'y', 'v', 'yaw',
            'cte', 'ref_x', 'ref_y', 'ref_v', 'yaw_diff'
        ])
        self._csv_file.flush()
        self.get_logger().info(
            f'Logging {solver_label.upper()} (CVXPY variant) to: {self._csv_path}')

        if self.USE_NMPC:
            self.nmpc_prob_init()
        else:
            self._linear_mpc_prob_init()

    ###############################################################################
    # Visualization helpers
    ###############################################################################

    def visualize_ref_path(self):
        marker = Marker()
        marker.header.frame_id = "ego_racecar/odom"
        marker.header.stamp    = self.get_clock().now().to_msg()
        marker.ns              = "ref_path"
        marker.id              = 0
        marker.type            = Marker.LINE_STRIP
        marker.action          = Marker.ADD
        marker.scale.x         = 0.05

        #BRIGHT RED
        marker.color.r         = 1.0
        marker.color.g         = 0.0
        marker.color.b         = 0.0
        marker.color.a         = 1.0

        marker.pose.orientation.w = 1.0
        marker.scale.y         = 0.0
        marker.scale.z         = 0.0
        marker.lifetime        = Duration(sec=0)
        marker.points          = []
        for i in range(len(self.waypoints)):
            p   = Point()
            p.x = float(self.waypoints[i, 0])
            p.y = float(self.waypoints[i, 1])
            p.z = 0.05
            marker.points.append(p)
        self.ref_pub.publish(marker)
    ##################################################################################

    def _publish_corridor_markers(self, ref_traj):
        T       = self.config.TK
        d       = getattr(self, '_corridor_d_max', 1)
        normals = np.zeros((2, T + 1))
        origins = np.zeros((2, T + 1))
        for t in range(T + 1):
            if t < T:
                dx = ref_traj[0, t + 1] - ref_traj[0, t]
                dy = ref_traj[1, t + 1] - ref_traj[1, t]
            else:
                dx = ref_traj[0, t] - ref_traj[0, t - 1]
                dy = ref_traj[1, t] - ref_traj[1, t - 1]
            norm = np.hypot(dx, dy)
            normals[:, t] = [-dy / norm, dx / norm] if norm >= 1e-6 else [0.0, 1.0]
            origins[:, t] = ref_traj[:2, t]

        now   = self.get_clock().now().to_msg()
        frame = "ego_racecar/odom"

        def _wall(uid, r, g, b, sign):
            m = Marker()
            m.header.frame_id = frame
            m.header.stamp    = now
            m.ns   = "corridor"
            m.id   = uid
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.scale.x = 0.04
            m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, 1.0
            m.pose.orientation.w = 1.0
            m.lifetime = Duration(sec=2)
            for t in range(T + 1):
                pt   = Point()
                pt.x = float(origins[0, t] + sign * d * normals[0, t])
                pt.y = float(origins[1, t] + sign * d * normals[1, t])
                pt.z = 0.05
                m.points.append(pt)
            return m

        self.corridor_pub_left.publish( _wall(200, 0.0, 1.0, 1.0,  1))
        self.corridor_pub_right.publish(_wall(201, 1.0, 0.5, 0.0, -1))

    ########################################################################
    # Waypoint loading
    ########################################################################

    def load_waypoints(self, path):

        data = []
        with open(path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                x, y, yaw = map(float, row)
                data.append([x, y, yaw])

        wp = np.array(data)
        wp[:, 2] = np.unwrap(wp[:, 2])

        x0, y0, yaw0 = wp[0]
        x1, y1, yaw1 = wp[-1]

        gap = np.hypot(x1 - x0, y1 - y0)

        if gap > 0.05:    # Bridging points having unusually large space/gaps between them
            N_interp = max(int(gap / 0.03), 2) 

            # Interpolation, excluding the endpoints which already exist as wp[-1] and wp[0] 
            t = np.linspace(0.0, 1.0, N_interp + 2)[1:-1]
            x_new   = x1   + t * (x0   - x1)
            y_new   = y1   + t * (y0   - y1)
            yaw_new = np.zeros(len(t))  

            bridge = np.vstack([x_new, y_new, yaw_new]).T
            wp = np.vstack([wp, bridge])

        
        # Compute cumulative arc-length along the path
        diffs = np.diff(wp[:, :2], axis=0)                   
        seg_lengths = np.hypot(diffs[:, 0], diffs[:, 1])     
        s = np.concatenate([[0.0], np.cumsum(seg_lengths)])  
        total_length = s[-1]

        # Uniform sample points at dlk=0.03 m spacing
        dlk = 0.03
        s_new = np.arange(0.0, total_length, dlk)

        # Linear interpolation of x, y, yaw along arc-length
        wp_x   = np.interp(s_new, s, wp[:, 0])
        wp_y   = np.interp(s_new, s, wp[:, 1])
        wp_yaw = np.interp(s_new, s, wp[:, 2])

        wp = np.vstack([wp_x, wp_y, wp_yaw]).T

        return wp

    ########################################################################
    # Pose callback
    ########################################################################

    def pose_callback(self, pose_msg):
        x = pose_msg.pose.pose.position.x
        y = pose_msg.pose.pose.position.y
        q = pose_msg.pose.pose.orientation
        t3  = 2.0 * (q.w * q.z + q.x * q.y)
        t4  = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw_raw = math.atan2(t3, t4)      #Yaw evaluation from quaternions

        if self.prev_odom_yaw is None:
            self.prev_odom_yaw = yaw_raw
        dyaw = yaw_raw - self.prev_odom_yaw #Change in yaw
        if dyaw >  np.pi: self.yaw_offset -= 2 * np.pi
        elif dyaw < -np.pi: self.yaw_offset += 2 * np.pi
        yaw_z = yaw_raw + self.yaw_offset
        self.prev_odom_yaw = yaw_raw  #Prev_yaw updation

        vx = pose_msg.twist.twist.linear.x
        vy = pose_msg.twist.twist.linear.y
        v  = math.sqrt(vx**2 + vy**2)   #Speed evaluation 
        vehicle_state = State(x=x, y=y, v=v, yaw=yaw_z)

        # Stamp trail every 0.05 m
        if self.last_traj_x is None:
            self.last_traj_x = x
            self.last_traj_y = y
        if math.hypot(x - self.last_traj_x, y - self.last_traj_y) > 0.05:
            p = Point(); p.x = float(x); p.y = float(y); p.z = 0.02
            self.traj_marker.points.append(p)
            self.last_traj_x = x
            self.last_traj_y = y
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
        self._publish_corridor_markers(ref_path)

        x0 = [vehicle_state.x, vehicle_state.y, vehicle_state.v, vehicle_state.yaw]

        #Solve MPC (Depending on USE_NMPC flag)
        if self.USE_NMPC:
            self.oa, self.odelta_v, ox, oy, oyaw = self.nmpc_prob_solve(ref_path, x0)
        else:
            self.oa, self.odelta_v, ox, oy, oyaw, _ = self._linear_mpc_control(
                ref_path, x0, self.oa, self.odelta_v)
        #log performance and CTE
        self.log_cte(vehicle_state,
                     ref_x[self.target_ind], ref_y[self.target_ind],
                     ref_v[self.target_ind], ref_yaw[self.target_ind],
                     label='NMPC' if self.USE_NMPC else 'LMPC')

        if self.oa is None:
            drive = AckermannDriveStamped()
            drive.drive.steering_angle = 0.0
            drive.drive.speed          = vehicle_state.v
            try: self.drive_pub.publish(drive)
            except Exception: pass
            return

        steer_output = float(self.odelta_v[0])
        # oa[0] is the first acceleration command, which we apply together with the first steering command. 
        if getattr(self, '_lmpc_solve_ok', False) and self.oa is not None:
            raw_speed = vehicle_state.v + float(self.oa[0]) * self.config.DTK
            floor     = max(self.config.MIN_SPEED,
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

    #########################################################################
    # Reference trajectory
    #########################################################################
    def calc_ref_trajectory(self, state, cx, cy, cyaw, sp):
        ref_traj = np.zeros((self.config.NXK, self.config.TK + 1))
        ncourse  = len(cx)

        if not self.target_ind_initialized:
            dists = (cx - state.x)**2 + (cy - state.y)**2
            self.target_ind = int(np.argmin(dists))
            self.target_ind_initialized = True

        search_window   = self.config.N_IND_SEARCH
        search_indices  = [(self.target_ind + i) % ncourse for i in range(search_window)]
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

    #########################################################################
    def update_state(self, state, a, delta):
        #clip() is used to ensure that the steering angle and speed remain within the defined limits.
        delta = np.clip(delta, -self.config.MAX_STEER, self.config.MAX_STEER)
        state.x   = state.x + state.v * math.cos(state.yaw) * self.config.DTK
        state.y   = state.y + state.v * math.sin(state.yaw) * self.config.DTK
        state.yaw = (state.yaw
                     + (state.v / self.config.WB) * math.tan(delta) * self.config.DTK)
        state.v   = np.clip(state.v + a * self.config.DTK,
                            self.config.MIN_SPEED, self.config.MAX_SPEED)
        return state

    #########################################################################
    def _get_model_matrix(self, v, phi, delta):
        """
        Calculate linear and discrete time dynamic model
        """
        dt = self.config.DTK
        WB = self.config.WB
        # State matrix A, 4x4
        A  = np.eye(self.config.NXK)
        A[0, 2] =  dt * math.cos(phi)
        A[0, 3] = -dt * v * math.sin(phi)
        A[1, 2] =  dt * math.sin(phi)
        A[1, 3] =  dt * v * math.cos(phi)
        A[3, 2] =  dt * math.tan(delta) / WB

        # Input Matrix B, 4x2
        B = np.zeros((self.config.NXK, self.config.NU))
        B[2, 0] = dt
        B[3, 1] = dt * v / (WB * math.cos(delta)**2)

        C = np.zeros(self.config.NXK)
        C[0] =  dt * v * math.sin(phi) * phi
        C[1] = -dt * v * math.cos(phi) * phi
        C[3] = -dt * v * delta / (WB * math.cos(delta)**2)
        return A, B, C
    ###########################################################################
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

    #############################################################################
    def _linear_mpc_prob_init(self):
        """
        OSQP solves:  min  (1/2) z' P z + q' z
                      s.t. l <= A z <= u

        Decision vector layout (column-major flattening):
            z = [ x_0, x_1, ..., x_T  |  u_0, u_1, ..., u_{T-1} ]
            lengths: NX*(T+1)  and  NU*T
        """
        from scipy.sparse import csc_matrix as _csc
        NX = self.config.NXK
        NU = self.config.NU
        T  = self.config.TK

        self._l_NX = NX
        self._l_NU = NU
        self._l_T  = T
        self._l_nz = NX * (T + 1) + NU * T

        D_MAX = 1
        self._corridor_d_max = D_MAX

        # Symmetrized weights
        Qk  = (self.config.Qk  + self.config.Qk.T)  / 2
        Qfk = (self.config.Qfk + self.config.Qfk.T) / 2
        Rk  = (self.config.Rk  + self.config.Rk.T)  / 2
        Rdk = (self.config.Rdk + self.config.Rdk.T)  / 2
        self._l_Qk  = Qk
        self._l_Qfk = Qfk
        self._l_Rk  = Rk
        self._l_Rdk = Rdk

        # OSQP minimizes (1/2) z' P z + q' z
        nz = self._l_nz

        def xi(t): return NX * t               
        def ui(t): return NX * (T + 1) + NU * t  

        # Cost Hessian P construction 
        P = np.zeros((nz, nz))
        for t in range(T):
            s = xi(t)
            P[s:s+NX, s:s+NX] += 2 * Qk
        s = xi(T)
        P[s:s+NX, s:s+NX] += 2 * Qfk
        for t in range(T):
            s = ui(t)
            P[s:s+NU, s:s+NU] += 2 * Rk

        # Add rate penalty blocks for t=0...T-2, which involve ui(t) and ui(t+1)
        for t in range(T - 1):           # rate penalty tridiagonal block
            s0, s1 = ui(t), ui(t + 1)
            P[s0:s0+NU, s0:s0+NU] += 2 * Rdk
            P[s1:s1+NU, s1:s1+NU] += 2 * Rdk
            P[s0:s0+NU, s1:s1+NU] -= 2 * Rdk  # upper triangle (s0 < s1)

        self._l_P_csc = _csc(np.triu(P))

        # Warm-start storage (primal z, dual y)
        self._osqp_z_ws = None
        self._osqp_y_ws = None

        self.get_logger().info(
            f'LMPC: direct OSQP ready : nz={nz}, TK={T}, NX={NX}, NU={NU}')

    #####################################################################################
    def _run_osqp(self, P_csc, q, A_csc, l_v, u_v, z_ws, y_ws):
        """
        OSQP uses ADMM (Alternating Direction Method of Multipliers), an
        operator-splitting algorithm suited for large sparse QPs. It iterates
        on primal z and dual y variables until KKT conditions are satisfied
        within eps_abs and eps_rel tolerances.
        """
        import osqp as _osqp
        prob = _osqp.OSQP()

        prob.setup(
            P_csc, q, A_csc, l_v, u_v,
            warm_starting     = False,
            eps_abs           = 1e-3,
            eps_rel           = 1e-3,
            max_iter          = 8000,
            polish            = True,
            verbose           = False,
            adaptive_rho      = True,
            check_termination = 10,
        )

        # Warm-start with previous solution if available and dimensionally compatible
        if y_ws is not None and len(y_ws) == A_csc.shape[0]:
            prob.warm_start(x=z_ws, y=y_ws)
        else:
            prob.warm_start(x=z_ws)
        return prob.solve()

    #######################################################################################
    def _linear_mpc_prob_solve(self, ref_traj, path_predict, x0):
        """
        Build and solve the linearised MPC QP for the current tick.
        """
        from scipy.sparse import lil_matrix, csc_matrix as _csc

        NX, NU, T = self._l_NX, self._l_NU, self._l_T
        nz        = self._l_nz
        D_MAX     = self._corridor_d_max

        def xi(t): return NX * t
        def ui(t): return NX * (T + 1) + NU * t

        #Reference trajectory angle unwrapping and continuity enforcement
        ref_traj = ref_traj.copy()
        ref_traj[3, :] = x0[3] + wrap_angle(ref_traj[3, :] - x0[3])
        for i in range(1, ref_traj.shape[1]):
            d = ref_traj[3, i] - ref_traj[3, i - 1]
            if d >  np.pi: ref_traj[3, i] -= 2 * np.pi
            if d < -np.pi: ref_traj[3, i] += 2 * np.pi

        
        q = np.zeros(nz)
        for t in range(T):
            q[xi(t):xi(t)+NX] = -2 * self._l_Qk @ ref_traj[:, t]
        q[xi(T):xi(T)+NX] = -2 * self._l_Qfk @ ref_traj[:, T]

        # Pre-compute the time-varying model matrices along the reference trajectory => For the dynamics constraints.
        A_mats, B_mats, C_vecs = [], [], []
        for t in range(T):
            delta_bar = float(self.odelta_v[t]) if self.odelta_v is not None else 0.0
            A, B, C = self._get_model_matrix(
                float(path_predict[2, t]), float(path_predict[3, t]), delta_bar)
            A_mats.append(A); B_mats.append(B); C_vecs.append(C)

        # Compute corridor normals and origins for the constraints. 
        normals = np.zeros((2, T + 1))
        origins = np.zeros((2, T + 1))
        for t in range(T + 1):
            dx = (ref_traj[0, t+1]-ref_traj[0, t]) if t < T else (ref_traj[0, t]-ref_traj[0, t-1])
            dy = (ref_traj[1, t+1]-ref_traj[1, t]) if t < T else (ref_traj[1, t]-ref_traj[1, t-1])
            nm = np.hypot(dx, dy)
            normals[:, t] = [-dy/nm, dx/nm] if nm >= 1e-6 else [0.0, 1.0]
            origins[:, t] = ref_traj[:2, t]

        #########################################################################
        """
          r0_init  [NX]         : x_0 = x0 (equality)
          r0_dyn   [NX*T]       : x_{t+1} = A_t x_t + B_t u_t + C_t
          r0_vspd  [T+1]        : speed bounds on each state
          r0_accel [T]          : accel magnitude
          r0_steer [T]          : steer magnitude
          r0_srate [T-1]        : steer   <= MAX_DSTEER * DTK
          r0_cor   [T]          : corridor (t=1..T)"""
        r0_init  = 0
        r0_dyn   = r0_init  + NX
        r0_vspd  = r0_dyn   + NX * T
        r0_accel = r0_vspd  + (T + 1)
        r0_steer = r0_accel + T
        r0_srate = r0_steer + T
        r0_cor   = r0_srate + (T - 1)
        n_rows   = r0_cor   + T

        A_sp = lil_matrix((n_rows, nz)) #Sparse constraint matrix in LIL format for efficient construction; will convert to CSC for OSQP
        l_v  = np.full(n_rows, -np.inf)
        u_v  = np.full(n_rows,  np.inf)

        # Initial state equality
        for j in range(NX):
            A_sp[r0_init + j, xi(0) + j] = 1.0
        l_v[r0_init:r0_init+NX] = x0
        u_v[r0_init:r0_init+NX] = x0

        # I·x_{t+1} - A_t·x_t - B_t·u_t = C_t
        for t in range(T):
            rb = r0_dyn + NX * t
            for i in range(NX):
                A_sp[rb + i, xi(t+1) + i] = 1.0        # I block for x_{t+1}
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

        # Angle bounds
        max_ds = self.config.MAX_DSTEER * self.config.DTK
        for t in range(T - 1):
            A_sp[r0_srate + t, ui(t+1) + 1] =  1.0
            A_sp[r0_srate + t, ui(t)   + 1] = -1.0
            l_v[r0_srate + t] = -max_ds
            u_v[r0_srate + t] =  max_ds

        # Corridor half-plane (t=1..T)
        for idx, t in enumerate(range(1, T + 1)):
            n = normals[:, t]
            p_ref = origins[:, t]
            A_sp[r0_cor + idx, xi(t)]     = n[0]
            A_sp[r0_cor + idx, xi(t) + 1] = n[1]
            lat_ref = float(n @ p_ref)
            l_v[r0_cor + idx] = -D_MAX + lat_ref
            u_v[r0_cor + idx] =  D_MAX + lat_ref

        A_csc = _csc(A_sp)
        ##############################################################################
        # Warm-start primal (shifted previous solution) and dual (previous multipliers)
        if self._prev_xk is not None and self._prev_uk is not None:
            x_ws         = np.zeros((NX, T + 1))
            u_ws         = np.zeros((NU, T))
            x_ws[:, :-1] = self._prev_xk[:, 1:]
            x_ws[:, -1]  = self._prev_xk[:, -1]
            u_ws[:, :-1] = self._prev_uk[:, 1:]
            u_ws[:, -1]  = self._prev_uk[:, -1]
            z_ws = np.concatenate([x_ws.flatten(order='F'),
                                   u_ws.flatten(order='F')])
        else:
            z_ws = np.concatenate([ref_traj.flatten(order='F'),
                                   np.zeros(NU * T)])

        
        oa     = self.oa       if self.oa       is not None else np.zeros(T)
        odelta = self.odelta_v if self.odelta_v is not None else np.zeros(T)
        ox = oy = oyaw = None

        
        _INFEASIBLE = {'primal infeasible', 'primal infeasible inaccurate'}
        try:
            res = self._run_osqp(
                self._l_P_csc, q, A_csc, l_v, u_v, z_ws, self._osqp_y_ws)

            #Fallback: strip corridor rows and retry if infeasible, especially at high speed when the corridor is tight.
            if res.info.status in _INFEASIBLE:
                print(f"LMPC : {res.info.status} retrying without corridor")
                # Discard dual variables from an infeasible problem 
                self._osqp_y_ws = None
                A_nc = _csc(A_sp[:r0_cor, :])
                l_nc = l_v[:r0_cor].copy()
                u_nc = u_v[:r0_cor].copy()
                res  = self._run_osqp(
                    self._l_P_csc, q, A_nc, l_nc, u_nc, z_ws, None)

            # Update warm-start for next tick.
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
                self._lmpc_solve_ok = True
                print(f"LMPC : {res.info.status}")
            else:
                # In case of complete failure, clear warm-start so next tick is fresh
                self._lmpc_solve_ok = False
                self._osqp_y_ws = None
                self._prev_xk   = None
                self._prev_uk   = None
                print(f"LMPC : no valid iterate {res.info.status}")

        except Exception as e:
            self._lmpc_solve_ok = False
            self._osqp_y_ws = None
            print(f"LMPC : exception - {e}")

        return oa, odelta, ox, oy, oyaw

    def _linear_mpc_control(self, ref_path, x0, oa, od):
        
        if oa is None or od is None:
            oa = [0.0] * self.config.TK
            od = [0.0] * self.config.TK
        x0_pred = list(x0)
        if abs(x0_pred[2]) < 0.5:
            x0_pred[2] = float(ref_path[2, 1])
        path_predict = self._predict_motion(x0_pred, oa, od, ref_path)
        mpc_a, mpc_d, mpc_x, mpc_y, mpc_yaw = self._linear_mpc_prob_solve(
            ref_path, path_predict, x0)
        return mpc_a, mpc_d, mpc_x, mpc_y, mpc_yaw, path_predict

    ##########################################################################
    # NMPC  -  scipy.optimize.minimize  (SLSQP)
    ##########################################################################

    def nmpc_prob_init(self):
        """
        Pre-compute and cache cost matrices for the SLSQP NMPC.
        scipy's SLSQP solves it by repeatedly forming and
        solving a QP approximation (the 'SQP' in SLSQP) using finite-difference
        gradients, then doing an L1 merit-function line search.
        """        
        NX = self.config.NXK
        NU = self.config.NU
        T  = self.config.TK

        self._nmpc_NX = NX
        self._nmpc_NU = NU
        self._nmpc_T  = T

        # Block-diagonal weight matrices (stored for fast objective evaluation)
        self._nmpc_R  = block_diag([self.config.Rk]  * T).toarray()
        self._nmpc_Rd = block_diag([self.config.Rdk] * (T - 1)).toarray()
        self._nmpc_Q  = block_diag([self.config.Qk]  * T + [self.config.Qfk]).toarray()

        # Size constants
        self._nmpc_nx = NX * (T + 1)   
        self._nmpc_nu = NU * T          
        self._nmpc_nz = self._nmpc_nx + self._nmpc_nu

        # Precompute D^T Rd D is precomputed for the analytical rate-cost gradient
        D = np.zeros((NU * (T - 1), NU * T))
        for j in range(T - 1):
            D[NU*j:NU*(j+1), NU*j:NU*(j+1)]     = -np.eye(NU)
            D[NU*j:NU*(j+1), NU*(j+1):NU*(j+2)] =  np.eye(NU)
        self._nmpc_DtRdD = D.T @ self._nmpc_Rd @ D

        # Corridor half-width
        self._corridor_d_max = 1

        self.get_logger().info(
            f'NMPC (scipy/SLSQP): ready  - horizon T={T}, '
            f'decision vars={self._nmpc_nz}')

    ##########################################################################   
    def _nmpc_unpack(self, z):
        """
        Unpack the flat decision vector z into (xk, uk) matrices.
        Returns:
            xk : (NX, T+1) state trajectory
            uk : (NU, T)   control trajectory
        """
        NX, NU, T = self._nmpc_NX, self._nmpc_NU, self._nmpc_T
        xk = z[:NX * (T + 1)].reshape(NX, T + 1, order='F')
        uk = z[NX * (T + 1):].reshape(NU, T, order='F')
        return xk, uk

    ##########################################################################
    def _nmpc_objective(self, z, ref_traj, normals=None, origins=None):
        """
        Evaluate the NMPC cost at decision vector z.

        The cost has three terms:
            J_track = (x - x_ref)' Q (x - x_ref)     - tracking error over horizon
            J_input = u' R u                         - input magnitude penalty
            J_rate  = Δu' Rd Δu                      - input rate penalty (smoothness)

        Plus a SOFT corridor penalty (quadratic hinge):
            J_cor = W_cor * sum_t  max(0, |n_t·x_t^xy - n_t·p_ref_t| - D_MAX)^2

        Adding a hard lateral inequality shrinks thatfeasible set further, 
        and when it becomes near-empty (e.g. car close to the corridor wall), 
        SLSQP's QP subproblem becomes infeasible and the solver fails.
        """
        xk, uk = self._nmpc_unpack(z)
        x_err  = (xk - ref_traj).flatten(order='F')
        u_flat = uk.flatten(order='F')
        du     = (uk[:, 1:] - uk[:, :-1]).flatten(order='F')
        cost = (float(x_err @ self._nmpc_Q @ x_err) +
                float(u_flat @ self._nmpc_R @ u_flat) +
                float(du @ self._nmpc_Rd @ du))

        # Soft corridor penalty: quadratic hinge on lateral deviation > D_MAX
        if normals is not None:
            D_MAX = self._corridor_d_max
            NX, T = self._nmpc_NX, self._nmpc_T
            W_cor = 2000.0      #Very high weight to strongly discourage violations while keeping the problem feasible
            for t in range(1, T + 1):
                n   = normals[:, t]
                lat = n[0] * xk[0, t] + n[1] * xk[1, t]
                lr  = float(n @ origins[:, t])
                viol = abs(lat - lr) - D_MAX
                if viol > 0:
                    cost += W_cor * viol ** 2
        return cost
    ##########################################################################
    def _nmpc_objective_jac(self, z, ref_traj, normals=None, origins=None):
        """
        Analytical gradient of _nmpc_objective w.r.t. z.
        Gradient components:
            g_x = 2 Q (x - x_ref)    - w.r.t. state block of z
            g_u = 2 R u + 2 D'Rd D u - w.r.t. control block of z
                  (second term uses the precomputed D'RdD matrix)

        Corridor hinge gradient (only where violation > 0):
            d/d(x_t^xy) [ W_cor * (|dev| - D_MAX)^2 ] = 2 W_cor * viol * sign(dev) * n_t
        where dev = n_t · x_t^xy - n_t · p_ref_t.
        """
        xk, uk = self._nmpc_unpack(z)
        NX, T  = self._nmpc_NX, self._nmpc_T
        g_x = 2.0 * (self._nmpc_Q @ (xk - ref_traj).flatten(order='F'))
        g_u = 2.0 * (self._nmpc_R @ uk.flatten(order='F')) + \
              2.0 * (self._nmpc_DtRdD @ uk.flatten(order='F'))
       
        # Gradient of soft corridor penalty
        if normals is not None:
            D_MAX = self._corridor_d_max
            W_cor = 2000.0
            for t in range(1, T + 1):
                n   = normals[:, t]
                lat = n[0] * xk[0, t] + n[1] * xk[1, t]
                lr  = float(n @ origins[:, t])
                dev = lat - lr
                viol = abs(dev) - D_MAX
                if viol > 0:
                    sign = 1.0 if dev > 0 else -1.0
                    
                    # d(cost)/d(xk[0,t]) and d(cost)/d(xk[1,t])
                    idx_x = NX * t
                    idx_y = NX * t + 1
                    g_x[idx_x] += 2.0 * W_cor * viol * sign * n[0]
                    g_x[idx_y] += 2.0 * W_cor * viol * sign * n[1]
        return np.concatenate([g_x, g_u])

    ##########################################################################
    def _nmpc_dynamics_eq(self, z, x0_arr):
        """
        Nonlinear equality constraint vector for SLSQP, must equal zero at a feasible point.
        Returns a vector of size NX*(T+1):
            residuals[:NX]          = x_0 - x0_arr          (initial state pin)
            residuals[NX + t*NX : ] = x_{t+1} - f(x_t, u_t) (nonlinear dynamics)
        """
        xk, uk = self._nmpc_unpack(z)
        NX, NU, T = self._nmpc_NX, self._nmpc_NU, self._nmpc_T
        dt = self.config.DTK
        WB = self.config.WB

        residuals = np.empty(NX * (T + 1))
        residuals[:NX] = xk[:, 0] - x0_arr

        for t in range(T):
            x_t = xk[:, t]
            u_t = uk[:, t]
            x_next_pred = np.array([
                x_t[0] + x_t[2] * math.cos(x_t[3]) * dt,
                x_t[1] + x_t[2] * math.sin(x_t[3]) * dt,
                x_t[2] + u_t[0] * dt,
                x_t[3] + (x_t[2] / WB) * math.tan(u_t[1]) * dt,
            ])
            residuals[NX + t * NX: NX + (t + 1) * NX] = xk[:, t + 1] - x_next_pred

        return residuals

    ###########################################################################
    def _nmpc_dynamics_jac(self, z, x0_arr):
        """
        Analytical Jacobian of _nmpc_dynamics_eq w.r.t. z, shape (NX*(T+1), nz).

        At each SQP iteration, SLSQP linearises the nonlinear constraints using
        this Jacobian to form the local QP subproblem. The Jacobian is re-evaluated
        at the current iterate z, making it a 're-linearise at current point' approach
        """
        xk, uk = self._nmpc_unpack(z)
        NX, NU, T = self._nmpc_NX, self._nmpc_NU, self._nmpc_T
        J = np.zeros((NX * (T + 1), self._nmpc_nz))

        # residuals[:NX] = xk[:,0] - x0  →  d/d(xk[:,0]) = I
        J[:NX, :NX] = np.eye(NX)
        for t in range(T):
            row      = NX + t * NX
            col_xnxt = NX * (t + 1)
            col_xcur = NX * t
            col_u    = NX * (T + 1) + NU * t
            
            # residual = xk[:,t+1] - f(xk[:,t], uk[:,t])
            J[row:row+NX, col_xnxt:col_xnxt+NX] = np.eye(NX)
            A_t, B_t, _ = self._get_model_matrix(
                float(xk[2, t]), float(xk[3, t]), float(uk[1, t]))
            J[row:row+NX, col_xcur:col_xcur+NX] = -A_t
            
            J[row:row+NX, col_u:col_u+NU]       = -B_t
        return J
    ###########################################################################
    def nmpc_prob_solve(self, ref_traj, x0):
        NX, NU, T = self._nmpc_NX, self._nmpc_NU, self._nmpc_T
        nz        = self._nmpc_nz
        dt        = self.config.DTK
        WB        = self.config.WB

        # Yaw reference wrapping to ensure continuity and correct angle error computation in the objective.
        ref_traj = ref_traj.copy()
        ref_traj[3, 0] = x0[3] + wrap_angle(ref_traj[3, 0] - x0[3])
        for i in range(1, ref_traj.shape[1]):
            d = ref_traj[3, i] - ref_traj[3, i - 1]
            if d >  np.pi: ref_traj[3, i] -= 2 * np.pi
            if d < -np.pi: ref_traj[3, i] += 2 * np.pi

        x0_arr = np.array(x0, dtype=float)

        
        lb = np.full(nz, -np.inf)
        ub = np.full(nz,  np.inf)

        #Speed bounds (state index 2 in each column)
        for i in range(T + 1):
            lb[NX * i + 2] = self.config.MIN_SPEED
            ub[NX * i + 2] = self.config.MAX_SPEED

        # Input magnitude bounds
        x_end = NX * (T + 1)
        for j in range(T):
            base = x_end + NU * j
            lb[base + 0] = -self.config.MAX_ACCEL;  ub[base + 0] = self.config.MAX_ACCEL
            lb[base + 1] = -self.config.MAX_STEER;  ub[base + 1] = self.config.MAX_STEER

        bounds = Bounds(lb, ub)

        # Input rate constraints (inequality)
        rate_constraints = []
        nz     = self._nmpc_nz
        max_ds = self.config.MAX_DSTEER * dt

        for j in range(1, T):
            jc = x_end + NU * j
            jp = x_end + NU * (j - 1)

            def _rfun(z, jc=jc, jp=jp, ds=max_ds):
                return np.array([
                    ds - (z[jc + 1] - z[jp + 1]),
                    ds + (z[jc + 1] - z[jp + 1]),
                ])

            def _rjac(z, jc=jc, jp=jp, _nz=nz):
                G = np.zeros((2, _nz))
                G[0, jc + 1] = -1.0;  G[0, jp + 1] =  1.0
                G[1, jc + 1] =  1.0;  G[1, jp + 1] = -1.0
                return G

            rate_constraints.append({'type': 'ineq', 'fun': _rfun, 'jac': _rjac})

        # Corridor geometry - passed to objective as a soft quadratic hinge penalty.
        # Hard inequality constraints are avoided here because SLSQP's line search
        # can fail when the feasible set shrinks. The penalty weight W_cor=2000
        # strongly discourages violation without shrinking the feasible set.
        D_MAX   = self._corridor_d_max
        normals = np.zeros((2, T + 1))
        origins = np.zeros((2, T + 1))
        for t in range(T + 1):
            dx = (ref_traj[0, t+1] - ref_traj[0, t]) if t < T else (ref_traj[0, t] - ref_traj[0, t-1])
            dy = (ref_traj[1, t+1] - ref_traj[1, t]) if t < T else (ref_traj[1, t] - ref_traj[1, t-1])
            nm = np.hypot(dx, dy)
            normals[:, t] = [-dy/nm, dx/nm] if nm >= 1e-6 else [0.0, 1.0]
            origins[:, t] = ref_traj[:2, t]

        # Dynamics equality constraints (FD Jacobian via linearisation causes linesearch divergence for SLSQP)
        eq_con = {
            'type': 'eq',
            'fun':  lambda z: self._nmpc_dynamics_eq(z, x0_arr),
        }
        all_constraints = [eq_con] + rate_constraints

        # Warm-start initial guess
        if self._prev_xk is not None and self._prev_uk is not None:
            x_init         = np.zeros((NX, T + 1))
            u_init         = np.zeros((NU, T))
            x_init[:, :-1] = self._prev_xk[:, 1:]
            x_init[:, -1]  = self._prev_xk[:, -1]
            u_init[:, :-1] = self._prev_uk[:, 1:]
            u_init[:, -1]  = self._prev_uk[:, -1]
        else:
            x_init = ref_traj.copy()
            u_init = np.zeros((NU, T))

        z0 = np.concatenate([
            x_init.flatten(order='F'),
            u_init.flatten(order='F'),
        ])

        # Solve 
        import warnings
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                res = minimize(
                    fun     = self._nmpc_objective,
                    jac     = self._nmpc_objective_jac,
                    x0      = z0,
                    args    = (ref_traj, normals, origins),
                    method  = 'SLSQP',
                    bounds  = bounds,
                    constraints = all_constraints,
                    options = {
                        'ftol':    1e-4,
                        'maxiter': 400,
                        'disp':    False,
                    },
                )

            xk_sol, uk_sol = self._nmpc_unpack(res.x)

            ox     = xk_sol[0, :]
            oy     = xk_sol[1, :]
            oyaw   = xk_sol[3, :]
            oa     = uk_sol[0, :]
            odelta = uk_sol[1, :]
            self._prev_xk = xk_sol.copy()
            self._prev_uk = uk_sol.copy()
            self._lmpc_solve_ok = True

            if res.success:
                print(f"NMPC : solved ")
            else:
                print(f"NMPC : error ({res.message})")

        except Exception as e:
            print(f"NMPC : failed - {e}")
            oa = self.oa if self.oa is not None else np.zeros(T)
            odelta = self.odelta_v if self.odelta_v is not None else np.zeros(T)
            ox = oy = oyaw = None

        return oa, odelta, ox, oy, oyaw
    ##################################################################################
    
    def log_cte(self, state, intended_x, intended_y, intended_v, intended_yaw,
                label='NMPC'):
        ind  = self.target_ind
        cx   = self.waypoints[ind, 0]
        cy   = self.waypoints[ind, 1]
        cyaw = self.waypoints[ind, 2]
        cte      = -(state.x - cx) * math.sin(cyaw) + (state.y - cy) * math.cos(cyaw)
        yaw_diff = wrap_angle(state.yaw - float(intended_yaw))
        self._csv_writer.writerow([
            round(time.time() - self._session_start, 4),
            label,
            round(state.x,   4), round(state.y,   4),
            round(state.v,   4), round(state.yaw, 4),
            round(abs(cte),  4),
            round(float(intended_x), 4), round(float(intended_y), 4),
            round(float(intended_v), 4), round(yaw_diff,          4),
        ])
        self._csv_file.flush()
    ###################################################################################
    
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
    print("MPC (CVXPY/OSQP + scipy/SLSQP) Initialized")
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
########################################################################################
