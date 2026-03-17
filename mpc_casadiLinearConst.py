#!/usr/bin/env python3
import math
import os
import time

from dataclasses import dataclass, field
import csv
import casadi as ca
import numpy as np
import rclpy

from geometry_msgs.msg import Point
#from visualization_msgs.msg import Marker
#from tf.transformations import euler_from_quaternion
from ackermann_msgs.msg import AckermannDrive, AckermannDriveStamped
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
#from scipy.linalg import block_diag
from scipy.sparse import block_diag
from sensor_msgs.msg import LaserScan
from utils import nearest_point
from builtin_interfaces.msg import Duration
from visualization_msgs.msg import Marker


# TODO CHECK: include needed ROS msg type headers and libraries


@dataclass
class mpc_config:
    NXK: int = 4  # length of kinematic state vector: z = [x, y, v, yaw]
    NU: int = 2  # length of input vector: u = = [steering speed, acceleration]
    TK: int = 20  # finite time horizon length kinematic (20 × 0.2 s = 4 s)
    k = 1
    # ---------------------------------------------------
    Rk: list = field(
    default_factory=lambda: np.diag([
        1.0,    # accel penalty — low so optimizer uses full authority
        1.0,    # steer penalty — low so optimizer uses full authority
    ])
    ) #60*0.05, 0.5 # input cost matrix, penalty for inputs - [steering_speed, accel]
   
    Rdk: list = field(
    default_factory=lambda: np.diag([
        0.6,
        0.6,
    ])
    )  # input difference cost matrix, penalty for change of inputs - [steering_speed, accel]
   
    Qk: list = field(
    default_factory=lambda: np.diag([
        50.0,   # x tracking — high to dominate R, drives fast xy convergence
        50.0,   # y tracking
        5.0,    # velocity tracking — low, don't compete with xy
        10.0,   # yaw tracking — moderate, helps path alignment
    ])
    )  # state error cost matrix
   
    Qfk: list = field(
    default_factory=lambda: np.diag([
        500.0,  # x terminal — 10× Qk forces early convergence (exponential decay)
        500.0,  # y terminal
        5.0,    # velocity terminal
        50.0,   # yaw terminal
    ])
    ) # final state error matrix — large terminal cost is the key to exponential CTE decay
    # ---------------------------------------------------

    N_IND_SEARCH: int = 20  # Search index number
    DTK: float = 0.2  # time step [s] kinematic (coarser step, same 4 s horizon at TK=20)
    dlk: float = 0.03  # dist step [m] kinematic
    LENGTH: float = 0.58  # Length of the vehicle [m]
    WIDTH: float = 0.31  # Width of the vehicle [m]
    WB: float = 0.33  # Wheelbase [m]
    MIN_STEER: float = -0.4189  # maximum steering angle [rad]
    MAX_STEER: float = 0.4189  # maximum steering angle [rad]
    MAX_DSTEER: float = np.deg2rad(180.0)  # maximum steering speed [rad/s]
    MAX_SPEED: float = 6.0  # maximum speed [m/s]
    MIN_SPEED: float = 0.0  # minimum backward speed [m/s]
    MAX_ACCEL: float = 3.0  # maximum acceleration [m/s*s]



def calc_speed_profile(cx, cy, cyaw):
    ncourse = len(cx)
    sp = np.full(ncourse, 5.0)  # straight speed (within MAX_SPEED=5.0)

    CURVE_THRESHOLD = 0.01   # yaw change per waypoint to count as a curve
    CURVE_SPEED     = 2.5    # speed through the curve
    ENTRY_BOOST_SPEED  = 5 # brief speed boost just before braking
    ENTRY_BOOST_COUNT  = 5   # how many waypoints before curve to boost
    EXIT_RAMP_COUNT    = 10  # how many waypoints after curve to ramp back up

    # First pass: mark curve waypoints
    is_curve = np.zeros(ncourse, dtype=bool)
    for i in range(ncourse):
        next_i = (i + 1) % ncourse
        dyaw = abs(cyaw[next_i] - cyaw[i])
        if dyaw > CURVE_THRESHOLD:
            is_curve[i] = True

    # Second pass: apply speeds
    for i in range(ncourse):
        if is_curve[i]:
            sp[i] = CURVE_SPEED
            continue

        # Check if we're just before a curve (entry boost)
        for k in range(1, ENTRY_BOOST_COUNT + 1):
            if is_curve[(i + k) % ncourse]:
                sp[i] = ENTRY_BOOST_SPEED
                break

        # Check if we're just after a curve (ramp back up)
        for k in range(1, EXIT_RAMP_COUNT + 1):
            if is_curve[(i - k) % ncourse]:
                # linearly ramp from CURVE_SPEED back to 5.0
                sp[i] = CURVE_SPEED + (5.0 - CURVE_SPEED) * (k / EXIT_RAMP_COUNT)
                break

    return sp


@dataclass
class State:
    x: float = 0.0
    y: float = 0.0
    delta: float = 0.0
    v: float = 0.0
    yaw: float = 0.0
    yawrate: float = 0.0
    beta: float = 0.0

def wrap_angle(a):
    return np.arctan2(np.sin(a), np.cos(a))

class MPC(Node):
    """ 
    Implement Kinematic MPC on the car
    This is just a template, you are free to implement your own node!
    """
    def __init__(self):
        super().__init__('mpc_node')
        #Nominal publishers and subscribers
        self.drive_pub = self.create_publisher(AckermannDriveStamped,'/drive',10)  #Publisher
        self.pose_sub = self.create_subscription(Odometry,'/ego_racecar/odom', self.pose_callback,10) #Pose Subscriber

        self.waypoints = self.load_waypoints("waypoints.csv") #Waypoints are loaded from a CSV file

        #Target indicator for calculating reference trajectory forward in time for MPC to optimize for
        self.target_ind = 0  # will be corrected on first callback
        self.target_ind_initialized = False # flag to indicate if target_ind has been initialized

        self.prev_odom_yaw = None
        self.yaw_offset = 0.0

        # MPC configuration parameters
        self.config = mpc_config() 
        self.odelta_v = None       # warm-start: previous input trajectory (steering speed) from last solve
        self.odelta = None      # warm-start: previous input trajectory (steering angle) from last solve
        self.oa = None          # warm-start: previous input trajectory (acceleration) from last solve
        self.init_flag = 0      # flag to indicate if MPC has been initialized with a valid trajectory
        self._prev_xk = None      # warm-start: previous state trajectory
        self._prev_uk = None      # warm-start: previous input trajectory

        #Specifics for path error logging in a CSV file
        self._session_start = time.time()                                       # session start time for logging
        self._csv_path = os.path.join(os.getcwd(), 'llmpc_performance.csv')     # Path to CSV file for logging
        self._csv_file = open(self._csv_path, 'w', newline='')                  # Open CSV file for writing 
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(['time_s', 'label', 'x', 'y', 'v', 'yaw', 'cte', 'abs_cte'])
        self._csv_file.flush()
        self.get_logger().info(f'Logging MPC performance to: {self._csv_path}')

        #Reference trajectory publishing
        self.ref_pub = self.create_publisher(Marker, '/ego_racecar/mpc_ref_traj', 10) # Publisher for visualizing the reference trajectory
        self.ref_id = 0
        self.ref_timer = self.create_timer(0.5, self.visualize_ref_path) # Timer
        self.traj_pub = self.create_publisher(
            Marker,
            '/ego_racecar/driven_traj',
            10
        ) 

        # Marker for visualizing the driven trajectory
        self.traj_marker = Marker()  
        self.traj_marker.header.frame_id = "ego_racecar/odom" # 
        self.traj_marker.ns = "driven_path" #
        self.traj_marker.id = 100
        self.traj_marker.type = Marker.LINE_STRIP
        self.traj_marker.action = Marker.ADD
        self.traj_marker.scale.x = 0.05
        self.traj_marker.color.r = 0.0
        self.traj_marker.color.g = 1.0
        self.traj_marker.color.b = 0.0
        self.traj_marker.color.a = 1.0
        self.traj_marker.pose.orientation.w = 1.0
        self.traj_marker.lifetime = Duration(sec=0)
        self.traj_marker.points = []
        self.last_traj_x = None
        self.last_traj_y = None

        # Left wall  = reference + d_max * normal  (cyan); Right wall = reference - d_max * normal  (orange)
        self.corridor_pub_left  = self.create_publisher(Marker, '/ego_racecar/corridor_left',  10)
        self.corridor_pub_right = self.create_publisher(Marker, '/ego_racecar/corridor_right', 10)

        # Initialize MPC problem solver : Linear/Non-linear
        self._linear_mpc_prob_init()

    def visualize_ref_path(self):

        marker = Marker()
        marker.header.frame_id = "ego_racecar/odom"
        marker.header.stamp = self.get_clock().now().to_msg()


        marker.ns = "ref_path"
        marker.id = 0

        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        
        marker.scale.x = 0.05

        # BRIGHT RED
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        marker.pose.orientation.w = 1.0
        marker.scale.y = 0.0
        marker.scale.z = 0.0


        marker.lifetime = Duration(sec=0)  #  - Doesn't fade out

        marker.points = []

        for i in range(len(self.waypoints)):
            p = Point()
            p.x = float(self.waypoints[i, 0])
            p.y = float(self.waypoints[i, 1])
            p.z = 0.05
            marker.points.append(p)

        self.ref_pub.publish(marker)

    def load_waypoints(self, path):

        data = []
        with open(path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                x, y, yaw = map(float, row)
                data.append([x, y, yaw])

        wp = np.array(data)
        wp[:, 2] = np.unwrap(wp[:, 2])

        x0, y0, yaw0 = wp[0]
        x1, y1, yaw1 = wp[-1]

        gap = np.hypot(x1 - x0, y1 - y0)

        if gap > 0.05:   # only close if actually broken
            N_interp = max(int(gap / 0.03), 2)  # at least 2 points

            # Parametric interpolation — interior points only (exclude endpoints which already exist as wp[-1] and wp[0] to avoid duplicates)
            t = np.linspace(0.0, 1.0, N_interp + 2)[1:-1]
            x_new   = x1   + t * (x0   - x1)
            y_new   = y1   + t * (y0   - y1)
            yaw_new = np.zeros(len(t))  # bridge is straight, force yaw=0  - Had to set this for the path to be straight and avoid unwrapping issues that came in

            bridge = np.vstack([x_new, y_new, yaw_new]).T
            wp = np.vstack([wp, bridge])

        
        # Compute cumulative arc-length parameter s along the path
        diffs = np.diff(wp[:, :2], axis=0)                   # (N-1, 2)
        seg_lengths = np.hypot(diffs[:, 0], diffs[:, 1])     # (N-1,)
        s = np.concatenate([[0.0], np.cumsum(seg_lengths)])   # (N,)
        total_length = s[-1]

        # New uniform sample points at dlk=0.03 m spacing
        dlk = 0.03
        s_new = np.arange(0.0, total_length, dlk)

        # Linear interpolation of x, y, yaw along arc-length
        wp_x   = np.interp(s_new, s, wp[:, 0])
        wp_y   = np.interp(s_new, s, wp[:, 1])
        wp_yaw = np.interp(s_new, s, wp[:, 2])

        wp = np.vstack([wp_x, wp_y, wp_yaw]).T

        return wp

    def pose_callback(self, pose_msg):

    # --- Extract pose ---
        x = pose_msg.pose.pose.position.x
        y = pose_msg.pose.pose.position.y

        q = pose_msg.pose.pose.orientation
        t3 = 2.0 * (q.w * q.z + q.x * q.y)
        t4 = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        
        yaw_raw = math.atan2(t3, t4)

        if self.prev_odom_yaw is None:
            self.prev_odom_yaw = yaw_raw

        dyaw = yaw_raw - self.prev_odom_yaw

        if dyaw > np.pi:
            self.yaw_offset -= 2*np.pi
        elif dyaw < -np.pi:
            self.yaw_offset += 2*np.pi

        yaw_z = yaw_raw + self.yaw_offset
        self.prev_odom_yaw = yaw_raw



    # --- Extract speed from odometry ---
        vx = pose_msg.twist.twist.linear.x
        vy = pose_msg.twist.twist.linear.y
        v = math.sqrt(vx**2 + vy**2)

        vehicle_state = State(x=x, y=y, v=v, yaw=yaw_z)

        # stamp trail every 5cm
        if self.last_traj_x is None:
            self.last_traj_x = vehicle_state.x
            self.last_traj_y = vehicle_state.y

        dist = math.hypot(
            vehicle_state.x - self.last_traj_x,
            vehicle_state.y - self.last_traj_y
        )

        if dist > 0.05:

            p = Point()
            p.x = float(vehicle_state.x)
            p.y = float(vehicle_state.y)
            p.z = 0.02

            self.traj_marker.points.append(p)

            self.last_traj_x = vehicle_state.x
            self.last_traj_y = vehicle_state.y

            self.traj_marker.header.stamp = self.get_clock().now().to_msg()

            self.traj_pub.publish(self.traj_marker)


    # --- Build reference from loaded waypoints ---
        ref_x   = self.waypoints[:, 0]
        ref_y   = self.waypoints[:, 1]
        ref_yaw = self.waypoints[:, 2]

    # simple constant speed profile (needed by MPC)
        ref_v = calc_speed_profile(ref_x, ref_y, ref_yaw)

        ref_path = self.calc_ref_trajectory(vehicle_state, ref_x, ref_y, ref_yaw, ref_v)
        


        x0 = [vehicle_state.x, vehicle_state.y, vehicle_state.v, vehicle_state.yaw]

    # --- Solve Linear MPC ---
        (self.oa, self.odelta_v, ox, oy, oyaw, ov, _) = self._linear_mpc_control(
            ref_path, x0, self.oa, self.odelta_v
        )

    # --- Log performance (always, even on solver failure) ---
        self.log_cte(vehicle_state, label='llMPC')

        if self.oa is None:

            drive = AckermannDriveStamped()
            drive.drive.steering_angle = 0.0
            drive.drive.speed = vehicle_state.v

            try:
                self.drive_pub.publish(drive)
            except Exception:
                pass
            return



        steer_output = float(self.odelta_v[0])
        speed_output = float(vehicle_state.v + self.oa[0] * self.config.DTK)

    # --- Publish ---
        drive = AckermannDriveStamped()
        drive.drive.steering_angle = steer_output
        drive.drive.speed = speed_output
        self.visualize_ref_path()

        try:
            self.drive_pub.publish(drive)
        except Exception:
            pass


    # LINEARIZED
    def _linear_mpc_prob_init(self):
        """
        Create MPC quadratic optimization problem using CasADi Opti stack.
        Solver: IPOPT (handles convex QP and general NLP alike).
        Problem structure is built once; A/B/C dynamics matrices are CasADi
        parameters that are updated cheaply on every solve call.
        """
        NX = self.config.NXK   # 4  (x, y, v, yaw)
        NU = self.config.NU    # 2  (accel, steer)
        T  = self.config.TK    # 4  (horizon steps)

        self.opti = ca.Opti()

        # Decision variables
        self.xk = self.opti.variable(NX, T + 1)   # predicted states => T+1 states
        self.uk = self.opti.variable(NU, T)        # control inputs

        # Parameters (values injected each solve call) 
        self.x0k        = self.opti.parameter(NX)          # initial current state
        self.ref_traj_k = self.opti.parameter(NX, T + 1)   # reference trajectory T+1 states

        # Linearised dynamics:  vec(x[:,1:]) = Ak @ vec(x[:,:-1]) + Bk @ vec(u) + Ck
        self.Ak_param = self.opti.parameter(NX * T, NX * T)
        self.Bk_param = self.opti.parameter(NX * T, NU * T)
        self.Ck_param = self.opti.parameter(NX * T)

        # Cost weight matrices (constant, built once as dense numpy) 
        R_block  = block_diag([self.config.Rk]  * T).toarray()
        Rd_block = block_diag([self.config.Rdk] * (T - 1)).toarray()
        Q_block  = block_diag([self.config.Qk]  * T + [self.config.Qfk]).toarray()

        # Objective 
        # 1) Input penalty:      u' R u
        u_vec  = ca.vec(self.uk)                              # (NU*T, 1)
        obj    = ca.mtimes([u_vec.T, R_block,  u_vec])

        # 2) Tracking penalty:   (x - x_ref)' Q (x - x_ref)
        x_err  = ca.vec(self.xk - self.ref_traj_k)           # (NX*(T+1), 1)
        obj   += ca.mtimes([x_err.T, Q_block, x_err])

        # 3) Input-rate penalty: Δu' Rd Δu
        du     = ca.vec(self.uk[:, 1:] - self.uk[:, :-1])    # (NU*(T-1), 1)
        obj   += ca.mtimes([du.T, Rd_block, du])

        self.opti.minimize(obj)

        # ── Corridor parameters (linear bounding constraints) ─────────────
        # path_normals_lk : (2, T+1) — unit left-perpendicular to the path
        #                              tangent at each horizon step.
        # path_origins_lk : (2, T+1) — reference (x, y) at each step.
        # Both are Opti parameters: they are cheap to update every solve
        # without rebuilding the problem structure.
        self.path_normals_lk = self.opti.parameter(2, T + 1)
        self.path_origins_lk = self.opti.parameter(2, T + 1)

        # ── Constraints ───────────────────────────────────────────────────
        # 1) Linearised dynamics over horizon
        x_next = ca.vec(self.xk[:, 1:])    # (NX*T, 1)
        x_curr = ca.vec(self.xk[:, :-1])   # (NX*T, 1)
        u_flat = ca.vec(self.uk)            # (NU*T, 1)
        self.opti.subject_to(
            x_next == ca.mtimes(self.Ak_param, x_curr)
                    + ca.mtimes(self.Bk_param, u_flat)
                    + self.Ck_param
        )

        # 2) Accel rate limit
        self.opti.subject_to(
            ca.fabs(self.uk[0, 1:] - self.uk[0, :-1]) <= 5.0
        )

        # 3) Steering rate limit
        self.opti.subject_to(
            ca.fabs(self.uk[1, 1:] - self.uk[1, :-1])
            <= self.config.MAX_DSTEER * self.config.DTK
        )

        # 4) Initial state pin
        self.opti.subject_to(self.xk[:, 0] == self.x0k)

        # 5) Speed bounds on all horizon steps
        self.opti.subject_to(self.xk[2, :] >= self.config.MIN_SPEED)
        self.opti.subject_to(self.xk[2, :] <= self.config.MAX_SPEED)

        # 6) Input magnitude bounds
        self.opti.subject_to(ca.fabs(self.uk[0, :]) <= self.config.MAX_ACCEL)
        self.opti.subject_to(ca.fabs(self.uk[1, :]) <= self.config.MAX_STEER)

        # 7) Linear lateral corridor constraints 
        # For each horizon step t, the predicted position p_t = (x_t, y_t)
        # must lie within ±MAX_LATERAL_DEV of the reference point along the
        # direction perpendicular to the path tangent (the unit left-normal).
        
        #   n_t' * (p_t - p_ref_t)  <=  MAX_LATERAL_DEV    (left wall)
        #  -n_t' * (p_t - p_ref_t)  <=  MAX_LATERAL_DEV    (right wall)
        
       
        D_MAX = 0.3        # [m]  half-width of the lateral corridor
        self._corridor_d_max = D_MAX   # store for marker publishing
        
        for t in range(1, T + 1):
            n    = self.path_normals_lk[:, t]          # (2,) symbolic column
            p    = self.xk[:2, t]                      # (2,) predicted [x, y]
            p_r  = self.path_origins_lk[:, t]          # (2,) reference [x, y]
            lat  = ca.dot(n, p - p_r)                  # signed lateral error
            self.opti.subject_to( lat <=  D_MAX)       # left  wall
            self.opti.subject_to(-lat <=  D_MAX)       # right wall

        
        solver_opts = {
            'ipopt.print_level':           0,
            'ipopt.max_iter':              10000,   # effectively unlimited
            'ipopt.tol':                   1e-3,
            'ipopt.acceptable_tol':        1e-2,
            'ipopt.acceptable_iter':       5,
            'ipopt.warm_start_init_point': 'yes',
            'print_time':                  0,
        }
        self.opti.solver('ipopt', solver_opts)

    def calc_ref_trajectory(self, state, cx, cy, cyaw, sp):
        """
        calc. reference trajectory, ref_traj in T steps: [x, y, v, yaw]
        using the current velocity, calc the T points along the reference path
        :param cx: Course X-Position
        :param cy: Course y-Position
        :param cyaw: Course Heading
        :param sp: speed profile
        :dl: distance step
        :pind: Setpoint Index
        :return: reference trajectory ref_traj, reference steering angle
        """

        # Create placeholder Arrays for the reference trajectory for T steps
        ref_traj = np.zeros((self.config.NXK, self.config.TK + 1))
        #ref_traj = np.zeros((self.config.NXK, self.config.TK + 1))
        ncourse = len(cx)

        if not self.target_ind_initialized:
            dists = (cx - state.x)**2 + (cy - state.y)**2
            self.target_ind = int(np.argmin(dists))
            self.target_ind_initialized = True

        # Search only within a local window ahead of last index
        search_window = self.config.N_IND_SEARCH  # = 20
        
        if not hasattr(self, 'target_ind'):
            self.target_ind = 0

        # Build local search candidates
        search_indices = [(self.target_ind + i) % ncourse for i in range(search_window)]
        local_cx = np.array([cx[i] for i in search_indices])
        local_cy = np.array([cy[i] for i in search_indices])

        # Find nearest within local window only
        dx = local_cx - state.x
        dy = local_cy - state.y
        dists = dx**2 + dy**2
        local_best = np.argmin(dists)
        
        # Only advance index, never go backward
        if local_best > 0:
            self.target_ind = search_indices[local_best]

        ind = self.target_ind

        # Load the initial parameters from the setpoint into the trajectory
        ref_traj[0, 0] = cx[ind]
        ref_traj[1, 0] = cy[ind]
        ref_traj[2, 0] = sp[ind]
        ref_traj[3, 0] = cyaw[ind]

        # based on current velocity, distance traveled on the ref line between time steps
        travel = 0.8 * self.config.DTK
        dind = travel / self.config.dlk
        ind_list = int(ind) + np.insert(
            np.cumsum(np.repeat(dind, self.config.TK)), 0, 0
        ).astype(int)
        
        ind_list = ind_list % ncourse
        ref_traj[0, :] = cx[ind_list]
        ref_traj[1, :] = cy[ind_list]
        ref_traj[2, :] = sp[ind_list]
        
        for i in range(len(ind_list)):
            ref_traj[3, i] = cyaw[ind_list[i]]

        return ref_traj
    # ── LINEARIZED — kept for reference ──────────────────────────────────
    def _predict_motion(self, x0, oa, od, xref):

        path_predict = xref * 0.0

        # initial state
        for i in range(len(x0)):
            path_predict[i, 0] = x0[i]

        state = State(
            x=x0[0],
            y=x0[1],
            v=x0[2],
            yaw=x0[3]
        )

        for i in range(1, self.config.TK + 1):

            ai = oa[i-1]
            di = od[i-1]

            state = self.update_state(state, ai, di)

            path_predict[0, i] = state.x
            path_predict[1, i] = state.y
            path_predict[2, i] = state.v
            path_predict[3, i] = state.yaw

        return path_predict


    def update_state(self, state, a, delta):

        # input check
        if delta >= self.config.MAX_STEER:
            delta = self.config.MAX_STEER
        elif delta <= -self.config.MAX_STEER:
            delta = -self.config.MAX_STEER

        state.x = state.x + state.v * math.cos(state.yaw) * self.config.DTK
        state.y = state.y + state.v * math.sin(state.yaw) * self.config.DTK
        state.yaw = (
            state.yaw + (state.v / self.config.WB) * math.tan(delta) * self.config.DTK
        )
        #state.yaw = wrap_angle(state.yaw)

        state.v = state.v + a * self.config.DTK

        if state.v > self.config.MAX_SPEED:
            state.v = self.config.MAX_SPEED
        elif state.v < self.config.MIN_SPEED:
            state.v = self.config.MIN_SPEED

        return state

    # ── LINEARIZED — kept for reference ──────────────────────────────────
    def _get_model_matrix(self, v, phi, delta):
        """
        Calc linear and discrete time dynamic model-> Explicit discrete time-invariant
        Linear System: Xdot = Ax +Bu + C
        State vector: x=[x, y, v, yaw]
        :param v: speed
        :param phi: heading angle of the vehicle
        :param delta: steering angle: delta_bar
        :return: A, B, C
        """

        # State (or system) matrix A, 4x4
        A = np.zeros((self.config.NXK, self.config.NXK))
        A[0, 0] = 1.0
        A[1, 1] = 1.0
        A[2, 2] = 1.0
        A[3, 3] = 1.0
        A[0, 2] = self.config.DTK * math.cos(phi)
        A[0, 3] = -self.config.DTK * v * math.sin(phi)
        A[1, 2] = self.config.DTK * math.sin(phi)
        A[1, 3] = self.config.DTK * v * math.cos(phi)
        A[3, 2] = self.config.DTK * math.tan(delta) / self.config.WB

        # Input Matrix B; 4x2
        B = np.zeros((self.config.NXK, self.config.NU))
        B[2, 0] = self.config.DTK
        B[3, 1] = self.config.DTK * v / (self.config.WB * math.cos(delta) ** 2)

        C = np.zeros(self.config.NXK)
        C[0] = self.config.DTK * v * math.sin(phi) * phi
        C[1] = -self.config.DTK * v * math.cos(phi) * phi
        C[3] = -self.config.DTK * v * delta / (self.config.WB * math.cos(delta) ** 2)

        return A, B, C

    # ── LINEARIZED — kept for reference 
    def _linear_mpc_prob_solve(self, ref_traj, path_predict, x0):
        # ── Update current-state parameter ───────────────────────────────
        self.opti.set_value(self.x0k, x0)

        # Update linear corridor constraint parameters 
        # For each horizon step t compute the unit left-perpendicular to the
        # path tangent from consecutive reference positions, then inject both
        # the normal vectors and the reference origins into the Opti problem.
        
        # This refreshes the half-plane corridor walls without rebuilding the problem 
        T       = self.config.TK
        normals = np.zeros((2, T + 1))
        origins = np.zeros((2, T + 1))

        for t in range(T + 1):
            # Estimate tangent direction using forward (or backward at the end)
            if t < T:
                dx = ref_traj[0, t + 1] - ref_traj[0, t]
                dy = ref_traj[1, t + 1] - ref_traj[1, t]
            else:
                dx = ref_traj[0, t] - ref_traj[0, t - 1]
                dy = ref_traj[1, t] - ref_traj[1, t - 1]
            norm = np.hypot(dx, dy)
            if norm < 1e-6:
                # Degenerate tangent — fall back to the y-axis normal
                normals[:, t] = [0.0, 1.0]
            else:
                # Rotate tangent 90° CCW to obtain the left-perpendicular
                normals[:, t] = [-dy / norm, dx / norm]
            origins[:, t] = ref_traj[:2, t]

        self.opti.set_value(self.path_normals_lk, normals)
        self.opti.set_value(self.path_origins_lk, origins)

        # ── Publish corridor-wall markers ─────────────────────────────────
        # Left wall (cyan) and right wall (orange) as LINE_STRIP markers.
        # Each wall is built by offsetting every reference point by ±d_max
        # along the pre-computed unit normal for that step.
        
        now          = self.get_clock().now().to_msg()
        frame        = "ego_racecar/odom"
        d            = self._corridor_d_max

        def _wall_marker(uid, r, g, b, offset_sign):
            m = Marker()
            m.header.frame_id = frame
            m.header.stamp    = now
            m.ns              = "corridor"
            m.id              = uid
            m.type            = Marker.LINE_STRIP
            m.action          = Marker.ADD
            m.scale.x         = 0.04          # line width
            m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, 1.0
            m.pose.orientation.w = 1.0
            m.lifetime = Duration(sec=2)       # auto-expire so stale walls disappear
            for t in range(T + 1):
                pt   = Point()
                pt.x = float(origins[0, t] + offset_sign * d * normals[0, t])
                pt.y = float(origins[1, t] + offset_sign * d * normals[1, t])
                pt.z = 0.05
                m.points.append(pt)
            return m

        self.corridor_pub_left.publish( _wall_marker(200, 0.0, 1.0, 1.0,  1))  # cyan
        self.corridor_pub_right.publish(_wall_marker(201, 1.0, 0.5, 0.0, -1))  # orange

        # ── Rebuild linearised dynamics matrices ─────────────────────────
        A_block, B_block, C_block = [], [], []
        for t in range(self.config.TK):
            delta_bar = self.odelta_v[t] if self.odelta_v is not None else 0.0
            A, B, C = self._get_model_matrix(
                path_predict[2, t], path_predict[3, t], delta_bar
            )
            A_block.append(A)
            B_block.append(B)
            C_block.extend(C)

        A_dense = block_diag(A_block).toarray()
        B_dense = block_diag(B_block).toarray()
        C_dense = np.array(C_block)

        self.opti.set_value(self.Ak_param, A_dense)
        self.opti.set_value(self.Bk_param, B_dense)
        self.opti.set_value(self.Ck_param, C_dense)

        # ── Yaw reference wrapping ────────────────────────────────────────
        ref_traj[3, :] = x0[3] + wrap_angle(ref_traj[3, :] - x0[3])
        for i in range(1, ref_traj.shape[1]):
            d = ref_traj[3, i] - ref_traj[3, i - 1]
            if d >  np.pi: ref_traj[3, i] -= 2 * np.pi
            if d < -np.pi: ref_traj[3, i] += 2 * np.pi
        self.opti.set_value(self.ref_traj_k, ref_traj)

        
        # Pre-initialize outputs with carry-forward values so the return is
        # always valid even if every fallback path fails (avoids NameError).
        oa     = self.oa       if self.oa       is not None else np.zeros(self.config.TK)
        odelta = self.odelta_v if self.odelta_v is not None else np.zeros(self.config.TK)
        ox = oy = oyaw = ov = None

       
        
        NX = self.config.NXK
        NU = self.config.NU
        T  = self.config.TK
        if self._prev_xk is not None and self._prev_uk is not None:
            x_init         = np.zeros((NX, T + 1))
            u_init         = np.zeros((NU, T))
            x_init[:, :-1] = self._prev_xk[:, 1:]
            x_init[:, -1]  = self._prev_xk[:, -1]
            u_init[:, :-1] = self._prev_uk[:, 1:]
            u_init[:, -1]  = self._prev_uk[:, -1]
            self.opti.set_initial(self.xk, x_init)
            self.opti.set_initial(self.uk, u_init)
        else:
            # First call: seed states from the reference trajectory
            self.opti.set_initial(self.xk, ref_traj)
            self.opti.set_initial(self.uk, np.zeros((NU, T)))

        try:
            sol    = self.opti.solve()
            ox     = np.array(sol.value(self.xk[0, :])).flatten()
            oy     = np.array(sol.value(self.xk[1, :])).flatten()
            ov     = np.array(sol.value(self.xk[2, :])).flatten()
            oyaw   = np.array(sol.value(self.xk[3, :])).flatten()
            oa     = np.array(sol.value(self.uk[0, :])).flatten()
            odelta = np.array(sol.value(self.uk[1, :])).flatten()
            self._prev_xk = np.array(sol.value(self.xk))
            self._prev_uk = np.array(sol.value(self.uk))
            print("LMPC status: Solved")
        except Exception as e:
            
            if 'KeyboardInterrupt' in str(e) or 'SystemExit' in str(e):
                raise
            # Solver did not reach full tolerance ; grab the last IPOPT iterate.
            try:
                ox     = np.array(self.opti.debug.value(self.xk[0, :])).flatten()
                oy     = np.array(self.opti.debug.value(self.xk[1, :])).flatten()
                ov     = np.array(self.opti.debug.value(self.xk[2, :])).flatten()
                oyaw   = np.array(self.opti.debug.value(self.xk[3, :])).flatten()
                oa     = np.array(self.opti.debug.value(self.uk[0, :])).flatten()
                odelta = np.array(self.opti.debug.value(self.uk[1, :])).flatten()
                self._prev_xk = np.array(self.opti.debug.value(self.xk))
                self._prev_uk = np.array(self.opti.debug.value(self.uk))
                print(f"LMPC status: best-effort iterate used ({e})")
            except Exception as e2:
                print(f"LMPC: debug fallback failed ({e2}) — holding previous output")
                # oa / odelta already set to carry-forward values above

        return oa, odelta, ox, oy, oyaw, ov

    # ── LINEARIZED — kept for reference ──────────────────────────────────
    def _linear_mpc_control(self, ref_path, x0, oa, od):
        """
        MPC control with updating operational point iteratively
        :param ref_path: reference trajectory in T steps
        :param x0: initial state vector
        :param oa: acceleration of T steps of last time
        :param od: delta of T steps of last time
        """

        if oa is None or od is None:
            oa = [0.0] * self.config.TK
            od = [0.0] * self.config.TK

        # Call the Motion Prediction function: Predict the vehicle motion for x-steps
        path_predict = self._predict_motion(x0, oa, od, ref_path)
        poa, pod = oa[:], od[:]

        # Run the MPC optimization: Create and solve the optimization problem
        mpc_a, mpc_delta, mpc_x, mpc_y, mpc_yaw, mpc_v = self._linear_mpc_prob_solve(
            ref_path, path_predict, x0
        )

        return mpc_a, mpc_delta, mpc_x, mpc_y, mpc_yaw, mpc_v, path_predict

    # ══════════════════════════════════════════════════════════════════════
    # NONLINEAR MPC — full NLP solved directly by IPOPT, no linearization
    # ══════════════════════════════════════════════════════════════════════

    def nmpc_prob_init(self):
        """
        Build the nonlinear MPC (NMPC) problem using CasADi Opti.
        Dynamics constraints are the exact kinematic bicycle model expressed
        symbolically — no A/B/C linearization matrices needed.
        IPOPT solves the resulting NLP directly each control tick.
        """
        NX = self.config.NXK   # 4  (x, y, v, yaw)
        NU = self.config.NU    # 2  (accel, steer)
        T  = self.config.TK    # 4  (horizon steps)
        dt = self.config.DTK
        WB = self.config.WB

        self.opti = ca.Opti()

        # ── Decision variables ────────────────────────────────────────────
        self.xk = self.opti.variable(NX, T + 1)   # predicted states
        self.uk = self.opti.variable(NU, T)        # control inputs

        # ── Parameters (injected each solve call) ─────────────────────────
        self.x0k        = self.opti.parameter(NX)
        self.ref_traj_k = self.opti.parameter(NX, T + 1)

        # ── Cost weight matrices ───────────────────────────────────────────
        R_block  = block_diag([self.config.Rk]  * T).toarray()
        Rd_block = block_diag([self.config.Rdk] * (T - 1)).toarray()
        Q_block  = block_diag([self.config.Qk]  * T + [self.config.Qfk]).toarray()

        # ── Objective ─────────────────────────────────────────────────────
        # 1) Input penalty: u' R u
        u_vec  = ca.vec(self.uk)
        obj    = ca.mtimes([u_vec.T, R_block, u_vec])

        # 2) Tracking penalty: (x - x_ref)' Q (x - x_ref)
        x_err  = ca.vec(self.xk - self.ref_traj_k)
        obj   += ca.mtimes([x_err.T, Q_block, x_err])

        # 3) Input-rate penalty: Δu' Rd Δu
        du     = ca.vec(self.uk[:, 1:] - self.uk[:, :-1])
        obj   += ca.mtimes([du.T, Rd_block, du])

        self.opti.minimize(obj)

        # ── Nonlinear dynamics constraints (exact kinematic bicycle model) ─
        for t in range(T):
            x_t = self.xk[:, t]
            u_t = self.uk[:, t]
            # x_t = [x, y, v, psi],  u_t = [accel, steer_angle]
            x_next = ca.vertcat(
                x_t[0] + x_t[2] * ca.cos(x_t[3]) * dt,          # x
                x_t[1] + x_t[2] * ca.sin(x_t[3]) * dt,          # y
                x_t[2] + u_t[0] * dt,                             # v
                x_t[3] + (x_t[2] / WB) * ca.tan(u_t[1]) * dt,   # yaw
            )
            self.opti.subject_to(self.xk[:, t + 1] == x_next)

        # ── Accel rate limit ───────────────────────────────────────────────
        self.opti.subject_to(
            ca.fabs(self.uk[0, 1:] - self.uk[0, :-1]) <= 5.0
        )

        # ── Steering rate limit ────────────────────────────────────────────
        self.opti.subject_to(
            ca.fabs(self.uk[1, 1:] - self.uk[1, :-1])
            <= self.config.MAX_DSTEER * self.config.DTK
        )

        # ── Initial state pin ──────────────────────────────────────────────
        self.opti.subject_to(self.xk[:, 0] == self.x0k)

        # ── Speed bounds ───────────────────────────────────────────────────
        self.opti.subject_to(self.xk[2, :] >= self.config.MIN_SPEED)
        self.opti.subject_to(self.xk[2, :] <= self.config.MAX_SPEED)

        # ── Input magnitude bounds ─────────────────────────────────────────
        self.opti.subject_to(ca.fabs(self.uk[0, :]) <= self.config.MAX_ACCEL)
        self.opti.subject_to(ca.fabs(self.uk[1, :]) <= self.config.MAX_STEER)

        # ── Solver ────────────────────────────────────────────────────────
        solver_opts = {
            'ipopt.print_level':           0,
            'ipopt.max_iter':              200,
            'ipopt.tol':                   1e-3,
            'ipopt.acceptable_tol':        1e-2,
            'ipopt.acceptable_iter':       5,
            'ipopt.max_cpu_time':          0.15,
            'ipopt.warm_start_init_point': 'yes',
            'print_time':                  0,
        }
        self.opti.solver('ipopt', solver_opts)

    def nmpc_prob_solve(self, ref_traj, x0):
        """
        Solve the NMPC NLP.  No linearization — IPOPT sees the full
        nonlinear model symbolically via CasADi's automatic differentiation.
        Warm-starts from the shifted previous solution each tick.
        """
        self.opti.set_value(self.x0k, x0)

        # ── Yaw reference wrapping ────────────────────────────────────────
        ref_traj[3, :] = x0[3] + wrap_angle(ref_traj[3, :] - x0[3])
        for i in range(1, ref_traj.shape[1]):
            d = ref_traj[3, i] - ref_traj[3, i - 1]
            if d >  np.pi: ref_traj[3, i] -= 2 * np.pi
            if d < -np.pi: ref_traj[3, i] += 2 * np.pi
        self.opti.set_value(self.ref_traj_k, ref_traj)

        # ── Warm-start: shift previous solution by one step ───────────────
        if self._prev_xk is not None and self._prev_uk is not None:
            T  = self.config.TK
            NX = self.config.NXK
            NU = self.config.NU
            x_init         = np.zeros((NX, T + 1))
            u_init         = np.zeros((NU, T))
            x_init[:, :-1] = self._prev_xk[:, 1:]
            x_init[:, -1]  = self._prev_xk[:, -1]
            u_init[:, :-1] = self._prev_uk[:, 1:]
            u_init[:, -1]  = self._prev_uk[:, -1]
            self.opti.set_initial(self.xk, x_init)
            self.opti.set_initial(self.uk, u_init)

        # ── Solve ─────────────────────────────────────────────────────────
        try:
            sol    = self.opti.solve()
            ox     = np.array(sol.value(self.xk[0, :])).flatten()
            oy     = np.array(sol.value(self.xk[1, :])).flatten()
            ov     = np.array(sol.value(self.xk[2, :])).flatten()
            oyaw   = np.array(sol.value(self.xk[3, :])).flatten()
            oa     = np.array(sol.value(self.uk[0, :])).flatten()
            odelta = np.array(sol.value(self.uk[1, :])).flatten()
            self._prev_xk = np.array(sol.value(self.xk))
            self._prev_uk = np.array(sol.value(self.uk))
            print("NMPC status: Solved")
        except Exception as e:
            # On timeout / acceptable-level convergence, recover last iterate
            try:
                ox     = np.array(self.opti.debug.value(self.xk[0, :])).flatten()
                oy     = np.array(self.opti.debug.value(self.xk[1, :])).flatten()
                ov     = np.array(self.opti.debug.value(self.xk[2, :])).flatten()
                oyaw   = np.array(self.opti.debug.value(self.xk[3, :])).flatten()
                oa     = np.array(self.opti.debug.value(self.uk[0, :])).flatten()
                odelta = np.array(self.opti.debug.value(self.uk[1, :])).flatten()
                self._prev_xk = np.array(self.opti.debug.value(self.xk))
                self._prev_uk = np.array(self.opti.debug.value(self.uk))
                print(f"NMPC status: Best-effort solution used ({e})")
            except Exception as e2:
                print(f"Error: Cannot solve nmpc.. {e} | debug failed: {e2}")
                oa, odelta, ox, oy, oyaw, ov = None, None, None, None, None, None

        return oa, odelta, ox, oy, oyaw, ov

    def mpc_control(self, ref_path, x0):
        """
        Nonlinear MPC entry point.
        Solves the full NLP over the horizon — no linearization.
        :param ref_path: reference trajectory (NX x T+1)
        :param x0: current state [x, y, v, yaw]
        """
        mpc_a, mpc_delta, mpc_x, mpc_y, mpc_yaw, mpc_v = self.nmpc_prob_solve(
            ref_path, x0
        )
        return mpc_a, mpc_delta, mpc_x, mpc_y, mpc_yaw, mpc_v

    # ══════════════════════════════════════════════════════════════════════
    # PERFORMANCE LOGGING
    # ══════════════════════════════════════════════════════════════════════

    def log_cte(self, state, label='nMPC'):
        """
        Compute signed cross-track error and immediately write one row to the
        open CSV file.  Flushes on every call so no data is lost on kill.

        Signed CTE: positive = vehicle is to the LEFT of the path direction.
        Change `label` to 'LMPC' when running the linearized version.
        """
        ind  = self.target_ind
        cx   = self.waypoints[ind, 0]
        cy   = self.waypoints[ind, 1]
        cyaw = self.waypoints[ind, 2]

        cte = -(state.x - cx) * math.sin(cyaw) + (state.y - cy) * math.cos(cyaw)

        self._csv_writer.writerow([
            round(time.time() - self._session_start, 4),
            label,
            round(state.x,   4),
            round(state.y,   4),
            round(state.v,   4),
            round(state.yaw, 4),
            round(cte,       4),
            round(abs(cte),  4),
        ])
        self._csv_file.flush()

    def destroy_node(self):
        try:
            self._csv_file.flush()
            self._csv_file.close()
            self.get_logger().info(f'CSV file saved and closed: {self._csv_path}')
        except Exception:
            pass
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    print("MPC Initialized")
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
