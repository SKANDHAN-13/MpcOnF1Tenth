#!/usr/bin/env python3
import math
from dataclasses import dataclass, field
import csv
import cvxpy
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
from scipy.sparse import block_diag, csc_matrix, diags
from sensor_msgs.msg import LaserScan
from utils import nearest_point
from builtin_interfaces.msg import Duration
from visualization_msgs.msg import Marker
from scipy.interpolate import interp1d


# TODO CHECK: include needed ROS msg type headers and libraries


@dataclass
class mpc_config:
    NXK: int = 4  # length of kinematic state vector: z = [x, y, v, yaw]
    NU: int = 2  # length of input vector: u = = [steering speed, acceleration]
    TK: int = 4  # finite time horizon length kinematic

    # ---------------------------------------------------
    Rk: list = field(
    default_factory=lambda: np.diag([
        3.0,        
        5.0,       
    ])
    ) #60*0.05, 0.5 # input cost matrix, penalty for inputs - [steering_speed, accel]
   
    Rdk: list = field(
    default_factory=lambda: np.diag([
        3.0,        
       5,     
    ])
    )  # input difference cost matrix, penalty for change of inputs - [accel, steering_speed]  #60*0.01, 0.5
   
    Qk: list = field(
    default_factory=lambda: np.diag([
        8.0,         # x tracking
        8.0,         # y tracking
        20.0,          # velocity tracking
        20.0,       # yaw tracking 
        
    ])
    )  # state error cost matrix, for the the next (T) prediction time steps [x, y, delta, v, yaw, yaw-rate, beta] #100*13.5, 100*13.5, 2200, 40*13.5
   
    Qfk: list = field(
    default_factory=lambda: np.diag([
       8.0,         # x tracking
       8.0,         # y tracking
       20.0,          # velocity tracking
       20.0,       # yaw tracking 
        #3*(1/2.0**2)          # velocity tracking
    ])
    ) # final state error matrix, penalty  for the final state constraints: [x, y, delta, v, yaw, yaw-rate, beta]#100*13.5, 100*13.5, 2200, 40*13.5
    # ---------------------------------------------------

    N_IND_SEARCH: int = 20  # Search index number
    DTK: float = 0.1  # time step [s] kinematic
    dlk: float = 0.03  # dist step [m] kinematic
    LENGTH: float = 0.58  # Length of the vehicle [m]
    WIDTH: float = 0.31  # Width of the vehicle [m]
    WB: float = 0.33  # Wheelbase [m]
    MIN_STEER: float = -0.4189  # maximum steering angle [rad]
    MAX_STEER: float = 0.4189  # maximum steering angle [rad]
    MAX_DSTEER: float = np.deg2rad(180.0)  # maximum steering speed [rad/s]
    MAX_SPEED: float = 6.0  # maximum speed [m/s]
    MIN_SPEED: float = 0.0  # minimum backward speed [m/s]
    MAX_ACCEL: float = 3.0  # maximum acceleration [m/ss]

"""def calc_speed_profile(cx, cy, cyaw):
    speed = np.zeros(len(cx))

    for i in range(len(cx)-1):
        dyaw = abs(cyaw[i+1] - cyaw[i])
        dyaw = min(dyaw, 2*np.pi - dyaw)

        curvature = dyaw / 0.03   # dlk
        #speed[i] = max(0.8 - 3*curvature, 0.2)/2.0
        speed[i] = 0.5 #np.clip(speed[i], 0.49, 1.01)
        #print("Curvature : ", curvature)
        if curvature>0.1:
            speed[i] = 4
            
        
    #speed[-2] = 0.0
    #speed[-1] = 0.0
    return speed"""

def calc_speed_profile(cx, cy, cyaw):
    ncourse = len(cx)
    sp = np.full(ncourse, 5.0)  # straight speed

    CURVE_THRESHOLD = 0.01   # yaw change per waypoint to count as a curve
    CURVE_SPEED     = 3.0    # speed through the curve
    ENTRY_BOOST_SPEED  = 6.0 # brief speed boost just before braking
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
                # linearly ramp from CURVE_SPEED back to 3.0
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
        
        self.drive_pub = self.create_publisher(AckermannDriveStamped,'/drive',10)
        self.pose_sub = self.create_subscription(Odometry,'/ego_racecar/odom', self.pose_callback,10)
	
        self.waypoints = self.load_waypoints("waypoints.csv")

        self.target_ind = 0  # will be corrected on first callback
        self.target_ind_initialized = False
        self.prev_odom_yaw = None
        self.yaw_offset = 0.0



        self.config = mpc_config()
        self.odelta_v = None
        self.odelta = None
        self.oa = None
        self.init_flag = 0
        self.ref_pub = self.create_publisher(Marker, '/ego_racecar/mpc_ref_traj', 10)
        self.ref_id = 0
        self.ref_timer = self.create_timer(0.5, self.visualize_ref_path)
        self.traj_pub = self.create_publisher(
            Marker,
            '/ego_racecar/driven_traj',
            10
        )

        self.traj_marker = Marker()
        self.traj_marker.header.frame_id = "ego_racecar/odom"
        self.traj_marker.ns = "driven_path"
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




        # initialize MPC problem
        self.mpc_prob_init()

    def visualize_ref_path(self):

        marker = Marker()
        marker.header.frame_id = "ego_racecar/odom"
        marker.header.stamp = self.get_clock().now().to_msg()


        marker.ns = "ref_path"
        marker.id = 0

        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        # BIG DOTS
        marker.scale.x = 0.05
        #marker.scale.y = 0.15

        # BRIGHT RED
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        marker.pose.orientation.w = 1.0
        marker.scale.y = 0.0
        marker.scale.z = 0.0


        marker.lifetime = Duration(sec=0)  # permanent

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
        start = wp[0]
        end   = wp[-1]

        x0, y0, yaw0 = start
        x1, y1, yaw1 = end

        gap = np.hypot(x1-x0, y1-y0)

        if gap > 0.05:   # only close if actually broken

            N_interp = int(gap / 0.03)

            # interpolate ALONG X direction
            f_y   = interp1d([x1, x0], [y1, y0])
            f_yaw = interp1d([x1, x0], [yaw1, yaw0])

            x_new = np.linspace(x1, x0, N_interp)
            y_new = f_y(x_new)
            yaw_new = f_yaw(x_new)

            bridge = np.vstack([x_new, y_new, yaw_new]).T

            wp = np.vstack([wp, bridge])
        wp[:,2] = np.unwrap(wp[:,2])


            #print(f"[MPC] Closed loop by interpolating {N_interp} points")

        return wp

    def pose_callback(self, pose_msg):

    # --- Extract pose ---
        x = pose_msg.pose.pose.position.x
        y = pose_msg.pose.pose.position.y

        q = pose_msg.pose.pose.orientation
        t3 = 2.0 * (q.w * q.z + q.x * q.y)
        t4 = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        #yaw_z = math.atan2(t3, t4)
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

    # IMPORTANT: use keyword args so State fields map correctly
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

    # --- Solve MPC ---
        (self.oa,self.odelta_v,ox,oy,oyaw,ov,_, ) = self.linear_mpc_control(ref_path, x0, self.oa, self.odelta_v)

        if self.oa is None:

            drive = AckermannDriveStamped()
            drive.drive.steering_angle = 0.0
            drive.drive.speed = vehicle_state.v

            self.drive_pub.publish(drive)
            return



        steer_output = float(self.odelta_v[0])
        speed_output = float(vehicle_state.v + self.oa[0] * self.config.DTK)

    # --- Publish ---
        drive = AckermannDriveStamped()
        drive.drive.steering_angle = steer_output
        drive.drive.speed = speed_output
        self.visualize_ref_path()

        self.drive_pub.publish(drive)


    def mpc_prob_init(self):
        """
        Create MPC quadratic optimization problem using cvxpy, solver: OSQP
        Will be solved every iteration for control.
        More MPC problem information here: https://osqp.org/docs/examples/mpc.html
        More QP example in CVXPY here: https://www.cvxpy.org/examples/basic/quadratic_program.html
        """
        # Initialize and create vectors for the optimization problem
        # Vehicle State Vector
        self.xk = cvxpy.Variable(
            (self.config.NXK, self.config.TK + 1)
        )
        # Control Input vector
        self.uk = cvxpy.Variable(
            (self.config.NU, self.config.TK)
        )
        objective = 0.0  # Objective value of the optimization problem
        constraints = []  # Create constraints array

        # Initialize reference vectors
        self.x0k = cvxpy.Parameter((self.config.NXK,))
        self.x0k.value = np.zeros((self.config.NXK,))

        # Initialize reference trajectory parameter
        self.ref_traj_k = cvxpy.Parameter((self.config.NXK, self.config.TK + 1))
        self.ref_traj_k.value = np.zeros((self.config.NXK, self.config.TK + 1))

        # Initializes block diagonal form of R = [R, R, ..., R] (NU*T, NU*T)
        R_block = block_diag(tuple([self.config.Rk] * self.config.TK))

        # Initializes block diagonal form of Rd = [Rd, ..., Rd] (NU*(T-1), NU*(T-1))
        Rd_block = block_diag(tuple([self.config.Rdk] * (self.config.TK - 1)))

        # Initializes block diagonal form of Q = [Q, Q, ..., Qf] (NX*T, NX*T)
        Q_block = [self.config.Qk] * (self.config.TK)
        Q_block.append(self.config.Qfk)
        Q_block = block_diag(tuple(Q_block))

        objective += cvxpy.quad_form(cvxpy.vec(self.uk),R_block)
        objective += cvxpy.quad_form(cvxpy.vec(self.xk - self.ref_traj_k), Q_block)
        objective += cvxpy.quad_form(cvxpy.vec(self.uk[:,1:]-self.uk[:,:-1]), Rd_block)
        # Formulate and create the finite-horizon optimal control problem (objective function)
        # The FTOCP has the horizon of T timesteps

        # --------------------------------------------------------
        # TODO: fill in the objectives here, you should be using cvxpy.quad_form() somehwhere

        # TODO: Objective part 1: Influence of the control inputs: Inputs u multiplied by the penalty R => (self.uk)' @ R @ (self.uk)

        # TODO: Objective part 2: Deviation of the vehicle from the reference trajectory weighted by Q, including final Timestep T weighted by Qf self. => (self.xk-self.ref_traj_k)' @ Q @ (self.xk-self.ref_traj_k)

        # TODO: Objective part 3: Difference from one control input to the next control input weighted by Rd  => Had to check the dimensions of uk

        # --------------------------------------------------------

        # Constraints 1: Calculate the future vehicle behavior/states based on the vehicle dynamics model matrices
        # Evaluate vehicle Dynamics for next T timesteps
        A_block = []
        B_block = []
        C_block = []
        # init path to zeros
        path_predict = np.zeros((self.config.NXK, self.config.TK + 1))
        for t in range(self.config.TK):
            #delta_bar = od[t] if od is not None else 0.0
            A, B, C = self.get_model_matrix(
                path_predict[2, t], path_predict[3, t], 0.0
            )
            A_block.append(A)
            B_block.append(B)
            C_block.extend(C)

        A_block = block_diag(tuple(A_block))
        B_block = block_diag(tuple(B_block))
        C_block = np.array(C_block)

        # [AA] Sparse matrix to CVX parameter for proper stuffing
        # Reference: https://github.com/cvxpy/cvxpy/issues/1159#issuecomment-718925710
        m, n = A_block.shape
        #print("Am = ", m, "An = ", n)
        self.Annz_k = cvxpy.Parameter(A_block.nnz)
        data = np.ones(self.Annz_k.size)
        rows = A_block.row * n + A_block.col
        cols = np.arange(self.Annz_k.size)
        Indexer = csc_matrix((data, (rows, cols)), shape=(m * n, self.Annz_k.size))

        # Setting sparse matrix data
        self.Annz_k.value = A_block.data

        # Now we use this sparse version instead of the old A_ block matrix
        self.Ak_ = cvxpy.reshape(Indexer @ self.Annz_k, (m, n), order="C")

        # Same as A
        m, n = B_block.shape
        #print("Bm = ", m, "Bn = ", n)
        self.Bnnz_k = cvxpy.Parameter(B_block.nnz)
        data = np.ones(self.Bnnz_k.size)
        rows = B_block.row * n + B_block.col
        cols = np.arange(self.Bnnz_k.size)
        Indexer = csc_matrix((data, (rows, cols)), shape=(m * n, self.Bnnz_k.size))
        self.Bk_ = cvxpy.reshape(Indexer @ self.Bnnz_k, (m, n), order="C")
        self.Bnnz_k.value = B_block.data

        # No need for sparse matrices for C as most values are parameters
        self.Ck_ = cvxpy.Parameter(C_block.shape)
        #print("A =", self.Ak_.shape)
        #print("B =", self.Bk_.shape)
        #print("Ck =", self.Ck_.shape)
        #print("uk =", self.uk.shape)
        #print("xK =", self.xk.shape)
        self.Ck_.value = C_block
        #print("C =", C_block.shape)
        
        # Vectorized dynamics over the horizon (avoids 3D indexing into Ak_/Bk_)
        constraints += [cvxpy.vec(self.xk[:, 1:]) 
        == self.Ak_ @ cvxpy.vec(self.xk[:, :-1]) 
        + self.Bk_ @ cvxpy.vec(self.uk) 
        + self.Ck_]

        constraints += [
            cvxpy.abs(self.uk[0,1:] - self.uk[0,:-1]) <= 5.0
        ]

        # steering rate limit
        constraints += [
            cvxpy.abs(self.uk[1,1:] - self.uk[1,:-1]) 
            <= self.config.MAX_DSTEER * self.config.DTK
        ]

# Initial state and state/input bounds
        constraints += [ self.xk[:, 0] == self.x0k ]
        constraints += [ self.xk[2, :] >= self.config.MIN_SPEED, self.xk[2, :] <= self.config.MAX_SPEED ]
        constraints += [ cvxpy.abs(self.uk[0, :]) <= self.config.MAX_ACCEL, cvxpy.abs(self.uk[1, :]) <= self.config.MAX_STEER ]

        # -------------------------------------------------------------
         # TODO: Constraint part 1:
        #       Add dynamics constraints to the optimization problem
        #       This constraint should be based on a few variables:
        #       self.xk, self.Ak_, self.Bk_, self.uk, and self.Ck_  => Linear dynamics constraint
        
        # TODO: Constraint part 2:
        #       Add constraints on steering, change in steering angle
        #       cannot exceed steering angle speed limit. Should be based on:
        #       self.uk, self.config.MAX_DSTEER, self.config.DTK

        # TODO: Constraint part 3:
        #       Add constraints on upper and lower bounds of states and inputs
        #       and initial state constraint, should be based on:
        #       self.xk, self.x0k, self.config.MAX_SPEED, self.config.MIN_SPEED,
        #       self.uk, self.config.MAX_ACCEL, self.config.MAX_STEER
        
        # -------------------------------------------------------------

        # Create the optimization problem in CVXPY and setup the workspace
        # Optimization goal: minimize the objective function
        self.MPC_prob = cvxpy.Problem(cvxpy.Minimize(objective), constraints)

    def calc_ref_trajectory(self, state, cx, cy, cyaw, sp):
        """
        calc referent trajectory ref_traj in T steps: [x, y, v, yaw]
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
    def predict_motion(self, x0, oa, od, xref):

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

    def get_model_matrix(self, v, phi, delta):
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

    def mpc_prob_solve(self, ref_traj, path_predict, x0):
        self.x0k.value = x0

        A_block = []
        B_block = []
        C_block = []
        for t in range(self.config.TK):
            if self.odelta_v is None:
                delta_bar = 0.0
            else:
                delta_bar = self.odelta_v[t]
            A, B, C = self.get_model_matrix(
                path_predict[2, t], path_predict[3, t], delta_bar
            )
            A_block.append(A)
            B_block.append(B)
            C_block.extend(C)

        A_block = block_diag(tuple(A_block))
        B_block = block_diag(tuple(B_block))
        C_block = np.array(C_block)

        self.Annz_k.value = A_block.data
        self.Bnnz_k.value = B_block.data
        self.Ck_.value = C_block

        # in mpc_prob_solve, before setting ref_traj_k.value:
        ref_traj[3, :] = x0[3] + wrap_angle(ref_traj[3, :] - x0[3])
        # then unwrap smoothly across the horizon
        for i in range(1, ref_traj.shape[1]):
            d = ref_traj[3, i] - ref_traj[3, i-1]
            if d > np.pi: ref_traj[3, i] -= 2*np.pi
            if d < -np.pi: ref_traj[3, i] += 2*np.pi
        self.ref_traj_k.value = ref_traj

        #self.ref_traj_k.value = ref_traj

        # Solve the optimization problem in CVXPY
        # Solver selections: cvxpy.OSQP; cvxpy.GUROBI
        self.MPC_prob.solve(solver=cvxpy.OSQP, verbose=False, warm_start=True)
        #self.MPC_prob.solve(solver=cvxpy.OSQP, verbose=True, warm_start=True)
        print("MPC status:", self.MPC_prob.status)


        if (
            self.MPC_prob.status == cvxpy.OPTIMAL
            or self.MPC_prob.status == cvxpy.OPTIMAL_INACCURATE
        ):
            ox = np.array(self.xk.value[0, :]).flatten()
            oy = np.array(self.xk.value[1, :]).flatten()
            ov = np.array(self.xk.value[2, :]).flatten()
            oyaw = np.array(self.xk.value[3, :]).flatten()
            oa = np.array(self.uk.value[0, :]).flatten()
            odelta = np.array(self.uk.value[1, :]).flatten()

        else:
            print("Error: Cannot solve mpc..")
            oa, odelta, ox, oy, oyaw, ov = None, None, None, None, None, None

        return oa, odelta, ox, oy, oyaw, ov

    def linear_mpc_control(self, ref_path, x0, oa, od):
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
        path_predict = self.predict_motion(x0, oa, od, ref_path)
        poa, pod = oa[:], od[:]

        # Run the MPC optimization: Create and solve the optimization problem
        mpc_a, mpc_delta, mpc_x, mpc_y, mpc_yaw, mpc_v = self.mpc_prob_solve(
            ref_path, path_predict, x0
        )

        return mpc_a, mpc_delta, mpc_x, mpc_y, mpc_yaw, mpc_v, path_predict

def main(args=None):
    rclpy.init(args=args)
    print("MPC Initialized")
    mpc_node = MPC()
    rclpy.spin(mpc_node)

    mpc_node.destroy_node()
    rclpy.shutdown()
    
if __name__ == "__main__":
    main()
