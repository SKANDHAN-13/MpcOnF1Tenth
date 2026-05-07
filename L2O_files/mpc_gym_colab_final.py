#!/usr/bin/env python3

import math
import os
import time
import csv
from dataclasses import dataclass, field

import numpy as np
import casadi as ca
from scipy.sparse import block_diag
import matplotlib

matplotlib.use("Agg")                  

import matplotlib.pyplot as plt
import f1tenth_gym                     

# Configurable parameters in the environment
#####################################################################################

@dataclass
class mpc_config:
    NXK: int = 4
    NU:  int = 2
    TK:  int = 20      # 1 seconds lookahead at 0.05s timestep

    #User-defined weights for the MPC cost function (tune these!)
    Rk:  list = field(default_factory=lambda: np.diag([1.2, 6.0]))
    Rdk: list = field(default_factory=lambda: np.diag([2.0, 320.0]))
    Qk:  list = field(default_factory=lambda: np.diag([260, 260, 420, 900]))
    Qfk: list = field(default_factory=lambda: np.diag([1400, 1400, 1400, 4200]))
    #############################################################

    N_IND_SEARCH: int   = 20

    DTK:          float = 0.05
    dlk:          float = 0.03
    WB:           float = 0.33

    #Actuation limits
    MIN_STEER:    float = -0.4189
    MAX_STEER:    float =  0.4189
    MAX_DSTEER:   float = np.deg2rad(180.0)
    MAX_SPEED:    float = 5.0
    MIN_SPEED:    float = 0.0
    MAX_ACCEL:    float = 3.0

    # Corridor constraints - fixed metric separation from raceline on each side.
    # If the track is narrower than CORRIDOR_WIDTH at a waypoint the constraint
    # falls back to the actual wall so the car still stays on track. Currently its the WB
    CORRIDOR_WIDTH: float = 0.33  # metres

   


@dataclass
class State:
    x:    float = 0.0
    y:    float = 0.0
    delta: float = 0.0
    v:    float = 0.0
    yaw:  float = 0.0


def wrap_angle(a):
    return np.arctan2(np.sin(a), np.cos(a))


def calc_speed_profile(cx, cy, max_speed: float = 5.0, max_accel: float = 3.0, ds: float = 0.03):
    
    min_speed = max_speed * 0.25 

    
    x_ext = np.concatenate((cx[-2:], cx, cx[:2]))
    y_ext = np.concatenate((cy[-2:], cy, cy[:2]))
    dx  = np.gradient(x_ext, edge_order=2)
    dy  = np.gradient(y_ext, edge_order=2)
    d2x = np.gradient(dx,    edge_order=2)
    d2y = np.gradient(dy,    edge_order=2)
    denom = (dx**2 + dy**2) ** 1.5
    denom = np.where(denom < 1e-9, 1e-9, denom)
    kappa = np.abs((dx * d2y - d2x * dy) / denom)[2:-2]

    
    kappa_max = np.percentile(kappa, 95)
    if kappa_max < 1e-6:
        return np.full(len(cx), max_speed)
    t  = np.clip(kappa / kappa_max, 0.0, 1.0)
    sp = max_speed - t * (max_speed - min_speed)
    sp = np.clip(sp, min_speed, max_speed)

    
    for i in range(len(sp) - 2, -1, -1):
        v_can_reach = math.sqrt(sp[(i + 1) % len(sp)] ** 2 + 2.0 * max_accel * ds)
        sp[i] = min(sp[i], v_can_reach)

    
    for i in range(1, len(sp)):
        v_can_reach = math.sqrt(sp[i - 1] ** 2 + 2.0 * max_accel * ds)
        sp[i] = min(sp[i], v_can_reach)

    
    sp = np.clip(sp, min_speed, max_speed)
    return sp


# Headless MPC  (All ROS stripped)
######################################################################################

class HeadlessMPC:
    def __init__(self, waypoints_csv: str, border_coeffs_csv: str = None,
                 max_speed: float = None):
        self.config  = mpc_config()
        if max_speed is not None:
            self.config.MAX_SPEED = max_speed  # must be set before _linear_mpc_prob_init bakes it
        self.oa      = None
        self.odelta_v = None
        self.odelta   = None
        self._prev_xk = None
        self._prev_uk = None
        self.target_ind = 0
        self.target_ind_initialized = False
        self.prev_odom_yaw = None
        self.yaw_offset    = 0.0
        self._last_ind_list = None

        self.waypoints = self._load_waypoints(waypoints_csv)         # An array of {x, y, yaw} from the waypoints.csv file

        # Active mode flag
        self._active_mode = 'LMPC'   # 'LMPC' or 'NMPC'
        # Separate warm-start cache for each solver
        self._prev_xk_nl = None
        self._prev_uk_nl = None

        if border_coeffs_csv is not None:            
            self.border_coeffs = self._load_border_coeffs(border_coeffs_csv)  # An array of {w_r, w_l,a, b, tight_cl, tight_cr} is loaded to use as constraints on the border of the race-track.
            self._has_border = True
            print(f" Border constraints loaded: {self.border_coeffs.shape[0]} rows ")
            # Replace raceline x/y/yaw with geometric track-center so MPC tracks
            # the midpoint between the two walls instead of the optimised racing line.
            _cx = self.track_center_xy[:, 0]
            _cy = self.track_center_xy[:, 1]
            self.waypoints[:, 0] = _cx
            self.waypoints[:, 1] = _cy
            # Recompute yaw from the centerline tangent so ref_yaw matches the
            # actual direction of travel along the center
            _dx = np.gradient(_cx)
            _dy = np.gradient(_cy)
            self.waypoints[:, 2] = np.unwrap(np.arctan2(_dy, _dx))
            print(" Waypoints (x, y, yaw) replaced with track-center coordinates.")
       
        else:
            self.border_coeffs = None
            self._has_border = False

        self._linear_mpc_prob_init()
        self._nonlinear_mpc_prob_init()

    ##############################################################################

    def _load_waypoints(self, path: str) -> np.ndarray:
        data = []

        with open(path, 'r') as f:
            reader = csv.reader(f)
            next(reader)                                      # Skip header of the file
            
            for row in reader:
                x, y, yaw = map(float, row)
                data.append([x, y, yaw])
        
        wp = np.array(data)
        wp[:, 2] = np.unwrap(wp[:, 2])
        x0, y0, _ = wp[0]
        x1, y1, _ = wp[-1]
        gap = np.hypot(x1 - x0, y1 - y0)

         # If the gap between the last and first waypoint is large,
         # we interpolate to create a smooth loop.
        if gap > 0.05:
            N_interp = max(int(gap / 0.03), 2)
            t = np.linspace(0.0, 1.0, N_interp + 2)[1:-1]
            bridge = np.vstack([x1 + t*(x0-x1), y1 + t*(y0-y1), np.zeros(len(t))]).T
            wp = np.vstack([wp, bridge])

        diffs = np.diff(wp[:, :2], axis=0)

        s = np.concatenate([[0.0], np.cumsum(np.hypot(diffs[:,0], diffs[:,1]))]) #Cumulative distance along the waypoints
        s_new  = np.arange(0.0, s[-1], 0.03)

        # New waypoints interpolated at regular 0.03m intervals along the track + unwrapped yaw angles
        wp_x   = np.interp(s_new, s, wp[:, 0])
        wp_y   = np.interp(s_new, s, wp[:, 1])
        wp_yaw = np.interp(s_new, s, wp[:, 2])

        return np.vstack([wp_x, wp_y, wp_yaw]).T

    ####################################################################################

    def _load_border_coeffs(self, path: str) -> np.ndarray:
        data = []

        with open(path, 'r') as f:
            reader = csv.DictReader(f)   #DictReader() is used to read the CSV file as a dictionary; accessing by column names.
            for row in reader:
                data.append([
                    float(row['a']),
                    float(row['b']),
                    float(row['c_center']),
                    float(row['c_left']),
                    float(row['c_right']),
                ])
        
        bc = np.array(data)  
        
        # Since new waypoints are interpolated at regular 0.03m intervals,
        # we use linear interpolation for the border coefficients to match the new waypoints.
        N_orig   = len(bc)
        N_wp     = len(self.waypoints)
        idx_orig = np.arange(N_orig, dtype=float)
        idx_new  = np.linspace(0.0, N_orig - 1, N_wp)

        bc_rs = np.column_stack([
            np.interp(idx_new, idx_orig, bc[:, i]) for i in range(5)
        ])                   
        a_rs, b_rs, cc_rs, cl_rs, cr_rs = bc_rs.T
        # Resampled border coefficients are now aligned with the waypoints

        wp_x = self.waypoints[:, 0]
        wp_y = self.waypoints[:, 1]

        # Track wall XY - Offset raceline by (c_wall - c_center) × normal
        self.border_walls_xy = np.column_stack([
            wp_x + a_rs * (cl_rs - cc_rs),   # Left  wall x
            wp_y + b_rs * (cl_rs - cc_rs),   # Left  wall y
            wp_x + a_rs * (cr_rs - cc_rs),   # Right wall x
            wp_y + b_rs * (cr_rs - cc_rs),   # Right wall y
        ])

        # Geometric track center: midpoint between left and right walls
        center_c = (cl_rs + cr_rs) / 2.0
        self.track_center_xy = np.column_stack([
            wp_x + a_rs * (center_c - cc_rs),   # Center x
            wp_y + b_rs * (center_c - cc_rs),   # Center y
        ])

        # Tight corridor bounds: inset CORRIDOR_WIDTH metres from each wall inward.
        # This uses the full free space between both walls, minus a safety buffer w.
        # When the track is narrower than 2w the limits are clamped to cc_rs so
        # the constraint always stays feasible and the car tracks the raceline.
        w        = self.config.CORRIDOR_WIDTH
        tight_cl = np.maximum(cl_rs - w, cc_rs)   # left  limit  = left wall  − w, ≥ raceline
        tight_cr = np.minimum(cr_rs + w, cc_rs)   # right limit  = right wall + w, ≤ raceline

        self.border_corridor_xy = np.column_stack([
            wp_x + a_rs * (tight_cl - cc_rs),   # Left  corridor x
            wp_y + b_rs * (tight_cl - cc_rs),   # Left  corridor y
            wp_x + a_rs * (tight_cr - cc_rs),   # Right corridor x
            wp_y + b_rs * (tight_cr - cc_rs),   # Right corridor y
        ])

        return np.column_stack([a_rs, b_rs, tight_cl, tight_cr])   

    ##################################################################################

    def _linear_mpc_prob_init(self):
        NX = self.config.NXK
        NU = self.config.NU
        T  = self.config.TK

        self.opti = ca.Opti() # Optimization problem instance

        self.xk = self.opti.variable(NX, T + 1) # State trajectory over the horizon 
        self.uk = self.opti.variable(NU, T)     # Control trajectory over the horizon

        self.x0k        = self.opti.parameter(NX)         # Initial state parameter
        self.ref_traj_k = self.opti.parameter(NX, T + 1)  # Reference trajectory parameter

        ##############################################
        self.Ak_param   = self.opti.parameter(NX * T, NX * T) 
        self.Bk_param   = self.opti.parameter(NX * T, NU * T)
        self.Ck_param   = self.opti.parameter(NX * T)

        R_block  = block_diag([self.config.Rk]  * T).toarray()
        Rd_block = block_diag([self.config.Rdk] * (T - 1)).toarray()
        Q_block  = block_diag([self.config.Qk]  * T + [self.config.Qfk]).toarray()
        ################################################
        # Cost function evaluation
        u_vec = ca.vec(self.uk)              # Control inputs flattened into a vector 
        x_err = ca.vec(self.xk - self.ref_traj_k)  # State error flattened into a vector
        du    = ca.vec(self.uk[:, 1:] - self.uk[:, :-1])  # Changes in controlled input flattened into a vector

        obj   = (ca.mtimes([u_vec.T, R_block, u_vec])
               + ca.mtimes([x_err.T, Q_block, x_err])
               + ca.mtimes([du.T, Rd_block, du]))
        self._lmpc_obj = obj   # Keep reference for cost logging
        self.opti.minimize(obj)

        # Linearized dynamics constraints
        x_next = ca.vec(self.xk[:, 1:])
        x_curr = ca.vec(self.xk[:, :-1])
        u_flat = ca.vec(self.uk)
        self.opti.subject_to(
            x_next == ca.mtimes(self.Ak_param, x_curr)
                    + ca.mtimes(self.Bk_param, u_flat)
                    + self.Ck_param
        )

        # Actuation and state constraints
        #self.opti.subject_to(ca.fabs(self.uk[0, 1:] - self.uk[0, :-1]) <= 5.0)
        self.opti.subject_to(
            ca.fabs(self.uk[1, 1:] - self.uk[1, :-1])
            <= self.config.MAX_DSTEER * self.config.DTK
        )
        self.opti.subject_to(self.xk[:, 0] == self.x0k)
        self.max_speed_param = self.opti.parameter(1)
        self.opti.subject_to(self.xk[2, :] >= self.config.MIN_SPEED)
        self.opti.subject_to(self.xk[2, :] <= self.max_speed_param)
        self.opti.subject_to(ca.fabs(self.uk[0, :]) <= self.config.MAX_ACCEL)
        self.opti.subject_to(ca.fabs(self.uk[1, :]) <= self.config.MAX_STEER)

        # Hard corridor wall constraints
        if self._has_border:
            # Per horizon step
            self.border_a_k  = self.opti.parameter(T + 1)   # normal-x component
            self.border_b_k  = self.opti.parameter(T + 1)   # normal-y component
            self.border_cl_k = self.opti.parameter(T + 1)   # left  limit (outer)
            self.border_cr_k = self.opti.parameter(T + 1)   # right limit (inner)
            
            # Project each point onto the normal vector defined by (a, b)
            # and constrain it to be between the "fractioned" left and right limits.
            proj = (self.border_a_k * ca.vec(self.xk[0, :])
                  + self.border_b_k * ca.vec(self.xk[1, :]))
            self.opti.subject_to(proj <= self.border_cl_k)
            self.opti.subject_to(proj >= self.border_cr_k)

        self.opti.solver('ipopt', {
            'ipopt.print_level': 0, 'print_time': 0,
            'ipopt.max_iter': 200,    
            'ipopt.tol': 1e-3,
            'ipopt.acceptable_tol': 5e-2,
            'ipopt.acceptable_iter': 5,
            'ipopt.warm_start_init_point': 'yes',
            'ipopt.warm_start_bound_push': 1e-6,
            'ipopt.max_cpu_time': 0.15,
        })

    #####################################################################
    def _get_model_matrix(self, v, phi, delta):
        # Linearized kinematic bicycle model
        A = np.eye(self.config.NXK)
        A[0, 2] =  self.config.DTK * math.cos(phi)
        A[0, 3] = -self.config.DTK * v * math.sin(phi)
        A[1, 2] =  self.config.DTK * math.sin(phi)
        A[1, 3] =  self.config.DTK * v * math.cos(phi)
        A[3, 2] =  self.config.DTK * math.tan(delta) / self.config.WB

        B = np.zeros((self.config.NXK, self.config.NU))
        B[2, 0] = self.config.DTK
        B[3, 1] = self.config.DTK * v / (self.config.WB * math.cos(delta) ** 2)

        C = np.zeros(self.config.NXK)
        C[0] =  self.config.DTK * v * math.sin(phi) * phi
        C[1] = -self.config.DTK * v * math.cos(phi) * phi
        C[3] = -self.config.DTK * v * delta / (self.config.WB * math.cos(delta) ** 2)
        return A, B, C

    #####################################################################
    def update_state(self, state, a, delta):
        # delta : steering angle in radians
        # a     : acceleration in m/s^2 ; WB : wheelbase in m
        # v/WB * tan(delta) : yaw rate in radians/s
        delta = np.clip(delta, -self.config.MAX_STEER, self.config.MAX_STEER)

        state.x   += state.v * math.cos(state.yaw) * self.config.DTK
        state.y   += state.v * math.sin(state.yaw) * self.config.DTK

        state.yaw += (state.v / self.config.WB) * math.tan(delta) * self.config.DTK
        state.v    = np.clip(state.v + a * self.config.DTK,
                             self.config.MIN_SPEED, self.config.MAX_SPEED)
        return state

    #####################################################################
    def _predict_motion(self, x0, oa, od, xref):
        path_predict = xref * 0.0

        for i in range(len(x0)):
            path_predict[i, 0] = x0[i]

        state = State(x=x0[0], y=x0[1], v=x0[2], yaw=x0[3])
        for i in range(1, self.config.TK + 1):
            state = self.update_state(state, oa[i-1], od[i-1])
            path_predict[0, i] = state.x
            path_predict[1, i] = state.y
            path_predict[2, i] = state.v
            path_predict[3, i] = state.yaw
        return path_predict

    #####################################################################
    def calc_ref_trajectory(self, state, cx, cy, cyaw, sp):

        ref_traj = np.zeros((self.config.NXK, self.config.TK + 1))
        ncourse  = len(cx)

        if not self.target_ind_initialized:
            dists = (cx - state.x)**2 + (cy - state.y)**2
            self.target_ind = int(np.argmin(dists))
            self.target_ind_initialized = True
        # search_indices : indices of the waypoints to consider for the nearest point search
        # centered around the current target_ind and wrapping around the track if necessary.
        
        # local_best : index of the closest waypoint among the search_indices, to update target_ind for better tracking 
        search_indices = [(self.target_ind + i) % ncourse
                          for i in range(self.config.N_IND_SEARCH)]
        dx  = np.array([cx[i] for i in search_indices]) - state.x
        dy  = np.array([cy[i] for i in search_indices]) - state.y
        local_best = np.argmin(dx**2 + dy**2)


        if local_best > 0:
            self.target_ind = search_indices[local_best]
        ind   = self.target_ind

        
        travel = max(state.v, self.config.MAX_SPEED * 0.5) * self.config.DTK
        dind   = travel / self.config.dlk
        ind_list = (int(ind) + np.insert(
            np.cumsum(np.repeat(dind, self.config.TK)), 0, 0
        ).astype(int)) % ncourse
        ref_traj[0, :] = cx[ind_list]
        ref_traj[1, :] = cy[ind_list]
        ref_traj[2, :] = sp[ind_list]
        for i, idx in enumerate(ind_list):
            ref_traj[3, i] = cyaw[idx]
        self._last_ind_list = ind_list   # store for border-constraint lookup
        return ref_traj

    ################################################################
    def _linear_mpc_prob_solve(self, ref_traj, path_predict, x0):
        self.opti.set_value(self.max_speed_param, self.config.MAX_SPEED)
        self.opti.set_value(self.x0k, x0)
        NX, NU, T = self.config.NXK, self.config.NU, self.config.TK
        A_bd = np.zeros((NX * T, NX * T))
        B_bd = np.zeros((NX * T, NU * T))
        C_bd = np.zeros(NX * T)
        for t in range(T):
            delta_bar = self.odelta_v[t] if self.odelta_v is not None else 0.0
            A, B, C = self._get_model_matrix(
                path_predict[2, t], path_predict[3, t], delta_bar)
            A_bd[t*NX:(t+1)*NX, t*NX:(t+1)*NX] = A
            B_bd[t*NX:(t+1)*NX, t*NU:(t+1)*NU] = B
            C_bd[t*NX:(t+1)*NX] = C
        self.opti.set_value(self.Ak_param, A_bd)
        self.opti.set_value(self.Bk_param, B_bd)
        self.opti.set_value(self.Ck_param, C_bd)

        #Border constraints lookup : last target_ind and the pre-loaded border coefficients.
        if self._has_border and self._last_ind_list is not None:
            bc = self.border_coeffs[self._last_ind_list]   # (T+1, 4)
            self.opti.set_value(self.border_a_k,  bc[:, 0])
            self.opti.set_value(self.border_b_k,  bc[:, 1])
            self.opti.set_value(self.border_cl_k, bc[:, 2])
            self.opti.set_value(self.border_cr_k, bc[:, 3])

        
        ref_traj[3, 0] = x0[3] + wrap_angle(ref_traj[3, 0] - x0[3])
        for i in range(1, ref_traj.shape[1]):
            d = ref_traj[3, i] - ref_traj[3, i - 1]
            if d >  np.pi: ref_traj[3, i] -= 2*np.pi
            if d < -np.pi: ref_traj[3, i] += 2*np.pi
        self.opti.set_value(self.ref_traj_k, ref_traj)

        oa     = self.oa       if self.oa       is not None else np.zeros(self.config.TK)
        odelta = self.odelta_v if self.odelta_v is not None else np.zeros(self.config.TK)

        NX, NU, T = self.config.NXK, self.config.NU, self.config.TK
        if self._prev_xk is not None and self._prev_uk is not None:
            x_init = np.zeros((NX, T+1)); u_init = np.zeros((NU, T))
            x_init[:, :-1] = self._prev_xk[:, 1:]; x_init[:, -1] = self._prev_xk[:, -1]
            u_init[:, :-1] = self._prev_uk[:, 1:]; u_init[:, -1] = self._prev_uk[:, -1]
            self.opti.set_initial(self.xk, x_init)
            self.opti.set_initial(self.uk, u_init)
        else:
            self.opti.set_initial(self.xk, ref_traj)
            self.opti.set_initial(self.uk, np.zeros((NU, T)))

        cost = float('nan')
        try:
            sol    = self.opti.solve()
            oa     = np.array(sol.value(self.uk[0, :])).flatten()
            odelta = np.array(sol.value(self.uk[1, :])).flatten()
            self._prev_xk = np.array(sol.value(self.xk))
            self._prev_uk = np.array(sol.value(self.uk))
            cost   = float(sol.value(self._lmpc_obj))
        except Exception as e:
            try:
                oa     = np.array(self.opti.debug.value(self.uk[0, :])).flatten()
                odelta = np.array(self.opti.debug.value(self.uk[1, :])).flatten()
                self._prev_xk = np.array(self.opti.debug.value(self.xk))
                self._prev_uk = np.array(self.opti.debug.value(self.uk))
                cost   = float(self.opti.debug.value(self._lmpc_obj))
            except Exception:
                pass   # carry-forward oa/odelta already set above
        return oa, odelta, cost

    ##########################################################################
    def _linear_mpc_control(self, ref_path, x0):
        oa = self.oa       if self.oa       is not None else [0.0]*self.config.TK
        od = self.odelta_v if self.odelta_v is not None else [0.0]*self.config.TK

        # Use the previous optimal trajectory as the linearization point when
        # available, it stays on the reference path and gives IPOPT an accurate local model, preventing corner-cut exploitation of linearization error.
        
        if self._prev_xk is not None and self._prev_uk is not None:
            # Shift the cached solution one step forward as the new operating point
            NX, NU, T = self.config.NXK, self.config.NU, self.config.TK
            path_predict = np.zeros((NX, T + 1))
            path_predict[:, :-1] = self._prev_xk[:, 1:]
            path_predict[:, -1]  = self._prev_xk[:, -1]
            # Keep x0 anchored to current state
            path_predict[:, 0] = x0
        else:
            x0_pred = list(x0)
            if abs(x0_pred[2]) < 0.5:
                x0_pred[2] = float(ref_path[2, 1])   # seed ref velocity at stand-still
            path_predict = self._predict_motion(x0_pred, oa, od, ref_path)

        return self._linear_mpc_prob_solve(ref_path, path_predict, x0)  # (oa, odelta, cost)

    ##########################################################################
    def _nonlinear_mpc_prob_init(self):
       
        NX = self.config.NXK
        NU = self.config.NU
        T  = self.config.TK

        self.opti_nl = ca.Opti()

        self.xk_nl        = self.opti_nl.variable(NX, T + 1)
        self.uk_nl        = self.opti_nl.variable(NU, T)
        self.x0k_nl       = self.opti_nl.parameter(NX)
        self.ref_traj_nl  = self.opti_nl.parameter(NX, T + 1)

        R_block  = block_diag([self.config.Rk]  * T).toarray()
        Rd_block = block_diag([self.config.Rdk] * (T - 1)).toarray()
        Q_block  = block_diag([self.config.Qk]  * T + [self.config.Qfk]).toarray()

        u_vec = ca.vec(self.uk_nl)
        x_err = ca.vec(self.xk_nl - self.ref_traj_nl)
        du    = ca.vec(self.uk_nl[:, 1:] - self.uk_nl[:, :-1])
        obj   = (ca.mtimes([u_vec.T, R_block, u_vec])
               + ca.mtimes([x_err.T, Q_block, x_err])
               + ca.mtimes([du.T,    Rd_block, du]))
        self._nmpc_obj = obj   # Keep reference for cost logging
        self.opti_nl.minimize(obj)

        # Nonlinear kinematic bicycle dynamics (Euler, DTK step)
        dt = self.config.DTK
        WB = self.config.WB
        for t in range(T):
            x_t   = self.xk_nl[:, t]
            x_tp1 = self.xk_nl[:, t + 1]
            u_t   = self.uk_nl[:, t]
            # [x, y, v, yaw]
            x_next = ca.vertcat(
                x_t[0] + x_t[2] * ca.cos(x_t[3]) * dt,
                x_t[1] + x_t[2] * ca.sin(x_t[3]) * dt,
                x_t[2] + u_t[0] * dt,
                x_t[3] + (x_t[2] / WB) * ca.tan(u_t[1]) * dt,
            )
            self.opti_nl.subject_to(x_tp1 == x_next)

        # Initial state
        self.opti_nl.subject_to(self.xk_nl[:, 0] == self.x0k_nl)

        # Actuation + state box constraints
        #self.opti_nl.subject_to(
        #    ca.fabs(self.uk_nl[0, 1:] - self.uk_nl[0, :-1]) <= 5.0)
        self.opti_nl.subject_to(
            ca.fabs(self.uk_nl[1, 1:] - self.uk_nl[1, :-1])
            <= self.config.MAX_DSTEER * self.config.DTK)
        self.max_speed_param_nl = self.opti_nl.parameter(1)
        self.opti_nl.subject_to(self.xk_nl[2, :] >= self.config.MIN_SPEED)
        self.opti_nl.subject_to(self.xk_nl[2, :] <= self.max_speed_param_nl)
        self.opti_nl.subject_to(ca.fabs(self.uk_nl[0, :]) <= self.config.MAX_ACCEL)
        self.opti_nl.subject_to(ca.fabs(self.uk_nl[1, :]) <= self.config.MAX_STEER)

        # Corridor wall constraints (same as LMPC)
        if self._has_border:
            self.border_a_k_nl  = self.opti_nl.parameter(T + 1)
            self.border_b_k_nl  = self.opti_nl.parameter(T + 1)
            self.border_cl_k_nl = self.opti_nl.parameter(T + 1)
            self.border_cr_k_nl = self.opti_nl.parameter(T + 1)
            proj_nl = (self.border_a_k_nl * ca.vec(self.xk_nl[0, :])
                     + self.border_b_k_nl * ca.vec(self.xk_nl[1, :]))
            self.opti_nl.subject_to(proj_nl <= self.border_cl_k_nl)
            self.opti_nl.subject_to(proj_nl >= self.border_cr_k_nl)

        self.opti_nl.solver('ipopt', {
            'ipopt.print_level': 0, 'print_time': 0,
            'ipopt.max_iter': 200,
            'ipopt.tol': 1e-3,
            'ipopt.acceptable_tol': 1e-2,
            'ipopt.acceptable_iter': 5,
            'ipopt.max_cpu_time': 0.15,
            'ipopt.warm_start_init_point': 'yes',
            'ipopt.warm_start_bound_push': 1e-6,
            'ipopt.warm_start_mult_bound_push': 1e-6,
        })

    ##########################################################################
    def _nonlinear_mpc_prob_solve(self, ref_traj, x0):
        self.opti_nl.set_value(self.max_speed_param_nl, self.config.MAX_SPEED)
        self.opti_nl.set_value(self.x0k_nl, x0)

        ref_traj = ref_traj.copy()
        ref_traj[3, 0] = x0[3] + wrap_angle(ref_traj[3, 0] - x0[3])
        for i in range(1, ref_traj.shape[1]):
            d = ref_traj[3, i] - ref_traj[3, i - 1]
            if d >  np.pi: ref_traj[3, i] -= 2 * np.pi
            if d < -np.pi: ref_traj[3, i] += 2 * np.pi
        self.opti_nl.set_value(self.ref_traj_nl, ref_traj)

        if self._has_border and self._last_ind_list is not None:
            bc = self.border_coeffs[self._last_ind_list]   # (T+1, 4)
            self.opti_nl.set_value(self.border_a_k_nl,  bc[:, 0])
            self.opti_nl.set_value(self.border_b_k_nl,  bc[:, 1])
            self.opti_nl.set_value(self.border_cl_k_nl, bc[:, 2])
            self.opti_nl.set_value(self.border_cr_k_nl, bc[:, 3])

        NX, NU, T = self.config.NXK, self.config.NU, self.config.TK

        # Warm start: shift previous NMPC solution; seed from reference on first call
        if self._prev_xk_nl is not None and self._prev_uk_nl is not None:
            x_init = np.zeros((NX, T + 1)); u_init = np.zeros((NU, T))
            x_init[:, :-1] = self._prev_xk_nl[:, 1:]
            x_init[:, -1]  = self._prev_xk_nl[:, -1]
            u_init[:, :-1] = self._prev_uk_nl[:, 1:]
            u_init[:, -1]  = self._prev_uk_nl[:, -1]
        else:
            x_init = ref_traj
            u_init = np.zeros((NU, T))
        self.opti_nl.set_initial(self.xk_nl, x_init)
        self.opti_nl.set_initial(self.uk_nl, u_init)

        oa     = self.oa       if self.oa       is not None else np.zeros(T)
        odelta = self.odelta_v if self.odelta_v is not None else np.zeros(T)
        cost   = float('nan')
        try:
            sol    = self.opti_nl.solve()
            oa     = np.array(sol.value(self.uk_nl[0, :])).flatten()
            odelta = np.array(sol.value(self.uk_nl[1, :])).flatten()
            self._prev_xk_nl = np.array(sol.value(self.xk_nl))
            self._prev_uk_nl = np.array(sol.value(self.uk_nl))
            cost   = float(sol.value(self._nmpc_obj))
        except Exception:
            try:
                oa     = np.array(self.opti_nl.debug.value(self.uk_nl[0, :])).flatten()
                odelta = np.array(self.opti_nl.debug.value(self.uk_nl[1, :])).flatten()
                self._prev_xk_nl = np.array(self.opti_nl.debug.value(self.xk_nl))
                self._prev_uk_nl = np.array(self.opti_nl.debug.value(self.uk_nl))
                cost   = float(self.opti_nl.debug.value(self._nmpc_obj))
            except Exception:
                pass
        return oa, odelta, cost

    ##########################################################################
    def _nonlinear_mpc_control(self, ref_path, x0):
        return self._nonlinear_mpc_prob_solve(ref_path, x0)  # (oa, odelta, cost)

    ##########################################################################
    def mpc_control(self, ref_path, x0):
        mode = getattr(self, '_fixed_mode', None) or 'LMPC'

        if mode == 'NMPC':
            oa, odelta, cost = self._nonlinear_mpc_control(ref_path, x0)
            self._active_mode = 'NMPC'
        else:
            oa, odelta, cost = self._linear_mpc_control(ref_path, x0)
            self._active_mode = 'LMPC'

        return oa, odelta, self._active_mode, cost

    #########################################################################
    def update_yaw(self, yaw_raw: float) -> float:
        if self.prev_odom_yaw is None:
            self.prev_odom_yaw = yaw_raw
        dyaw = yaw_raw - self.prev_odom_yaw
        if dyaw >  np.pi: self.yaw_offset -= 2*np.pi
        if dyaw < -np.pi: self.yaw_offset += 2*np.pi
        self.prev_odom_yaw = yaw_raw
        return yaw_raw + self.yaw_offset

# Gym runner
#########################################################################################

class GymMPCRunner:
    
    def __init__(self, waypoints_csv: str, map_name: str = "Spielberg",
                 border_coeffs_csv: str = None, max_speed: float = None):
        self.mpc      = HeadlessMPC(waypoints_csv, border_coeffs_csv=border_coeffs_csv,
                                    max_speed=max_speed)
        self.map_name = map_name
        self._log: list    = []   # list of dicts, one per control step
        self._frames: list = []   # annotated RGB frames
        
        # mutable state shared between run-loop and render callbacks
        self._driven_state = {'renderer': None, 'points': np.zeros((1, 2), dtype=np.float32)}
        self._pred_state   = {'renderer': None, 'points': np.zeros((1, 2), dtype=np.float32)}

    def run(self,
            max_sim_seconds: float = 60.0):

        import gymnasium as gym
        from f1tenth_gym.envs.observation import ObservationType
        from f1tenth_gym.envs.dynamic_models import DynamicModel
        from f1tenth_gym.envs.integrators import IntegratorType
        from f1tenth_gym.envs.reset import ResetStrategy
        from f1tenth_gym.envs.env_config import (
            ControlConfig, EnvConfig, ObservationConfig, ResetConfig, SimulationConfig,
        )

        env_cfg = EnvConfig(
            map_name=self.map_name,
            observation_config=ObservationConfig(type=ObservationType.KINEMATIC_STATE),
            simulation_config=SimulationConfig(
                timestep=0.01,
                integrator_timestep=0.01,
                integrator=IntegratorType.RK4,
                dynamics_model=DynamicModel.KS,
                compute_frenet_frame=False,
                max_laps=None,
            ),
            control_config=ControlConfig(steer_delay_steps=1),
            reset_config=ResetConfig(strategy=ResetStrategy.RL_RANDOM_STATIC),
        )
        env = gym.make(
            'f1tenth_gym:f1tenth-v0',
            config=env_cfg,
            render_mode="rgb_array",
        )

        # Spawn at waypoints[0]: this is the track-center coordinate when
        # border_coeffs are loaded (waypoints replaced in __init__), otherwise
        # the raceline start.  Wrap yaw to [-pi, pi].
        _spawn_x   = float(self.mpc.waypoints[0, 0])
        _spawn_y   = float(self.mpc.waypoints[0, 1])
        _spawn_yaw = float(wrap_angle(self.mpc.waypoints[0, 2]))
        print(f"Spawn pose → x={_spawn_x:.3f}  y={_spawn_y:.3f}  "
              f"yaw={math.degrees(_spawn_yaw):.1f}°")
        poses = np.array([[_spawn_x, _spawn_y, _spawn_yaw]])
        obs, info = env.reset(options={"poses": poses})


        # Reset per-run overlay state and register render callbacks
        self._driven_state = {'renderer': None, 'points': np.zeros((1, 2), dtype=np.float32)}
        self._pred_state   = {'renderer': None, 'points': np.zeros((1, 2), dtype=np.float32)}
        self._driven_pts_list: list = []
        self._setup_render_callbacks(env)

        cfg      = self.mpc.config
        ref_x    = self.mpc.waypoints[:, 0]
        ref_y    = self.mpc.waypoints[:, 1]
        ref_yaw  = self.mpc.waypoints[:, 2]
        ref_v    = calc_speed_profile(ref_x, ref_y, cfg.MAX_SPEED,
                                        max_accel=cfg.MAX_ACCEL, ds=cfg.dlk)

      
        GYM_DT        = 0.01
        MPC_INTERVAL  = max(1, int(cfg.DTK / GYM_DT))
        steer         = 0.0

        # speed_desired is the integrated velocity target sent to the gym.
        # Seeded from the reference profile so the car moves before the first
        # successful IPOPT solve; thereafter updated by MPC acceleration output.

        speed_desired = 0.0   # start from rest; ramp up via ref-profile rate limiter
        ref_path      = None           # persisted across MPC intervals
        sim_time      = 0.0
        step_count    = 0
        t_wall_start   = time.time()
       
        RENDER_EVERY   = max(1, int(1.0 / GYM_DT))   # 1 frame / sim-second
        self._frames   = []
        self._log      = []

        pred_horizon_x: list = []   # MPC horizon x
        pred_horizon_y: list = []   # MPC horizon y
        _cost: float = float('nan') # latest solver objective value

        # Live per-second snapshot display
        try:
            from IPython import get_ipython as _gip
            _IN_COLAB = _gip() is not None
        except Exception:
            _IN_COLAB = False
        _last_preview_sec = -1

        DEBUG_EVERY = max(1, int(1.0 / GYM_DT))   # print once per sim-second
        print(f"Running MPC in gym for up to {max_sim_seconds:.0f}s sim-time ...")
        print(f"{'step':>7}  {'sim_t':>7}  {'x':>8}  {'y':>8}  {'v_obs':>7}  {'v_cmd':>7}  {'steer_deg':>10}  {'CTE':>8}  {'cost':>14}")

        try:
            while sim_time < max_sim_seconds:
                
                agent_id = list(obs.keys())[0]
                agent_obs = obs[agent_id]

                yaw_raw = float(agent_obs['pose_theta'])
                yaw_c   = self.mpc.update_yaw(yaw_raw)
                vx      = float(agent_obs.get('linear_vel_x', 0.0))
                vy      = float(agent_obs.get('linear_vel_y', 0.0))
                v       = math.sqrt(vx**2 + vy**2)

                vehicle_state = State(
                    x   = float(agent_obs['pose_x']),
                    y   = float(agent_obs['pose_y']),
                    v   = v,
                    yaw = yaw_c,
                )

               
                if step_count % MPC_INTERVAL == 0:
                    ref_path = self.mpc.calc_ref_trajectory(
                        vehicle_state, ref_x, ref_y, ref_yaw, ref_v
                    )
                    x0 = [vehicle_state.x, vehicle_state.y, vehicle_state.v, vehicle_state.yaw]
                    self.mpc.oa, self.mpc.odelta_v, _mode, _cost = self.mpc.mpc_control(
                        ref_path, x0)

                    # Cache predicted horizon for visualisation
                    # Use the correct cache depending on which solver just ran.
                    _xk_cache = (self.mpc._prev_xk_nl if _mode == 'NMPC'
                                 else self.mpc._prev_xk)
                    if _xk_cache is not None:
                        pred_horizon_x = _xk_cache[0, :].tolist()
                        pred_horizon_y = _xk_cache[1, :].tolist()

                    # Steering: from MPC odelta_v[0]
                    if self.mpc.odelta_v is not None:
                        steer = float(np.clip(self.mpc.odelta_v[0],
                                              cfg.MIN_STEER, cfg.MAX_STEER))

                    
                    _ref_v_now = float(ref_v[self.mpc.target_ind])
                    step_size  = cfg.MAX_ACCEL * cfg.DTK
                    if self.mpc.oa is not None:
                        v_mpc = vehicle_state.v + float(self.mpc.oa[0]) * cfg.DTK
                        speed_desired = float(np.clip(
                            v_mpc,
                            max(cfg.MIN_SPEED, _ref_v_now - step_size),  # floor: ref − 1 step
                            cfg.MAX_SPEED))
                    else:
                        # Fallback before first MPC solve: ramp toward reference profile
                        speed_desired = float(np.clip(
                            speed_desired + np.sign(_ref_v_now - speed_desired) * step_size,
                            cfg.MIN_SPEED, cfg.MAX_SPEED))

                ind  = self.mpc.target_ind
                cx_  = ref_x[ind]; cy_ = ref_y[ind]; cy_yaw = ref_yaw[ind]
                cte  = abs(-(vehicle_state.x - cx_) * math.sin(cy_yaw)
                        + (vehicle_state.y - cy_) * math.cos(cy_yaw))
                self._log.append({
                    't':        sim_time,
                    'x':        vehicle_state.x,
                    'y':        vehicle_state.y,
                    'yaw':      yaw_raw,          # Uncorrected raw yaw from visualization
                    'v':        v,
                    'speed_cmd': speed_desired,
                    'steer':    steer,
                    'cte':      cte,
                    'mode':     self.mpc._active_mode,
                    'cost':     _cost if step_count % MPC_INTERVAL == 0 else float('nan'),
                    'pred_x':   list(pred_horizon_x),
                    'pred_y':   list(pred_horizon_y),
                })

                action = np.array([[steer, speed_desired]])
                obs, step_reward, done, truncated, info = env.step(action)

                # Update driven-path overlay
                self._driven_pts_list.append(
                    [vehicle_state.x, vehicle_state.y])

                # Always call render so GL callbacks fire every step:
                # this keeps the driven trail and MPC horizon live in the
                # viewport.  Only store the returned frame every RENDER_EVERY
                # steps to avoid excessive memory use.
                frame = env.render()
                if step_count % RENDER_EVERY == 0:
                    frame = self._hud_overlay(    # PIL text HUD on top
                        frame, sim_time, v, steer, cte,
                        mode=self.mpc._active_mode)
                    self._frames.append(frame)


                    # Live snapshot
                    cur_sec = int(sim_time)
                    if _IN_COLAB and cur_sec > _last_preview_sec:
                        _last_preview_sec = cur_sec
                        try:
                            from IPython.display import display, Image as _Img
                            from io import BytesIO
                            _buf = BytesIO()
                            from PIL import Image as _PILImg
                            _PILImg.fromarray(frame).save(_buf, format='PNG')
                            _buf.seek(0)
                            from IPython.display import clear_output
                            clear_output(wait=True)
                            display(_Img(data=_buf.read()))
                        except Exception:
                            pass

                # Debug print once per sim-second with state and MPC info
                if step_count % DEBUG_EVERY == 0:
                    _cost_str = f"{_cost:>10.2f}" if not math.isnan(_cost) else f"{'---':>10}"
                    print(f"{step_count:>7}  {sim_time:>7.2f}s  "
                          f"{vehicle_state.x:>8.3f}  {vehicle_state.y:>8.3f}  "
                          f"{v:>7.3f}  {speed_desired:>7.3f}  "
                          f"{math.degrees(steer):>10.2f}  {cte:>8.4f}  cost={_cost_str}")

                sim_time   += GYM_DT
                step_count += 1

                # Stop on collision
                collided = any(done.values()) if isinstance(done, dict) else bool(done)
                if collided:
                    print(f"[COLLISION] step={step_count}  sim_t={sim_time:.3f}s  "
                          f"x={vehicle_state.x:.3f}  y={vehicle_state.y:.3f}  "
                          f"v={v:.3f} m/s")
                    break

        except KeyboardInterrupt:
            print(f"\n[INTERRUPTED] step={step_count}  sim_t={sim_time:.3f}s  "
                  f"frames={len(self._frames)}  log-entries={len(self._log)}")
            print("Saving plots and frames before exit ...")
        finally:
            # Always runs: on normal finish, collision, or KeyboardInterrupt.
            # Generates plots here so they are saved even if the cell is stopped.
            env.close()
            wall = time.time() - t_wall_start
            try:
                collided_str = "  COLLISION" if (any(done.values()) if isinstance(done, dict) else bool(done)) else ""
            except Exception:
                collided_str = ""
            print(f"Simulation finished.  sim-time={sim_time:.2f}s  wall-time={wall:.1f}s  "
                  f"steps={step_count}  log-entries={len(self._log)}{collided_str}")
            self.save_log_csv("mpc_log.csv")
            self.plot_cte("cte_plot.png")
            self.plot_cost("cost_plot.png")

    #####################################################################################
    def _setup_render_callbacks(self, env):
        """
        Registers line + point renderers inside the gym's GL pipeline onto every frame.
        """
        wp_x = self.mpc.waypoints[:, 0]
        wp_y = self.mpc.waypoints[:, 1]
        raceline_pts = np.column_stack([wp_x, wp_y]).astype(np.float32)

        has_border = (self.mpc._has_border
                      and hasattr(self.mpc, 'border_walls_xy')
                      and self.mpc.border_walls_xy is not None)

        # Static geometry: created once on first callback invocation
        static = {'done': False}

        def _static_cb(er):
            if static['done']:
                return
            static['done'] = True
            
            er.get_lines_renderer(raceline_pts, color=(255, 255, 255), size=1)
            if has_border:
                bw = self.mpc.border_walls_xy
                er.get_lines_renderer(
                    np.column_stack([bw[:, 0], bw[:, 1]]).astype(np.float32),
                    color=(255, 215, 0), size=2)   # left wall
                er.get_lines_renderer(
                    np.column_stack([bw[:, 2], bw[:, 3]]).astype(np.float32),
                    color=(255, 107, 53), size=2)  # right wall
                bc = self.mpc.border_corridor_xy
                er.get_lines_renderer(
                    np.column_stack([bc[:, 0], bc[:, 1]]).astype(np.float32),
                    color=(200, 170, 0), size=1)   # left corridor
                er.get_lines_renderer(
                    np.column_stack([bc[:, 2], bc[:, 3]]).astype(np.float32),
                    color=(200, 90, 30), size=1)   # right corridor

        # Currently they remain static at the last updated position; but they need to be made dynamically updating
        # Driven trail - updated every frame
        # Capture self directly; read _driven_pts_list at callback time (no stale alias)
        runner = self

        def _driven_cb(er):
            pts_list = runner._driven_pts_list
            if len(pts_list) < 2:
                return
            pts = np.array(pts_list, dtype=np.float32)
            if runner._driven_state['renderer'] is None:
                runner._driven_state['renderer'] = er.get_lines_renderer(
                    pts, color=(0, 207, 255), size=2)   #Driven path
            else:
                runner._driven_state['renderer'].update(pts)

        # MPC predicted horizon
        def _pred_cb(er):
            xk = (runner.mpc._prev_xk_nl if runner.mpc._active_mode == 'NMPC'
                  else runner.mpc._prev_xk)
            if xk is None:
                return
            pts = np.column_stack([xk[0, :], xk[1, :]]).astype(np.float32)
            if len(pts) < 2:
                return
            if runner._pred_state['renderer'] is None:
                runner._pred_state['renderer'] = er.get_lines_renderer(
                    pts, color=(57, 255, 20), size=2)   #MPC predicted horizon
            else:
                runner._pred_state['renderer'].update(pts)

        env.unwrapped.add_render_callback(_static_cb)
        env.unwrapped.add_render_callback(_driven_cb)
        env.unwrapped.add_render_callback(_pred_cb)

    ###########################################################################
    @staticmethod
    def _hud_overlay(frame: np.ndarray, sim_t, v, steer, cte, **kwargs) -> np.ndarray:
        """Telemetry text heads up display onto the numpy frame array using PIL."""
        from PIL import Image, ImageDraw
        img  = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        lines = [
            (f"t  = {sim_t:.1f} s",      (255, 255, 255)),
            (f"v  = {v:.2f} m/s",        (0,   207, 255)),
            (f"\u03b4  = {math.degrees(steer):.1f}\u00b0", (255, 215,   0)),
            (f"CTE= {cte:+.3f} m",       (255, 107,  53)),
        ]
        x0, y0, dy = 10, 10, 18
        for i, (text, colour) in enumerate(lines):
            # thin shadow for readability
            draw.text((x0 + 1, y0 + i*dy + 1), text, fill=(0, 0, 0))
            draw.text((x0,     y0 + i*dy),     text, fill=colour)
        return np.asarray(img)

    ###########################################################################
    def save_log_csv(self, out_path: str = "mpc_log.csv"):
        """Write the per-step telemetry log to a CSV file."""
        if not self._log:
            print("No log data to save."); return
        import csv as _csv
        # pred_x / pred_y are lists. Serialise as space-separated strings
        fieldnames = ['t', 'x', 'y', 'yaw', 'v', 'speed_cmd', 'steer',
                      'cte', 'mode', 'cost', 'pred_x', 'pred_y']
        with open(out_path, 'w', newline='') as f:
            writer = _csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for e in self._log:
                row = dict(e)
                row['pred_x'] = ' '.join(f"{v:.4f}" for v in e['pred_x'])
                row['pred_y'] = ' '.join(f"{v:.4f}" for v in e['pred_y'])
                writer.writerow(row)
        size_kb = os.path.getsize(out_path) / 1e3
        print(f"Log saved: {out_path}  ({len(self._log)} rows, {size_kb:.1f} KB)")

    ###########################################################################
    def save_mp4(self, out_path: str = "mpc_run.mp4", fps: int = 5):
        """Write the annotated gym frames captured during run() to .mp4."""
        if not self._frames:
            print("No frames captured "); return
        
        import imageio
        print(f"Writing {len(self._frames)} annotated frames → {out_path} ...")
        imageio.mimwrite(out_path, self._frames, fps=fps)
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"Saved: {out_path}  ({size_mb:.1f} MB)")

    #########################################################################
    def plot_cost(self, out_path: str = "cost_plot.png"):
        if not self._log:
            print("No log data."); return

        ts    = [e['t']    for e in self._log]
        costs = [e['cost'] for e in self._log]
        modes = [e['mode'] for e in self._log]

        # Trim warmup (first 2.5 s) and end artefacts (last 2 s) for plotting only
        _t_max = max(ts) if ts else 0.0
        _t_lo, _t_hi = 2.5, _t_max - 2.0

        # Separate by solver so they can be coloured differently
        t_lmpc = [t for t, c, m in zip(ts, costs, modes) if m == 'LMPC' and not math.isnan(c) and _t_lo <= t <= _t_hi]
        c_lmpc = [c for t, c, m in zip(ts, costs, modes) if m == 'LMPC' and not math.isnan(c) and _t_lo <= t <= _t_hi]
        t_nmpc = [t for t, c, m in zip(ts, costs, modes) if m == 'NMPC' and not math.isnan(c) and _t_lo <= t <= _t_hi]
        c_nmpc = [c for t, c, m in zip(ts, costs, modes) if m == 'NMPC' and not math.isnan(c) and _t_lo <= t <= _t_hi]

        fig, ax = plt.subplots(figsize=(10, 3))
        if t_lmpc:
            k=[]
            ax.scatter(t_lmpc, c_lmpc, s=4, color='#00cfff', label='LMPC', zorder=3)
            k.append(c_lmpc)
        if t_nmpc:
            k=[]
            ax.scatter(t_nmpc, c_nmpc, s=4, color='#ff6b35', label='NMPC', zorder=3)
            k.append(c_nmpc)
        ax.set_xlabel('sim time [s]')
        ax.set_ylabel('Objective cost')
        ax.set_title(f'MPC Optimization Cost per Solve (mean={np.mean(costs):.4f}, max={np.max(costs):.4f})')
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Cost plot saved: {out_path}")

    #########################################################################
    def plot_cte(self, out_path: str = "cte_plot.png"):
        if not self._log:
            print("No log data."); return

        ts   = [e['t']   for e in self._log]
        ctes = [e['cte'] for e in self._log]

        # Trim warmup (first 2.5 s) and end artefacts (last 2 s) for plotting only
        _t_max = max(ts) if ts else 0.0
        _t_lo, _t_hi = 2.5, _t_max - 2.0
        ts_plot   = [t for t, c in zip(ts, ctes) if _t_lo <= t <= _t_hi]
        ctes_plot = [c for t, c in zip(ts, ctes) if _t_lo <= t <= _t_hi]

        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(ts_plot, ctes_plot, color='#e94560', linewidth=1)
        ax.set_xlabel('sim time [s]'); ax.set_ylabel('CTE [m]')
        _ctes_stat = ctes_plot if ctes_plot else ctes
        ax.set_title(f'Cross-Track Error  (mean={np.mean(_ctes_stat):.4f} m  '
                     f'max={np.max(_ctes_stat):.4f} m)')
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"CTE plot saved: {out_path}")


###################################################################################################################
def colab_main(
    max_speed:         float = 5.0,
    solver_mode:       str   = 'AUTO',   # 'LMPC' | 'NMPC' | 'AUTO'
    max_sim_seconds:   float = 60.0 * 12,
    waypoints_csv:     str   = "waypoints.csv",
    border_coeffs_csv: str   = "Spielberg_border_coeffs.csv",
    map_name:          str   = "Spielberg",
):
    """
    Entry point for Colab.  All parameters can be passed from the notebook cell:

        from mpc_gym_colab import colab_main
        colab_main(max_speed=7.0, solver_mode='NMPC', max_sim_seconds=300)
    """
    solver_mode = solver_mode.upper()
    assert solver_mode in ('LMPC', 'NMPC', 'AUTO'), \
        "solver_mode must be 'LMPC', 'NMPC', or 'AUTO'"

    print(f"Configuration → MAX_SPEED={max_speed} m/s  |  solver_mode={solver_mode}  |"
          f"  sim={max_sim_seconds:.0f}s")
    print("Starting Constrained-MPC runner ...")


    runner = GymMPCRunner(
        waypoints_csv     = waypoints_csv,
        map_name          = map_name,
        border_coeffs_csv = border_coeffs_csv,
        max_speed         = max_speed,
    )
    # Pin the solver: 'LMPC', 'NMPC', or None (defaults to LMPC, no switching)
    runner.mpc._fixed_mode = solver_mode if solver_mode in ('LMPC', 'NMPC') else None

    try:
        runner.run(
            max_sim_seconds = max_sim_seconds,
        )
    finally:
        runner.save_mp4("mpc_run.mp4", fps=5)
        # cte_plot.png and cost_plot.png are already saved inside run()'s finally.
        # Each display is in its own try/except so a failed video export
        # does not prevent the CTE and cost plots from being shown.
        try:
            from IPython.display import display, Video, Image
            print("\n=== Final video ===")
            display(Video("mpc_run.mp4", embed=True, width=800))
        except Exception:
            pass
        try:
            from IPython.display import display, Image
            print("\n=== CTE plot ===")
            display(Image("cte_plot.png"))
        except Exception:
            pass
        try:
            from IPython.display import display, Image
            print("\n=== Optimization cost plot ===")
            display(Image("cost_plot.png"))
        except Exception:
            pass


if __name__ == "__main__":
    colab_main()
