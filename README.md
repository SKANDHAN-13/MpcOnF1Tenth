# MPC on F1TENTH (MpcOnF1Tenth)

This repository contains **ROS 2** & **Python** implementations of **Model Predictive Control (MPC)** for the **F1TENTH**.
The nodes subscribe to the car state (odometry), track a waypoint-defined racing line, solve an MPC problem at each control tick, and publish commands to `/drive`.

---

All controllers here assume a **kinematic bicycle** model with:

**State**
- `x` = global x position [m]  
- `y` = global y position [m]  
- `v` = speed [m/s]  
- `yaw` (ψ) = heading angle [rad]  

So the state vector is:

- **x = [x, y, v, yaw]ᵀ**

**Inputs**
- `a` = acceleration [m/s²]
- `delta` (δ) = steering rate [rad/s]

So the input vector is:

- **u = [a, delta]ᵀ**

**Constants**
- `WB` = wheelbase L [m] (from `mpc_config.WB`)
- `DTK` = time-step Δt [s] (from `mpc_config.DTK`)

---

## Nonlinear discrete-time update used in the scripts:
- **x_next   = x   + v * cos(yaw) * DTK**
- **y_next   = y   + v * sin(yaw) * DTK**
- **v_next   = v   + a * DTK**
- **yaw_next = yaw + (v / WB) * tan(delta) * DTK**

---

## Linearized model used in the linear MPC solvers:

The “linear MPC” versions approximate the nonlinear model around an operating point:
- `v` = operating speed
- `phi` (φ) = operating yaw

Then they use :
- **x_next ≈ A * x + B * u + C**
where:
- `x = [x, y, v, yaw]ᵀ`
- `u = [a, delta]ᵀ`

#### A matrix (4×4)
- A[0,0], A[1,1], A[2,2], A[3,3] = 1 , and they carry the current state forward.
- A[0,2] = DTK * cos(phi)  
- A[0,3] = -DTK * v * sin(phi)  
- A[1,2] = DTK * sin(phi)  
- A[1,3] =  DTK * v * cos(phi)  
- A[3,2] = DTK * tan(delta_bar) / WB  


#### B matrix (4×2)
This captures how inputs affect the next state:
- B[2,0] = DTK  
- B[3,1] = DTK * v / (WB * cos(delta_bar)^2)  

So:
- acceleration `a` only affects `v_next`
- steering `delta` only affects `yaw_next` (in the linearized model)

#### C vector (4×1)
This is the affine correction (the “constant offset” term):
- C[0] =  DTK * v * sin(phi) * phi
- C[1] = -DTK * v * cos(phi) * phi
- C[2] =  0
- C[3] = -DTK * v * delta_bar / (WB * cos(delta_bar)^2)

Thus C vector helps compensate for linearization error terms.

---

## MPC controllers :

### 1) `mpc_node.py` - Nonlinear MPC (NMPC) with CasADi + IPOPT

This version solves a **nonlinear** optimization problem with the nonlinear dynamics constraints shown above:

- State: `[x, y, v, yaw]`
- Control: `[a, delta]`
- Solver: IPOPT through CasADi `Opti`
- Warm-starting: shifts previous solution forward each tick
- Logs performance: `nmpc_performance.csv`
- RViz markers:
  - `/ego_racecar/mpc_ref_traj`
  - `/ego_racecar/driven_traj`

### 2) `mpc_node_cvxpympc.py` - Linear MPC as a QP (CVXPY + OSQP)

This one solves a **QP**:

- Uses the linearized model `x_next ≈ A x + B u + C`
- Solver: OSQP through CVXPY
- Also uses the same waypoint reference trajectory generation as in the mpc_node.py

### 3) `mpc_casadiLinearConst.py` - Linear MPC with CasADi + IPOPT + corridor constraints

This is a linearized MPC but solved with CasADi/IPOPT, and adds a lateral corridor constraint around the path:

At each horizon step `t` it computes:
- `p_t = [x_t, y_t]`
- `p_ref_t = [x_ref_t, y_ref_t]`
- `n_t` : A unit normal vector pointing left of the path tangent

and then enforces:

- **n_t · (p_t - p_ref_t) ≤ d_max**
- **- n_t · (p_t - p_ref_t) ≤ d_max**

- `d_max = 0.3 m` in the script.

It also publishes corridor markers:
- `/ego_racecar/corridor_left`
- `/ego_racecar/corridor_right`

---

## ROS 2 topics (expected)

- Subscribes:
  - `/ego_racecar/odom` (`nav_msgs/msg/Odometry`)
- Publishes:
  - `/drive` (`ackermann_msgs/msg/AckermannDriveStamped`)
  - RViz (`visualization_msgs/msg/Marker`):
    - `/ego_racecar/mpc_ref_traj`
    - `/ego_racecar/driven_traj`
    - (corridor version) `/ego_racecar/corridor_left`, `/ego_racecar/corridor_right`

---

## Waypoints / track data

All controllers use a CSV file:

- `waypoints.csv` loaded from the working directory and contains waypoints.

---

## Workflow:

1. Add the controller files inside the scripts directory inside the mpc folder of F1tenth. Start your virtual environment.
2. Launch the F1TENTH gym bridge for ROS2 to control the ego-vehicle simulation on a Foxglove window.
3. In another terminal, run the controller:

```bash
ros2 run mpc mpc_node.py
# or
ros2 run mpc mpc_node_cvxpympc.py
# or
ros2 run mpc mpc_casadiLinearConst.py
```
