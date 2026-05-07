# MPC on F1TENTH (MpcOnF1Tenth)

This repository contains **ROS 2** & **Python** implementations of **Model Predictive Control (MPC)** for the **F1TENTH**.
The nodes subscribe to the car state (odometry), track a waypoint-defined racing line, solve an MPC problem at each control tick, and publish commands to `/drive`.

---

All controllers here assume a **kinematic bicycle** model.
---
## MPC Controllers:

### 1. `mpc_linear_bordered_casadi.py`
**Non-linear and Linear MPC (CasADi + IPOPT) with linear corridor visualization**

A dual-mode MPC controller implemented with CasADi.

- **Subscribes:** `/ego_racecar/odom` (`nav_msgs/Odometry`)
- **Publishes:** `/drive` (`ackermann_msgs/AckermannDriveStamped`)

- **State:** $[x, y, v, yaw]$
- **Control:** $[a, \delta]$ (acceleration, steering angle)
- **Horizon:** `TK = 2` steps with `DTK = 0.2 s` (≈ `0.4 s` total horizon)

#### Cost Structure:
- **State tracking penalty:**

$$
(x - x_{\text{ref}})^T Q (x - x_{\text{ref}})
$$

with terminal cost $Q_f$

- **Input magnitude penalty:**

$$
u^T R u
$$

- **Input rate penalty:**

$$
\Delta u^T R_d \Delta u
$$


#### Modes:

##### NMPC (`USE_NMPC=True`)
- Nonlinear kinematic bicycle rollout enforced as equality constraints:

$$
x_{t+1} = f(x_t, u_t)
$$

- Solver: **IPOPT via CasADi Opti**
- Warm-start: Shifts the previous $(x, u)$ solution forward one step each tick
- If the solve fails, it attempts to use the **last IPOPT iterate** (`opti.debug.value(...)`) as a best-effort fallback

##### LMPC (`USE_NMPC=False`)
- Dynamics are linearized each tick into:

$$
x_{t+1} \approx A_t x_t + B_t u_t + C_t
$$

- The optimization is still built and solved with **CasADi Opti + IPOPT**
- Warm-start: Shifts the previously solved $(x, u)$ trajectory forward

#### Reference Generation:
- Waypoints are loaded from `waypoints.csv`, with:
  - Yaw unwrapping for continuity
  - Loop-closure bridging if the last-to-first waypoint gap is large
  - Resampling to uniform spacing (`dlk = 0.03 m`)

- Speed reference comes from `calc_speed_profile()`:
  - Curvature-based speed scaling
  - Forward - Backward acceleration passes to avoid impossible step drops in speed at corner entry

#### Border Visualization:
- The node publishes corridor wall markers computed from the reference path tangent normals:
  - `/ego_racecar/corridor_left`
  - `/ego_racecar/corridor_right`

- Corridor half-width:
  - `D_MAX = 1.0 m` (stored as `_corridor_d_max`)

- In the CasADi LMPC path, the corridor is implemented as a **hard linear constraint** at each horizon step $t \ge 1$:

$$
n_t \cdot (p_t - p_t^{\text{ref}}) \le D_{\max}
$$

$$
-n_t \cdot (p_t - p_t^{\text{ref}}) \le D_{\max}
$$

where $p_t = [x_t, y_t]$ and $n_t$ is the left-pointing unit normal.

#### Logging data into:
- `nmpc_casadi_performance.csv` if `USE_NMPC=True`
- `lmpc_casadi_performance.csv` if `USE_NMPC=False`

---

### 2. `mpc_linear_bordered_cvxpy.py`
**Linear MPC using OSQP with hard corridor constraints, and a Non-linear MPC using SLSQP with corridor penalty**

A linear MPC controller solved as a sparse QP using OSQP.

- **Subscribes:** `/ego_racecar/odom`
- **Publishes:** `/drive`

#### Linear MPC (`USE_NMPC=False`):
- State: $[x, y, v, yaw]$
- Control: $[a, \delta]$

- Sparse QP form:

$$
\min \frac{1}{2} z^T P z + q^T z \quad \text{s.t. } l \le A z \le u
$$

with decision vector:

$$
z = [x_0, \dots, x_T, u_0, \dots, u_{T-1}]
$$

#### Constraints:
- Initial state equality $x_0 = x_{\text{current}}$
- Linearized dynamics equalities
- Speed bounds, acceleration bounds, steering bounds
- Steering-rate based bounds $|\delta_{t+1} - \delta_t| \le \dot{\delta}_{\max} DT$
- **Hard corridor constraint** for $t = 1, \dots, T$:

$$
-D_{\max} + n_t^T p_t^{\text{ref}} \le n_t^T p_t \le D_{\max} + n_t^T p_t^{\text{ref}}
$$

- Corridor half-width: `D_MAX = 1.0 m`

#### Warm Start:
- Shifts the previous $x, u$ solution forward each tick
- Reuses OSQP dual multipliers (`res.y`) when dimensions match

#### Non-linear MPC (`USE_NMPC=True`):
- Implemented with `scipy.optimize.minimize(..., method='SLSQP')`
- Non-linear dynamics equality constraints
- **Soft corridor enforcement** via a quadratic hinge penalty:

$$
W_{\text{cor}} \max\left(0, \left|n_t^T p_t - n_t^T p_t^{\text{ref}}\right| - D_{\max}\right)^2
$$

#### Logging data into:
- `lmpc_cvxpy_performance.csv`
- `nmpc_cvxpy_performance.csv`

---

### 3. `mpc_linear_cvxpy.py`
**Linear MPC (OSQP)**

A linear MPC baseline using OSQP.

#### Model:
- State: $[x, y, v, yaw]$
- Control: $[a, \delta]$
- Horizon: `TK = 4`, `DTK = 0.1 s`

#### Constraints:
- Initial state equality
- Linearized dynamics equalities
- Speed bounds, acceleration bounds, steering bounds
- Steering-rate based bounds

#### Warm Start:
- Shifts the previous solution forward
- Stores previous dual multipliers when available

#### Logging:
- `lmpc_performance.csv`

---

## Track data:

All controllers use a CSV file:

- `waypoints.csv` loaded from the working directory and contains waypoints.

---

## Workflow:

1. Add the controller files inside the scripts directory inside the mpc folder of F1tenth. Start your virtual environment.
2. Launch the F1TENTH gym bridge for ROS2 (through instructions on the dev-humble branch of F1Tenth gym ROS) to simulate the ego-vehicle on a Foxglove window.
3. Once the simulator is up and running, in another terminal, source the virtual environment and ROS2. Then you run the controller:

```bash
ros2 run mpc mpc_linear_bordered_casadi.py
# or
ros2 run mpc mpc_linear_bordered_cvxpy.py
# or
ros2 run mpc mpc_linear_cvxpy.py
```
