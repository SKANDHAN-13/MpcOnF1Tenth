

### Installation

```bash
pip install f1tenth-gym casadi scipy imageio pillow
```

[![Google Colaboratory](https://img.shields.io/badge/Colab%2B-blue)](https://colab.research.google.com/drive/19uFirhiSCK8sSTIBBFCdwYT9NJ3Batkd?authuser=1#scrollTo=jTXBJnhzrRGU): Implementation of a pure-Python and F1Tenth Gym that formulates a receding-horizon optimal
control problem with track-corridorconstraints and solves it online using
[CasADi](https://web.casadi.org/) + IPOPT at every control interval.

Outputs written to the working directory:

| File | Description |
|---|---|
| `mpc_run.mp4` | Annotated simulation video (5 fps summary) |
| `cte_plot.png` | Cross-track error over simulation time |

---

## Component Reference

### `mpc_config` - Hyperparameter Dataclass

```python
@dataclass
class mpc_config:
    NXK: int = 4        # state dimension  [x, y, v, yaw]
    NU:  int = 2        # control dimension [accel, steer]
    TK:  int = 10       # horizon steps (= 2 s at DTK=0.2 s)
    ...
```

| Parameter | Role |
|---|---|
| `NXK` | State vector size: `[x, y, v, yaw]` |
| `NU` | Control vector size: `[accel, steer]` |
| `TK`  | MPC horizon (steps) |
| `DTK`  | MPC timestep |
| `dlk` | Reference waypoint spacing |
| `WB` | Vehicle wheelbase |
| `Rk`  | Control effort weight |
| `Rdk` | Control rate-of-change weight |
| `Qk` | Stage state-error weight |
| `Qfk` | Terminal state-error weight |
| `MAX_STEER` | Steering angle limit |
| `MAX_ACCEL` | Acceleration limit |
| `CORRIDOR_WIDTH` | Fixed metric safety buffer inset inward from each wall |

### `HeadlessMPC` - Core MPC Controller

```python
mpc = HeadlessMPC(
    waypoints_csv    = "waypoints.csv",
    border_coeffs_csv= "Spielberg_border_coeffs.csv", 
)
---

### `GymMPCRunner` : Simulation Harness

```python
runner = GymMPCRunner(
    waypoints_csv     = "waypoints.csv",
    map_name          = "Spielberg",
    border_coeffs_csv = "Spielberg_border_coeffs.csv",
)
runner.run(max_sim_seconds=60.0)
runner.save_mp4("mpc_run.mp4", fps=5)
runner.plot_cte("cte_plot.png")
```

Owns the gym environment lifecycle, the MPC call cadence, the frame capture
pipeline, and all output artefacts.

---

## Data Files

### `waypoints_spielberg.csv`

Three-column CSV with header `x,y,yaw`.  Stores the pre-recorded Spielberg raceline
sampled at the original logging frequency.  `_load_waypoints` resamples it to a
uniform 3 cm arc-length grid internally.

### `Spielberg_border_coeffs.csv`

Seven-column CSV with headers `w_tr_right, w_tr_left,a,b,c_center,c_left,c_right`.  

| Column | Meaning |
|---|---|
| `a`, `b` | Unit normal vector components at each waypoint |
| `c_center` | Signed projection of the raceline onto the normal |
| `c_left` | Signed projection of the left wall onto the normal |
| `c_right` | Signed projection of the right wall onto the normal |
