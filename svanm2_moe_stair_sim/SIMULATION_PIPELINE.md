# SvanM2 MoE-CTS 90k Multi-Stair MuJoCo Simulation Pipeline

Comprehensive technical documentation of the simulation environment, robot kinematics, actuator transmission, neural policy inference, teleoperation interface, and asset structure developed for the **xTerra SvanM2** quadruped robot running the **MoE-CTS 90,000-update checkpoint**.

---

## 1. Directory Structure & File Manifest

The complete simulation package is self-contained and version-controlled within this repository:
* **Simulation Package Directory**: `./svanm2_moe_stair_sim/`
* **Robot MJCF & Meshes**: `./xterra_mjlab/assets/svanm2/`
* **Checkpoints**: `./checkpoints/svanm2_moe/`

### File Manifest

| File | Path | Description |
| :--- | :--- | :--- |
| **`build_scene.py`** | [`build_scene.py`](./build_scene.py) | Procedural scene generator that synthesizes the complete multi-stair environment XML. Sets up 6 color-coded tracks, entrance banners, guide curbs, firm contact parameters, and default standing keyframes. |
| **`scene.xml`** | [`scene.xml`](./scene.xml) | Fully assembled MuJoCo MJCF model containing the SvanM2 robot description, STL meshes, infinite floor plane, 72 stair steps, 6 turnaround platforms, and lighting. |
| **`controller.py`** | [`controller.py`](./controller.py) | MoE 90k TorchScript inference controller. Implements 50 Hz policy inference, 45-dim observation construction, 5-frame oldest-first history rolling, action scaling, and 200 Hz coupled belt-drive PD torque physics. |
| **`teleop.py`** | [`teleop.py`](./teleop.py) | Dual-input teleoperation manager with hardware-verified speed ceilings. Contains the background non-blocking terminal stdin listener (`TerminalKeyboardListener`) and viewer GLFW key callback. |
| **`run_interactive_sim.py`** | [`run_interactive_sim.py`](./run_interactive_sim.py) | Main interactive simulation entrypoint. Launches MuJoCo passive viewer, runs real-time synchronization, manages viewport 3D HUD, terminal HUD, and visual state protection. |
| **`test_sim_headless.py`** | [`test_sim_headless.py`](./test_sim_headless.py) | Headless automated test suite verifying scene compilation, standing stability, multi-track locomotion, and top-platform turnaround descent. |
| **`README.md`** | [`README.md`](./README.md) | Quickstart guide and control reference. |
| **`feet_ground_preview.png`** | [`feet_ground_preview.png`](./feet_ground_preview.png) | High-resolution offscreen render verifying clean foot-to-ground contact without sinking. |
| **`stairs_climbing_preview.png`** | [`stairs_climbing_preview.png`](./stairs_climbing_preview.png) | High-resolution offscreen render showing dynamic stair ascension on the 10cm track. |

---

## 2. Upstream Dependencies & Model Checkpoints

The pipeline consumes assets and checkpoints included in this repository:

1. **MoE-CTS 90k Policy Checkpoint**:
   * **TorchScript Model**: [`checkpoints/svanm2_moe/offline_inference.ts`](../checkpoints/svanm2_moe/offline_inference.ts)
   * **Training Checkpoint**: [`checkpoints/svanm2_moe/model_90000.pt`](../checkpoints/svanm2_moe/model_90000.pt)
   * **Check SHA-256**: `440bbb28d554f75db38bfd163351c04ea4a32d9d06764c37b7e6071d41daa5cb`
   * **Architecture**: Privileged MoE student policy (deterministic encoder + gating actor). Evaluates using proprioception only, with zero ground-truth heightmap requirements.
2. **Robot URDF/MJCF & STL Meshes**:
   * **Base MJCF**: [`xterra_mjlab/assets/svanm2/xml/svanm2_mjlab.xml`](../xterra_mjlab/assets/svanm2/xml/svanm2_mjlab.xml)
   * **Meshes Directory**: [`xterra_mjlab/assets/svanm2/meshes/`](../xterra_mjlab/assets/svanm2/meshes/)
3. **Python Environment**:
   * **Prerequisites**: Python 3.10+, MuJoCo >= 3.2.0, PyTorch >= 2.0.0.

---

## 3. Environment & Terrain Architecture

The world consists of **6 side-by-side stair tracks** aligned along the +X axis, separated by $3.80\text{ m}$ along the Y axis:

```
Y = +9.50 m  ─── [Track 6: 20 cm Stairs]  (12 steps, total rise 2.40 m) ─── [Top Platform]
Y = +5.70 m  ─── [Track 5: 16 cm Stairs]  (12 steps, total rise 1.92 m) ─── [Top Platform]
Y = +1.90 m  ─── [Track 4: 15 cm Stairs]  (12 steps, total rise 1.80 m) ─── [Top Platform]
Y = -1.90 m  ─── [Track 3: 14 cm Stairs]  (12 steps, total rise 1.68 m) ─── [Top Platform]
Y = -5.70 m  ─── [Track 2: 12 cm Stairs]  (12 steps, total rise 1.44 m) ─── [Top Platform]
Y = -9.50 m  ─── [Track 1: 10 cm Stairs]  (12 steps, total rise 1.20 m) ─── [Top Platform]
```

### Track Specifications

| Track | Step Height ($h$) | Steps | Step Tread ($d$) | Stair Width | Total Rise | Top Platform Size | Theme Color |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **10 cm** ($0.10\text{ m}$) | 12 | $0.30\text{ m}$ | $2.00\text{ m}$ | $1.20\text{ m}$ | $3.0\text{ m} \times 2.4\text{ m}$ | Sky Blue |
| **2** | **12 cm** ($0.12\text{ m}$) | 12 | $0.30\text{ m}$ | $2.00\text{ m}$ | $1.44\text{ m}$ | $3.0\text{ m} \times 2.4\text{ m}$ | Teal |
| **3** | **14 cm** ($0.14\text{ m}$) | 12 | $0.30\text{ m}$ | $2.00\text{ m}$ | $1.68\text{ m}$ | $3.0\text{ m} \times 2.4\text{ m}$ | Emerald Green |
| **4** | **15 cm** ($0.15\text{ m}$) | 12 | $0.30\text{ m}$ | $2.00\text{ m}$ | $1.80\text{ m}$ | $3.0\text{ m} \times 2.4\text{ m}$ | Amber |
| **5** | **16 cm** ($0.16\text{ m}$) | 12 | $0.30\text{ m}$ | $2.00\text{ m}$ | $1.92\text{ m}$ | $3.0\text{ m} \times 2.4\text{ m}$ | Coral Red |
| **6** | **20 cm** ($0.20\text{ m}$) | 12 | $0.30\text{ m}$ | $2.00\text{ m}$ | $2.40\text{ m}$ | $3.0\text{ m} \times 2.4\text{ m}$ | Purple |

### Ground & Collision Architecture
* **Infinite Floor Plane**: Defined at $Z = 0.000\text{ m}$ as a single `<geom name="floor" type="plane"/>`.
* **Approach Runway**: Extends from $X = -4.00\text{ m}$ to $X = 0.00\text{ m}$. To prevent overlapping contact normal conflict between the floor and runway box, the runway is set to a thin visual marker (`contype="0" conaffinity="0"`). Ground contact on the runway is calculated cleanly by the floor plane.
* **Steps Riser Alignment**: Step 1 top surface is at $Z = 1 \times h$ ($0.10\text{ m}$ on Track 1). Step $s$ top surface is at $Z = s \times h$. Each step extends from $X = (s-1) \times 0.30\text{ m}$ to $X = s \times 0.30\text{ m}$.
* **Top Turnaround Platform**: Extends from $X = 3.60\text{ m}$ to $X = 6.60\text{ m}$ ($3.0\text{ m}$ length) with width $2.40\text{ m}$. Top surface is flush with step 12 at $Z = 12 \times h$. Features $0.08\text{ m}$ side guide curbs to prevent edge slips while rotating $180^\circ$.

---

## 4. Actuator Model & Kinematic Transmission

The SvanM2 quadruped features a **parallel belt mechanism**: the calf motor is mounted in the thigh housing and couples to the calf joint via a belt with transmission ratio $0.5$.

### A. Leg Transmission Jacobian
The per-leg mapping from joint space $(q_{\text{hip}}, q_{\text{thigh}}, q_{\text{calf}})$ to motor rotor space $(\theta_{\text{hip}}, \theta_{\text{thigh}}, \theta_{\text{calf}})$ is:

$$J_{\text{leg}} = \begin{bmatrix} 8.0 & 0.0 & 0.0 \\ 0.0 & 8.0 & 0.0 \\ 0.0 & 8.0 & 16.0 \end{bmatrix}$$

Full 12-DoF transmission matrix across all 4 legs ($\text{FL}, \text{FR}, \text{RL}, \text{RR}$):
$$J_{\text{coupling}} = I_4 \otimes J_{\text{leg}}$$

### B. Coupled Motor-Space PD Torque Law
Per-motor PD gains are scaled by the gear ratios squared so effective joint stiffness matches configured gains:
$$\mathbf{G} = [8.0, 8.0, 16.0] \quad (\text{hip, thigh, calf})$$
$$\boldsymbol{\tau}_{\text{motor}} = \left(\frac{k_p}{\mathbf{G}^2}\right) \odot \left( J_{\text{coupling}} (q_{\text{target}} - q) \right) - \left(\frac{k_d}{\mathbf{G}^2}\right) \odot \left( J_{\text{coupling}} \dot{q} \right)$$
$$\boldsymbol{\tau}_{\text{joint}} = \text{clamp}\left( J_{\text{coupling}}^T \boldsymbol{\tau}_{\text{motor}}, -\boldsymbol{\tau}_{\max}, \boldsymbol{\tau}_{\max} \right)$$
* $k_p = 20.0$, $k_d = 0.7$
* $\boldsymbol{\tau}_{\max} = [\pm 12.0, \pm 12.0, \pm 24.0]\text{ Nm}$ (hip, thigh, calf)
* Actuator loop rate: $200\text{ Hz}$ ($dt = 0.005\text{ s}$)

---

## 5. Neural Policy Controller Architecture

The policy is executed in [`controller.py`](./controller.py) at **50 Hz** (decimation 4 from the 200 Hz physics loop).

### A. 45-Dimensional Observation Frame Layout

| Slice | Dimensions | Meaning | Scaling / Preprocessing |
| :---: | :---: | :--- | :--- |
| `[0:3]` | 3 | Velocity command $(v_x, v_y, \omega_z)$ | Raw $(m/s, rad/s)$ |
| `[3:6]` | 3 | IMU base angular velocity | $0.25 \times \boldsymbol{\omega}_{\text{body}}$ |
| `[6:9]` | 3 | Projected gravity vector | $\mathbf{R}_{\text{body}}^T [0, 0, -1]^T$ |
| `[9:21]` | 12 | Joint position error | $q - q_0$ (nominal standing pose) |
| `[21:33]` | 12 | Joint velocities | $0.05 \times \dot{q}$ |
| `[33:45]` | 12 | Previous policy action | $\text{clip}(a_{t-1}, -6.0, 6.0)$ |

### B. 5-Frame History Buffer
* **Buffer Shape**: $5 \times 45 = 225$ elements.
* **Order**: **Oldest-first** (`history[0]` is the oldest frame, `history[4]` is the newest).
* **Reset Behavior**: When teleported or reset, the first frame is repeated 5 times across the buffer.
* **Inference**: `raw_actions = policy(history.flatten())` producing $12$ raw continuous actions.

### C. Action Mapping & Target Bounds
$$q_{\text{target}} = q_0 + \text{diag}([0.15, 0.15, 0.30]) \times \text{clip}(a_t, -6.0, 6.0)$$
* **Nominal Pose ($q_0$)**:
  $$q_0 = [0.0, 0.5806, -1.1716, 0.0, 0.5806, -1.1716, 0.0, 0.7167, -1.1342, 0.0, 0.7167, -1.1342]$$
* **Simulation Target Envelopes**:
  * Abduction/Hip: $[-1.0472, +1.0472]\text{ rad}$
  * Front Thigh: $[-1.5708, +3.4907]\text{ rad}$
  * Rear Thigh: $[-0.5236, +4.5379]\text{ rad}$
  * Calf: $[-2.7227, +0.5000]\text{ rad}$ (allows forward stride swing without hyperextension)

---

## 6. Teleoperation & Dual-Input Interface

To resolve MuJoCo's built-in GLFW keybinding conflicts (where `W`, `A`, `S`, `L` toggle wireframe and actuator lines), the teleoperation system provides dual input streams:

### A. Terminal Non-Blocking Listener (Primary)
Implemented in `TerminalKeyboardListener` using Linux POSIX `termios`, `tty.setcbreak`, and `select.select`:
* Reads keystrokes directly from the terminal console without needing to press Enter.
* Has **zero contact with MuJoCo GLFW**, completely eliminating visual toggle glitches.

### B. MuJoCo Viewer Key Handler (Secondary)
* Added non-conflicting arrow keys: `Up` (forward), `Down` (backward), `Left` (strafe left), `Right` (strafe right).
* Added letter alternatives: `I` (forward), `K` (backward), `J` (strafe left), `U` (strafe right).
* Added per-frame visual state guard in `run_interactive_sim.py` that continuously resets:
  ```python
  viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = 0
  viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_LIGHT] = 0
  viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_SKIN] = 0
  viewer.opt.label = mujoco.mjtLabel.mjLABEL_NONE
  if hasattr(viewer, 'user_scn') and viewer.user_scn is not None:
      viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_WIREFRAME] = 0
  ```

### C. Speed Limits & Key Mappings

| Command / Action | Terminal Key | Viewer Window Key | Speed Step | Range Limit |
| :--- | :---: | :---: | :---: | :---: |
| **Forward Velocity** | `W` | `Up Arrow` / `I` | $+0.05\text{ m/s}$ | $[-0.50, +0.50]\text{ m/s}$ |
| **Backward Velocity** | `S` | `Down Arrow` / `K` | $-0.05\text{ m/s}$ | $[-0.50, +0.50]\text{ m/s}$ |
| **Strafe Left** | `A` | `Left Arrow` / `J` | $+0.05\text{ m/s}$ | $[-0.60, +0.60]\text{ m/s}$ |
| **Strafe Right** | `D` | `Right Arrow` / `U` | $-0.05\text{ m/s}$ | $[-0.60, +0.60]\text{ m/s}$ |
| **Turn Left** | `Q` | `Q` | $+0.05\text{ rad/s}$ | $[-0.40, +0.40]\text{ rad/s}$ |
| **Turn Right** | `E` | `E` | $-0.05\text{ rad/s}$ | $[-0.40, +0.40]\text{ rad/s}$ |
| **Emergency STOP** | `Space` | `Space` | Instant Zero | $0.0\text{ m/s}$ |
| **Teleport Tracks 1–6** | `1` – `6` | `1` – `6` | Instant | Approaches Track 1 to 6 |
| **Teleport Top Platform** | `T` | `T` | Instant | Top of current track (facing downstairs) |
| **Reset Pose** | `R` | `R` | Instant | Current track runway entrance |

---

## 7. Solved Technical Issues Summary

| Issue | Root Cause | Implemented Solution |
| :--- | :--- | :--- |
| **Feet submerged in ground** | 1. Base spawned at $0.32\text{ m}$ (foot sphere reached $-2.2\text{ cm}$).<br>2. Approach runway was a $3\text{ cm}$ raised physical box.<br>3. `solimp="0.015 1 0.01"` caused soft spongy sinking under $12\text{ kg}$ body weight. | 1. Spawn height raised to $0.365\text{ m}$ (and $H + 0.365\text{ m}$ on platform).<br>2. Runway converted to non-colliding visual plane (`contype=0, conaffinity=0`).<br>3. Contact stiffness upgraded to `solimp="0.9 0.95 0.001"`, `solref="0.002 1"`.<br>4. Added `home` standing keyframe. |
| **Robot paralyzed (no walking)** | `TARGET_MAX[calf]` was locked at $-0.88776\text{ rad}$ (hardware safety stop). Policy requires calf swing to $\approx -0.40\text{ rad}$ to step forward. | Simulation target bound relaxed to $+0.50\text{ rad}$. The robot now swings legs naturally and advances at $0.30 - 0.50\text{ m/s}$. |
| **MuJoCo GUI key toggles** | Built-in GLFW viewer hardcodes toggles for wireframe (`W`), actuators (`A`), light (`L`), skin (`S`). | 1. Added non-blocking terminal keyboard reader.<br>2. Added arrow keys to viewer callback.<br>3. Added frame-by-frame visual state guard to reset flags. |

---

## 8. Verification & Test Execution

### Automated Headless Tests
Run the test suite with:
```bash
# From repository root:
python svanm2_moe_stair_sim/test_sim_headless.py
```
* `test_01_scene_integrity`: Confirmed all 155 geoms, 13 joints, 12 actuators, and 72 steps present.
* `test_02_standing_stability`: Confirmed finite coordinates, base height $> 0.28\text{ m}$, tilt $< 15^\circ$.
* `test_03_forward_locomotion_on_all_tracks`: Confirmed forward advance across all 6 tracks.
* `test_04_top_platform_teleport_and_descent_setup`: Confirmed platform Z coordinates and downward heading.

### Interactive Simulation Launch
```bash
# Standard launch at Track 1 (10cm stairs)
python svanm2_moe_stair_sim/run_interactive_sim.py

# Launch directly at Track 4 (15cm stairs)
python svanm2_moe_stair_sim/run_interactive_sim.py --track 4

# Launch on top turnaround platform of Track 2 (12cm) to test descent
python svanm2_moe_stair_sim/run_interactive_sim.py --track 2 --top
```
