"""MoE 90k policy controller and actuator PD driver for SvanM2 in MuJoCo.

Matches the hardware transfer specification:
- Checkpoint: model_90000.pt / offline_inference.ts
- 5-frame x 45-dim history buffer (oldest-first into student network)
- Action scaling [0.15, 0.15, 0.30] with joint limit clamping
- Coupled belt-drive PD torque model (kp=20.0, kd=0.7, gear=[8,8,16])
- Policy at 50 Hz, physics at 200 Hz (decimation 4)
"""

import json
from pathlib import Path
import numpy as np
import torch
import mujoco

HERE = Path(__file__).resolve().parent

def find_policy_path() -> Path:
    candidates = [
        HERE.parent / "checkpoints/svanm2_moe/offline_inference.ts",
        HERE / "../checkpoints/svanm2_moe/offline_inference.ts",
        HERE.parent / "m2_training_artifacts/moe_hw_candidate_90000/offline_inference.ts",
        HERE.parent.parent / "m2_training_artifacts/moe_hw_candidate_90000/offline_inference.ts",
    ]
    for c in candidates:
        try:
            resolved = c.resolve()
            if resolved.exists():
                return resolved
        except Exception:
            continue
    raise FileNotFoundError("Could not find offline_inference.ts in checkpoints/svanm2_moe/")

POLICY_PATH = find_policy_path()

JOINTS = [f"{leg}_{stage}_joint" for leg in ("FL", "FR", "RL", "RR") for stage in ("hip", "thigh", "calf")]
Q0 = np.array([0., 0.5806, -1.1716, 0., 0.5806, -1.1716, 0., 0.7167, -1.1342, 0., 0.7167, -1.1342], dtype=np.float32)
STAND_Q = Q0.copy()
SCALE = np.tile([0.15, 0.15, 0.30], 4).astype(np.float32)
TORQUE_LIMITS = np.tile([12.0, 12.0, 24.0], 4).astype(np.float32)
GEAR = np.tile([8.0, 8.0, 16.0], 4).astype(np.float32)

# Joint target boundaries for simulation: allows full forward stepping stride and stair climbing
TARGET_MIN = np.array([
    -1.0472, -1.5708, -2.7227,  # FL
    -1.0472, -1.5708, -2.7227,  # FR
    -1.0472, -0.5236, -2.7227,  # RL
    -1.0472, -0.5236, -2.7227   # RR
], dtype=np.float32)

TARGET_MAX = np.array([
     1.0472,  3.4907,  0.5000,  # FL
     1.0472,  3.4907,  0.5000,  # FR
     1.0472,  4.5379,  0.5000,  # RL
     1.0472,  4.5379,  0.5000   # RR
], dtype=np.float32)

# Belt coupling Jacobian (4 legs, 3x3 block diagonal)
J_LEG = np.array([[8.0, 0.0, 0.0],
                  [0.0, 8.0, 0.0],
                  [0.0, 8.0, 16.0]], dtype=np.float32)
J_COUPLING = np.kron(np.eye(4, dtype=np.float32), J_LEG)


class MoeStairController:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data

        # Load TorchScript model
        assert POLICY_PATH.exists(), f"Policy missing at {POLICY_PATH}"
        self.policy = torch.jit.load(str(POLICY_PATH)).eval()

        # Cache joint indices
        self.qpos_idx = np.array([model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in JOINTS])
        self.qvel_idx = np.array([model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in JOINTS])
        self.base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
        self.gyro_sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")
        self.gyro_adr = model.sensor_adr[self.gyro_sensor_id] if self.gyro_sensor_id >= 0 else -1

        # Control state
        self.kp = 20.0
        self.kd = 0.7
        self.previous_action = np.zeros(12, dtype=np.float32)
        self.target_q = STAND_Q.copy()
        
        # 5 frames of 45-dim observations (oldest-first)
        # history[0] is oldest, history[4] is newest
        self.history = np.zeros((5, 45), dtype=np.float32)
        self.initialized = False
        self.step_count = 0

    def reset(self, target_pos: tuple[float, float, float] | None = None, yaw: float = 0.0):
        """Reset robot pose and history buffer."""
        if target_pos is not None:
            # Set base position
            self.data.qpos[0:3] = target_pos
            # Set base orientation quaternion [w, x, y, z] in MuJoCo
            half_yaw = yaw * 0.5
            self.data.qpos[3:7] = [np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)]
            self.data.qvel[:6] = 0.0

        # Set nominal joints
        self.data.qpos[self.qpos_idx] = STAND_Q
        self.data.qvel[self.qvel_idx] = 0.0
        self.target_q = STAND_Q.copy()
        self.previous_action = np.zeros(12, dtype=np.float32)
        self.history.fill(0.0)
        self.initialized = False
        self.step_count = 0
        mujoco.mj_forward(self.model, self.data)

    def get_current_frame(self, command: tuple[float, float, float]) -> np.ndarray:
        """Construct the 45-dim observation frame from current sensors."""
        q = self.data.qpos[self.qpos_idx].astype(np.float32)
        dq = self.data.qvel[self.qvel_idx].astype(np.float32)

        # Base orientation & projected gravity
        # In MuJoCo xmat is 9 floats (row-major 3x3 rotation matrix from body to world)
        rot = self.data.xmat[self.base_body_id].reshape(3, 3).astype(np.float32)
        # World gravity is [0, 0, -1]. Projected gravity in body frame: rot.T @ [0, 0, -1]
        gravity_proj = rot.T @ np.array([0.0, 0.0, -1.0], dtype=np.float32)

        # Angular velocity from IMU sensor or estimated from body velocity
        if self.gyro_adr >= 0:
            gyro = self.data.sensordata[self.gyro_adr:self.gyro_adr + 3].astype(np.float32)
        else:
            gyro = (rot.T @ self.data.cvel[self.base_body_id][:3]).astype(np.float32)

        cmd = np.array(command, dtype=np.float32)

        # Observation frame layout (45 values):
        # [0:3]   command (vx, vy, yaw_rate)
        # [3:6]   gyro * 0.25
        # [6:9]   projected gravity
        # [9:21]  joint position error: q - Q0
        # [21:33] joint velocity * 0.05
        # [33:45] previous raw action
        frame = np.concatenate([
            cmd,
            gyro * 0.25,
            gravity_proj,
            q - Q0,
            dq * 0.05,
            self.previous_action
        ]).astype(np.float32)
        return frame

    def policy_step(self, command: tuple[float, float, float]):
        """Run 50 Hz policy inference and compute target joint positions."""
        frame = self.get_current_frame(command)

        if not self.initialized:
            # Repeat first valid frame 5 times
            self.history = np.tile(frame, (5, 1))
            self.initialized = True
        else:
            # Roll buffer: shift left by 1 and append newest at index 4 (oldest-first)
            self.history[:-1] = self.history[1:]
            self.history[-1] = frame

        # Inference with TorchScript model expecting [1, 225] oldest-first
        obs_tensor = torch.from_numpy(self.history.flatten()).unsqueeze(0)
        with torch.no_grad():
            raw_action = self.policy(obs_tensor).squeeze(0).numpy().astype(np.float32)

        if not np.all(np.isfinite(raw_action)):
            raw_action = np.zeros(12, dtype=np.float32)

        # Action clipping & scaling
        clipped_action = np.clip(raw_action, -6.0, 6.0)
        self.previous_action = clipped_action.copy()

        # Compute and clamp target positions
        scaled_target = Q0 + SCALE * clipped_action
        self.target_q = np.clip(scaled_target, TARGET_MIN, TARGET_MAX)

    def physics_step(self):
        """Execute one 200 Hz physics step with coupled belt-drive PD torque."""
        q = self.data.qpos[self.qpos_idx].astype(np.float32)
        dq = self.data.qvel[self.qvel_idx].astype(np.float32)

        # Coupled motor-space PD:
        # tau_motor = (kp/gear^2) * (J @ (target - q)) - (kd/gear^2) * (J @ dq)
        # tau_joint = tau_motor @ J^T
        pos_err_motor = J_COUPLING @ (self.target_q - q)
        vel_motor = J_COUPLING @ dq

        tau_motor = (self.kp / (GEAR ** 2)) * pos_err_motor - (self.kd / (GEAR ** 2)) * vel_motor
        tau_joint = tau_motor @ J_COUPLING

        # Clamp to hardware torque limits
        tau_clamped = np.clip(tau_joint, -TORQUE_LIMITS, TORQUE_LIMITS)
        self.data.ctrl[:] = tau_clamped

        mujoco.mj_step(self.model, self.data)
        self.step_count += 1
