"""Headless automated verification test for SvanM2 MoE 90k stair simulation.

Tests:
1. Scene compilation and integrity (all 6 staircases, 72 steps total).
2. MoE 90k policy loading and inference sanity.
3. Stepping across all 6 tracks with forward locomotion.
4. Top platform turnaround and descent initial step.
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import mujoco

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from build_scene import build_scene_xml, TRACK_CONFIGS, NUM_STEPS
from controller import MoeStairController
from teleop import KeyboardTeleop


class TestSvanM2StairSim(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene_path = HERE / "scene.xml"
        build_scene_xml(cls.scene_path)
        cls.model = mujoco.MjModel.from_xml_path(str(cls.scene_path))
        cls.data = mujoco.MjData(cls.model)
        cls.ctrl = MoeStairController(cls.model, cls.data)
        cls.teleop = KeyboardTeleop(cls.ctrl, TRACK_CONFIGS)

    def test_01_scene_integrity(self):
        """Verify model geometry, joints, and actuators."""
        self.assertEqual(len(TRACK_CONFIGS), 6)
        self.assertEqual(self.model.nu, 12)
        self.assertGreaterEqual(self.model.ngeom, 150)
        # Check that steps exist for each track
        for t_idx in range(1, 7):
            for s in range(1, NUM_STEPS + 1):
                gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"step_{t_idx}_{s}")
                self.assertGreaterEqual(gid, 0, f"Missing step_{t_idx}_{s}")
            plat_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"platform_{t_idx}")
            self.assertGreaterEqual(plat_id, 0, f"Missing platform_{t_idx}")

    def test_02_standing_stability(self):
        """Verify standing stability at zero command for 1 second."""
        self.teleop.teleport_to_track(1, on_top=False)
        self.teleop.stop()

        for step in range(200): # 1.0s of physics
            if step % 4 == 0:
                self.ctrl.policy_step(self.teleop.command)
            self.ctrl.physics_step()

        self.assertTrue(np.all(np.isfinite(self.data.qpos)))
        self.assertTrue(np.all(np.isfinite(self.data.qvel)))
        # Height should remain upright (approx 0.28m to 0.35m)
        self.assertGreater(self.data.qpos[2], 0.24)
        rot = self.data.xmat[self.ctrl.base_body_id].reshape(3, 3)
        tilt = np.degrees(np.arccos(np.clip(rot[2, 2], -1.0, 1.0)))
        self.assertLess(tilt, 15.0, f"Base tilt too high during standing: {tilt:.1f}°")

    def test_03_forward_locomotion_on_all_tracks(self):
        """Verify the robot steps forward towards stairs on each of the 6 tracks."""
        for t_idx in range(1, 7):
            self.teleop.teleport_to_track(t_idx, on_top=False)
            initial_x = float(self.data.qpos[0])
            self.teleop.handle_key(ord('W'))
            self.teleop.handle_key(ord('W'))
            self.teleop.handle_key(ord('W')) # 0.15 m/s

            # Step 1.0 second (50 policy steps, 200 physics steps)
            for step in range(200):
                if step % 4 == 0:
                    self.ctrl.policy_step(self.teleop.command)
                self.ctrl.physics_step()

            self.assertTrue(np.all(np.isfinite(self.data.qpos)))
            # Must have advanced forward
            final_x = float(self.data.qpos[0])
            self.assertGreater(final_x, initial_x, f"Robot did not advance on Track {t_idx}")

    def test_04_top_platform_teleport_and_descent_setup(self):
        """Verify teleporting to top platform sets the correct height and yaw."""
        for t_idx in (1, 2, 4):
            cfg = TRACK_CONFIGS[t_idx - 1]
            self.teleop.teleport_to_track(t_idx, on_top=True)
            
            expected_z = 12 * cfg["height"] + 0.32
            actual_z = float(self.data.qpos[2])
            self.assertAlmostEqual(actual_z, expected_z, delta=0.05)

            # Check that yaw is facing -X (downstairs)
            # quaternion is [w, x, y, z] = [0, 0, 0, 1] for yaw = pi
            rot = self.data.xmat[self.ctrl.base_body_id].reshape(3, 3)
            heading_x = rot[0, 0] # should be negative (approx -1.0)
            self.assertLess(heading_x, -0.8, f"Robot not facing stairs on Track {t_idx}")


if __name__ == "__main__":
    unittest.main()
