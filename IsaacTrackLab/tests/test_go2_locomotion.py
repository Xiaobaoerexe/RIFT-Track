from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))

from isaac_track_lab.control.go2_locomotion import Go2LocomotionPolicy


class Go2LocomotionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        policy_root = ROOT.parent / "MaSTrack/robot_runtime/models/locomotion_candidates/kaiwu2025_v6-2_124004"
        cls.policy = Go2LocomotionPolicy(
            policy_root / "exported/policy.onnx",
            policy_root / "params/deploy.yaml",
        )

    def setUp(self):
        self.policy.reset()

    def test_deployment_contract_and_zero_command(self):
        output = self.policy.step(
            np.zeros(3), np.array([0.0, 0.0, -1.0]), np.zeros(3),
            self.policy.default_joint_positions, np.zeros(12),
        )
        self.assertEqual(output.raw_actions.shape, (12,))
        self.assertEqual(output.joint_position_targets.shape, (12,))
        self.assertTrue(np.isfinite(output.joint_position_targets).all())

    def test_history_layout_is_225_values(self):
        observation = self.policy.build_observation(
            np.zeros(3), np.array([0.0, 0.0, -1.0]), np.array([0.2, 0.0, 0.1]),
            self.policy.default_joint_positions, np.zeros(12),
        )
        self.assertEqual(observation.shape, (225,))
        # Five command frames, each scaled by [2, 2, 0.25].
        np.testing.assert_allclose(
            observation[30:45].reshape(5, 3),
            np.tile([0.4, 0.0, 0.025], (5, 1)),
        )


if __name__ == "__main__":
    unittest.main()
