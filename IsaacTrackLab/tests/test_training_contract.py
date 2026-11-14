import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TrainingEnvironmentContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.environment = json.loads(
            (ROOT / "configs/training_environment.json").read_text()
        )
        cls.baseline = json.loads(
            (ROOT / "configs/baseline_305m_v1.json").read_text()
        )

    def test_pure_actor_contract_excludes_all_simulator_truth(self) -> None:
        pure = self.environment["control_modes"]["pure_e2e"]
        actor = set(pure["actor_inputs"] + pure["optional_actor_inputs"])
        forbidden = set(pure["forbidden_actor_inputs"])
        self.assertTrue(actor.isdisjoint(forbidden))
        self.assertTrue({
            "target_world_pose",
            "robot_world_pose",
            "scheduled_target_bearing",
            "ground_truth_bbox",
            "appearance_recovery_bbox",
        }.issubset(forbidden))

    def test_showcase_privilege_is_explicit_and_not_e2e(self) -> None:
        hybrid = self.environment["control_modes"]["showcase_hybrid"]
        self.assertFalse(hybrid["may_be_reported_as_pure_e2e"])
        self.assertEqual(
            set(hybrid["privileged_actor_inputs"]),
            {"scheduled_target_bearing", "purple_vest_recovery_bbox"},
        )

    def test_frozen_baseline_matches_official_result(self) -> None:
        result = json.loads(
            (ROOT / "tests/fixtures/baseline_305m_official.json").read_text()
        )
        expected = self.baseline["measured_result"]
        self.assertEqual(result["state"], "success")
        self.assertGreaterEqual(result["route_distance_completed_m"], 305.0)
        for key, value in expected.items():
            self.assertEqual(result[key], value, key)


if __name__ == "__main__":
    unittest.main()
