import unittest

from isaac_track_lab.spec import load_json, validate


class SpecTest(unittest.TestCase):
    def test_spec_is_valid(self) -> None:
        self.assertEqual(validate(), [])

    def test_six_scenarios(self) -> None:
        data = load_json("scenarios.json")
        self.assertEqual(len(data["scenarios"]), 6)
        self.assertEqual({s["duration_s"] for s in data["scenarios"]}, {300, 3600})
        self.assertEqual(
            {s["interference"] for s in data["scenarios"]},
            {"none", "distractors", "distractors_occlusion"},
        )


if __name__ == "__main__":
    unittest.main()
