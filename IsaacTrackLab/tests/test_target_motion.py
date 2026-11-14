import json
from pathlib import Path
import unittest

from isaac_track_lab.task import TerrainTargetFollower, terrain_mesh_height


ROOT = Path(__file__).resolve().parents[1]


class TerrainTargetFollowerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        catalog = json.loads(
            (ROOT / "configs/terrain_dataset_v3_realistic.json").read_text()
        )
        cls.map_config = next(item for item in catalog["maps"] if item["id"] == "forest_trail_01")
        route_data = json.loads(
            (ROOT / "assets/scenes/terrain_dataset_v3_realistic/forest_trail_01/routes/generated_routes.json").read_text()
        )
        cls.points = route_data["routes"][0]["points_m"]

    def test_motion_is_grounded_and_speed_limited(self):
        follower = TerrainTargetFollower(self.map_config, self.points)
        previous_yaw = follower.yaw_deg
        for _ in range(600):
            pose = follower.step(1.0 / 30.0)
            x_m, y_m, z_m = pose.position_m
            self.assertAlmostEqual(z_m, terrain_mesh_height(self.map_config, x_m, y_m), places=8)
            if not pose.completed:
                self.assertGreaterEqual(pose.speed_mps, 1.0)
                self.assertLessEqual(pose.speed_mps, 1.5)
            yaw_delta = abs((pose.yaw_deg - previous_yaw + 180.0) % 360.0 - 180.0)
            self.assertLessEqual(yaw_delta, 4.0 + 1e-8)
            previous_yaw = pose.yaw_deg

    def test_invalid_speed_contract_is_rejected(self):
        with self.assertRaises(ValueError):
            TerrainTargetFollower(
                self.map_config, self.points,
                nominal_speed_mps=0.9, minimum_speed_mps=1.0,
            )


if __name__ == "__main__":
    unittest.main()
