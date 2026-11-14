import json
from pathlib import Path
import unittest

from isaac_track_lab.task import (
    TerrainTargetFollower,
    terrain_mesh_height,
    traversable_feature_surface_height,
)


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
            surface = traversable_feature_surface_height(
                self.map_config, x_m, y_m,
                base_height_m=terrain_mesh_height(self.map_config, x_m, y_m),
            )
            self.assertAlmostEqual(pose.walkable_surface_z_m, surface, places=8)
            self.assertGreaterEqual(z_m + 1e-8, surface)
            if not pose.completed:
                self.assertGreaterEqual(pose.speed_mps, 1.0)
                self.assertLessEqual(pose.speed_mps, 1.5)
            yaw_delta = abs((pose.yaw_deg - previous_yaw + 180.0) % 360.0 - 180.0)
            self.assertLessEqual(yaw_delta, 4.0 + 1e-8)
            previous_yaw = pose.yaw_deg

    def test_required_features_use_cautious_continuous_non_penetrating_motion(self):
        follower = TerrainTargetFollower(
            self.map_config, self.points, traversal_speed_mps=0.65
        )
        pose = follower.reset()
        previous_z = pose.position_m[2]
        modes: set[str] = set()
        visited: set[str] = set()
        for _ in range(20_000):
            pose = follower.step(1.0 / 30.0)
            modes.add(pose.locomotion_mode)
            if pose.traversable_feature_id is not None:
                visited.add(pose.traversable_feature_id)
            self.assertGreaterEqual(
                pose.position_m[2] + 1e-8, pose.walkable_surface_z_m
            )
            self.assertLessEqual(
                abs(pose.position_m[2] - previous_z), 0.75 / 30.0 + 1e-8
            )
            if pose.locomotion_mode in {"step_up", "traversable_surface"}:
                self.assertLessEqual(pose.speed_mps, 0.65 + 1e-8)
            previous_z = pose.position_m[2]
            if pose.completed:
                break
        self.assertTrue(pose.completed)
        self.assertTrue({"step_up", "step_down", "traversable_surface"} <= modes)
        required = {
            feature["id"]
            for feature in self.map_config["traversable_step_features"]
            if feature.get("required_on_route")
        }
        self.assertTrue(required <= visited)

    def test_invalid_speed_contract_is_rejected(self):
        with self.assertRaises(ValueError):
            TerrainTargetFollower(
                self.map_config, self.points,
                nominal_speed_mps=0.9, minimum_speed_mps=1.0,
            )


if __name__ == "__main__":
    unittest.main()
