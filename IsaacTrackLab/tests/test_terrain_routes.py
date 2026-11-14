import json
from math import isfinite
from pathlib import Path
import unittest

from isaac_track_lab.task.terrain_routes import Obstacle2D, TerrainRouteGenerator, terrain_height


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/outdoor_map_batch.json").read_text(encoding="utf-8"))


class OutdoorTerrainRouteTest(unittest.TestCase):
    def test_outdoor_batch_has_six_distinct_static_training_maps(self):
        maps = CONFIG["maps"]
        self.assertEqual(len(maps), 6)
        self.assertEqual(len({item["id"] for item in maps}), 6)
        self.assertEqual(len({item["terrain"]["profile"] for item in maps}), 6)
        self.assertEqual(CONFIG["quality_contract"]["dynamic_events"], [])
        self.assertEqual(CONFIG["quality_contract"]["distractor_people"], 0)
        for item in maps:
            with self.subTest(map=item["id"]):
                self.assertTrue(item["obstacle_templates"])
                self.assertTrue(item["challenge_zones"])
                self.assertGreaterEqual(len(item["route_gates"]), 4)
                self.assertGreater(item["terrain"]["static_friction"], item["terrain"]["dynamic_friction"])
                self.assertGreater(item["terrain"]["dynamic_friction"], 0)
                for zone in item["challenge_zones"]:
                    if zone["traversable"]:
                        self.assertGreater(zone["static_friction"], zone["dynamic_friction"])
                        self.assertGreater(zone["dynamic_friction"], 0)

    def test_seeded_routes_are_reproducible_and_meet_contract(self):
        obstacle = Obstacle2D("test_obstacle", 0.0, 0.0, 3.0)
        defaults = CONFIG["route_defaults"]
        for map_config in CONFIG["maps"]:
            with self.subTest(map=map_config["id"]):
                generator = TerrainRouteGenerator(map_config, defaults, [obstacle])
                first = generator.generate(int(map_config["seed"]))
                second = generator.generate(int(map_config["seed"]))
                self.assertEqual(first, second)
                self.assertGreaterEqual(first.length_m, defaults["minimum_length_m"])
                self.assertLessEqual(first.length_m, defaults["maximum_length_m"])
                self.assertLessEqual(first.maximum_slope_deg, defaults["maximum_slope_deg"])
                self.assertGreaterEqual(first.minimum_obstacle_clearance_m, defaults["robot_clearance_m"])
                self.assertGreater(len(first.points_m), 200)
                self.assertTrue(all(isfinite(value) for point in first.points_m for value in point))

    def test_height_profiles_are_deterministic_and_finite(self):
        locations = ((-80, -70), (0, 0), (73, 55))
        for map_config in CONFIG["maps"]:
            with self.subTest(map=map_config["id"]):
                samples = [terrain_height(map_config, x, y) for x, y in locations]
                self.assertEqual(samples, [terrain_height(map_config, x, y) for x, y in locations])
                self.assertTrue(all(isfinite(value) for value in samples))


if __name__ == "__main__":
    unittest.main()
