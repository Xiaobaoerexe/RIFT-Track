import json
from math import atan2, degrees
from pathlib import Path
import unittest

from isaac_track_lab.task.terrain_routes import TerrainRouteGenerator, terrain_height


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/terrain_dataset_v3_realistic.json"
SCENES = ROOT / "assets/scenes/terrain_dataset_v3_realistic"


class RealisticTerrainDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = json.loads(CONFIG.read_text(encoding="utf-8"))

    def test_catalog_has_three_balanced_families(self):
        self.assertEqual(len(self.catalog["maps"]), 60)
        for family in ("city", "park", "forest_trail"):
            self.assertEqual(
                sum(item["family"] == family for item in self.catalog["maps"]), 20
            )

    def test_semantic_family_contracts(self):
        city = next(item for item in self.catalog["maps"] if item["family"] == "city")
        park = next(item for item in self.catalog["maps"] if item["family"] == "park")
        forest = next(item for item in self.catalog["maps"] if item["family"] == "forest_trail")
        self.assertGreaterEqual(
            sum(item.get("kind") == "road" for item in city["surface_corridors"]), 6
        )
        self.assertGreaterEqual(len(city["curb_segments"]), 40)
        self.assertTrue(any(zone.get("shape") == "ellipse" for zone in park["challenge_zones"]))
        self.assertTrue(any(feature["kind"] == "lake_basin"
                            for feature in park["terrain"]["features"]))
        self.assertGreaterEqual(
            sum(item["asset"] == "pine_tree_01" for item in forest["obstacle_instances"]), 90
        )
        self.assertGreaterEqual(len(forest["traversable_step_features"]), 5)

    def test_lake_is_a_bowl_not_a_pasted_plane(self):
        park = next(item for item in self.catalog["maps"] if item["family"] == "park")
        lake = next(feature for feature in park["terrain"]["features"]
                    if feature["kind"] == "lake_basin")
        cx, cy = lake["center_m"]
        center = terrain_height(park, cx, cy)
        shore = terrain_height(park, cx + lake["radius_x_m"], cy)
        self.assertGreater(shore - center, 0.7)

    def test_all_authored_routes_stay_in_navigation_corridors(self):
        for map_config in self.catalog["maps"]:
            route_path = SCENES / map_config["id"] / "routes/generated_routes.json"
            routes = json.loads(route_path.read_text(encoding="utf-8"))["routes"]
            self.assertEqual(len(routes), 20)
            for route in routes:
                self.assertTrue(280.0 <= float(route["length_m"]) <= 360.0)
                self.assertLessEqual(float(route["maximum_turn_deg"]), 150.0 + 1e-6)
                for x_m, y_m, _ in route["points_m"]:
                    self.assertTrue(any(
                        TerrainRouteGenerator._inside_corridor(x_m, y_m, corridor)
                        for corridor in map_config["navigation_corridors"]
                    ))

    def test_baseline_maps_have_collidable_route_mandatory_gait_courses(self):
        expected_kinds = {"curb_table", "platform", "stair_platform"}
        for map_id in ("city_01", "park_01", "forest_trail_01"):
            with self.subTest(map=map_id):
                map_config = next(
                    item for item in self.catalog["maps"] if item["id"] == map_id
                )
                required = [
                    feature for feature in map_config["traversable_step_features"]
                    if feature.get("required_on_route")
                ]
                self.assertEqual({feature["kind"] for feature in required}, expected_kinds)
                required_ids = {feature["id"] for feature in required}
                for feature in required:
                    self.assertGreaterEqual(float(feature["width_m"]), 2.5)
                    if feature["kind"] == "stair_platform":
                        self.assertTrue(0.08 <= float(feature["step_rise_m"]) <= 0.10)
                        self.assertGreaterEqual(float(feature["tread_depth_m"]), 0.45)
                        self.assertLessEqual(
                            int(feature["step_count"]) * float(feature["step_rise_m"]),
                            0.30,
                        )
                    else:
                        self.assertLessEqual(float(feature["height_m"]), 0.18)

                manifest = json.loads(
                    (SCENES / map_id / "scene_manifest.json").read_text(encoding="utf-8")
                )
                authored_ids = {
                    item["feature_id"] for item in manifest["traversable_features"]
                    if "feature_id" in item
                }
                self.assertTrue(required_ids <= authored_ids)
                self.assertTrue(all(
                    item.get("collision_prim")
                    for item in manifest["traversable_features"]
                    if item.get("feature_id") in required_ids
                ))

                routes = json.loads(
                    (SCENES / map_id / "routes/generated_routes.json").read_text(
                        encoding="utf-8"
                    )
                )["routes"]
                for route in routes:
                    self.assertTrue(
                        required_ids <= set(route["traversable_features_visited"])
                    )
                    self.assertLessEqual(float(route["maximum_step_height_m"]), 0.30)
                    points = route["points_m"]
                    for feature in required:
                        closest = min(
                            range(len(points)),
                            key=lambda index: (
                                (points[index][0] - feature["center_m"][0]) ** 2
                                + (points[index][1] - feature["center_m"][1]) ** 2
                            ),
                        )
                        before = points[max(0, closest - 2)]
                        after = points[min(len(points) - 1, closest + 2)]
                        route_yaw = degrees(atan2(
                            after[1] - before[1], after[0] - before[0]
                        ))
                        error = abs(
                            (route_yaw - float(feature["yaw_deg"]) + 180.0)
                            % 360.0 - 180.0
                        )
                        self.assertLessEqual(min(error, 180.0 - error), 15.0)


if __name__ == "__main__":
    unittest.main()
