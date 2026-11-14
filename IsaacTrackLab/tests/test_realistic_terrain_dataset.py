import json
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


if __name__ == "__main__":
    unittest.main()
