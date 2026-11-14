from __future__ import annotations

import json
from pathlib import Path
import unittest

from isaac_track_lab.task.terrain_routes import TerrainRouteGenerator


ROOT = Path(__file__).resolve().parents[1]


class TerrainDatasetV2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = json.loads(
            (ROOT / "configs/terrain_dataset_v2_catalog.json").read_text(encoding="utf-8")
        )

    def test_catalog_has_three_balanced_families(self) -> None:
        maps = self.catalog["maps"]
        self.assertEqual(len(maps), 60)
        self.assertEqual(len({item["id"] for item in maps}), 60)
        for family in ("city", "park", "forest_trail"):
            self.assertEqual(sum(item["family"] == family for item in maps), 20)

    def test_maps_have_seeded_rugged_surfaces_and_route_variants(self) -> None:
        for item in self.catalog["maps"]:
            terrain = item["terrain"]
            self.assertEqual(terrain["seed"], item["seed"])
            self.assertGreater(terrain["roughness_m"], 0.0)
            self.assertEqual(len(terrain["features"]), 4)
            self.assertEqual(len(item["route_gate_sets"]), 5)
            self.assertLessEqual(
                item["target_maximum_slope_deg"],
                self.catalog["route_defaults"]["maximum_slope_deg"],
            )

    def test_one_target_route_per_family_is_reachable(self) -> None:
        for family in ("city", "park", "forest_trail"):
            item = next(entry for entry in self.catalog["maps"] if entry["family"] == family)
            route = TerrainRouteGenerator(item, self.catalog["route_defaults"]).generate(item["seed"])
            self.assertTrue(route.target_reachable)
            self.assertLessEqual(route.maximum_slope_deg, item["target_maximum_slope_deg"])

    def test_forbidden_zone_segment_intersection_is_exact(self) -> None:
        self.assertTrue(TerrainRouteGenerator._segment_intersects_box((-2, 0), (2, 0), 0, 0, 1, 1))
        self.assertFalse(TerrainRouteGenerator._segment_intersects_box((-2, 1.1), (2, 1.1), 0, 0, 1, 1))


if __name__ == "__main__":
    unittest.main()
