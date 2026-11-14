import json
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

from isaac_track_lab.envs import ActorObservation
from isaac_track_lab.sim2real import (
    D455Nominal,
    D455PostProcessor,
    SensorControlTimeline,
    TemporalCameraVibration,
    build_episode_randomization,
    load_profile,
)
from isaac_track_lab.sim2real.events import ScenarioEventRuntime
from isaac_track_lab.task.scenario import DisturbanceEvent

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from validate_sim2real_report import EXPECTED_D455, validate_report


ROOT = Path(__file__).resolve().parents[1]


class Sim2RealPipelineTest(unittest.TestCase):
    def test_frozen_d455_values_are_exact(self) -> None:
        nominal = D455Nominal.from_calibration(ROOT / "configs/d455_sim_calibration.json")
        self.assertEqual(nominal, D455Nominal())
        self.assertEqual(nominal.body_relative_translation_m, (0.32, 0.0, 0.32))
        self.assertEqual(
            nominal.body_relative_quaternion_xyzw_ros,
            (0.54167522, -0.54167522, 0.45451948, -0.45451948),
        )

    def test_seeded_plan_is_reproducible_and_legal(self) -> None:
        first = build_episode_randomization(
            load_profile("sim2real_v1", seed=1234), root=ROOT, distractor_range=(2, 5)
        )
        second = build_episode_randomization(
            load_profile("sim2real_v1", seed=1234), root=ROOT, distractor_range=(2, 5)
        )
        different = build_episode_randomization(
            load_profile("sim2real_v1", seed=1235), root=ROOT, distractor_range=(2, 5)
        )
        self.assertEqual(first, second)
        self.assertNotEqual(first, different)
        self.assertIn(first.distractor_count, range(2, 6))
        self.assertTrue(0.0 <= first.materials["road"]["wetness"] <= 0.55)
        self.assertTrue(first.materials["road"]["roughness"] >= 0.48)

        city = build_episode_randomization(
            load_profile("sim2real_v1", seed=1234), root=ROOT,
            distractor_range=(0, 0), map_id="city_01",
        )
        self.assertIn("urban_street_01", city.hdri)

    def test_depth_mask_quantization_and_repeatability(self) -> None:
        profile = load_profile("sim2real_v1", seed=41)
        rgb = torch.full((32, 48, 3), 127, dtype=torch.uint8)
        depth = torch.linspace(0.1, 7.0, 32 * 48).reshape(32, 48)
        a = D455PostProcessor(profile).process(rgb, depth)
        b = D455PostProcessor(profile).process(rgb, depth)
        torch.testing.assert_close(a.rgb, b.rgb)
        torch.testing.assert_close(a.depth_m, b.depth_m, equal_nan=True)
        self.assertTrue(torch.equal(a.depth_valid_mask, b.depth_valid_mask))
        self.assertGreater(float((~a.depth_valid_mask).float().mean()), 0.05)
        valid_values = a.depth_m[a.depth_valid_mask]
        quantization_error = torch.abs(valid_values / 0.001 - torch.round(valid_values / 0.001))
        self.assertLess(float(quantization_error.max()), 0.001)

    def test_timeline_is_causal_and_drop_repeats_old_frame(self) -> None:
        profile = load_profile("sim2real_v1", seed=9)
        timing = dict(profile.section("timing"))
        timing["rgb"] = 10.0
        timing["depth"] = 20.0
        timing["rgb_drop_probability"] = 0.0
        timing["depth_drop_probability"] = 0.0
        line = SensorControlTimeline(timing, 9, profile.enabled)
        line.capture("rgb", 0.0, "old")
        line.capture("depth", 0.0, "depth-old")
        line.capture("rgb", 0.1, "future")
        # Depth has its own latency plus the configured one-frame lag.
        self.assertNotIn("depth", line.observation(0.03))
        self.assertEqual(line.observation(0.05)["rgb"].payload, "old")
        self.assertEqual(line.observation(0.055)["depth"].payload, "depth-old")
        self.assertEqual(line.observation(0.105)["rgb"].payload, "old")
        self.assertEqual(line.observation(0.111)["rgb"].payload, "future")

        timing["rgb_drop_probability"] = 1.0
        line = SensorControlTimeline(timing, 9, profile.enabled)
        line.capture("rgb", 0.0, "first")
        self.assertEqual(line.observation(0.02)["rgb"].payload, "first")
        repeated = line.capture("rgb", 0.1, "must_not_be_seen")
        self.assertTrue(repeated.repeated)
        self.assertEqual(line.observation(0.12)["rgb"].payload, "first")

    def test_action_latency_and_hold(self) -> None:
        profile = load_profile("sim2real_v1", seed=77)
        timing = dict(profile.section("timing"))
        timing.update({"inference": 10.0, "command": 5.0, "actuation": 5.0,
                       "missed_policy_update_probability": 0.0})
        line = SensorControlTimeline(timing, 77, profile.enabled)
        line.schedule_policy_command(1.0, (1.0, 0.0, 0.2))
        self.assertEqual(line.command(1.019), (0.0, 0.0, 0.0))
        self.assertEqual(line.command(1.020), (1.0, 0.0, 0.2))
        self.assertEqual(line.command(1.5), (1.0, 0.0, 0.2))

    def test_timeline_report_contains_actual_depth_skew_and_timestamps(self) -> None:
        profile = load_profile("sim2real_v1", seed=15)
        timing = dict(profile.section("timing"))
        timing.update({
            "rgb": 10.0, "depth": 20.0,
            "rgb_drop_probability": 0.0, "depth_drop_probability": 0.0,
        })
        line = SensorControlTimeline(timing, 15, profile.enabled)
        line.capture("rgb", 0.0, "rgb")
        line.capture("depth", 0.0, "depth")
        line.schedule_policy_command(
            0.1, (0.2, 0.0, 0.0), measured_inference_ms=12.5
        )
        report = line.report()
        self.assertAlmostEqual(
            report["latency"]["depth"]["samples_ms"][0],
            20.0 + float(timing["depth_extra_delay_ms"]),
        )
        self.assertEqual(len(report["frame_timestamps"]), 2)
        self.assertEqual(
            report["policy_command_timestamps"][0]["inference_ms"], 12.5
        )

    def test_report_validator_rejects_truth_leak_and_accepts_actual_shape(self) -> None:
        report = {
            "backend": "IsaacTrackingBackend", "d455_frozen": dict(EXPECTED_D455),
            "actor_input_fields": [
                "rgb", "depth", "depth_valid_mask", "proprioception",
            ],
            "privileged_input_fields": ["target_world_pose"],
            "truth_leakage_detected": False,
            "timing": {
                "frame_timestamps": [
                    {"capture_time_s": 0.0, "available_time_s": 0.01},
                ],
                "policy_command_timestamps": [{
                    "policy_time_s": 0.0, "effective_time_s": 0.01,
                    "scheduled": True,
                }],
                "latency": {
                    "rgb": {"count": 1, "samples_ms": [10.0]},
                    "depth": {"count": 1, "samples_ms": [43.0]},
                },
            },
            "sensor_statistics": {
                "processed_frame_count": 1,
                "depth_invalid_fraction": {"samples": [0.1]},
            },
            "events": [], "executed_event_count": 0,
        }
        self.assertEqual(validate_report(report), [])
        report["actor_input_fields"].append("target_world_pose")
        self.assertTrue(validate_report(report))

    def test_vibration_is_temporally_correlated_not_white_noise(self) -> None:
        profile = load_profile("sim2real_v1", seed=3)
        process = TemporalCameraVibration(profile.section("camera_vibration"), 3)
        values = np.asarray([
            process.step(1.0 / 30.0, base_angular_speed_rps=0.4)[0]
            for _ in range(240)
        ])
        lag_one = np.corrcoef(values[:-1, 0], values[1:, 0])[0, 1]
        self.assertGreater(lag_one, 0.2)

    def test_events_are_continuously_executed_and_recorded(self) -> None:
        runtime = ScenarioEventRuntime([
            DisturbanceEvent("lighting_change", "backlight", 1.0, 2.0),
            DisturbanceEvent("sensor_degradation", "depth_dropout", 2.0, 1.0),
            DisturbanceEvent("occlusion", "crossing_person", 3.0, 2.0),
        ])
        active_weights = []
        for step in range(61):
            active_weights.extend(weight for _, weight, _ in runtime.update(step * 0.1))
        report = runtime.report()
        self.assertTrue(all(row["apply_count"] > 0 for row in report))
        self.assertTrue(any(0.0 < value < 1.0 for value in active_weights))
        self.assertTrue(all(row["maximum_weight"] > 0.9 for row in report))

    def test_actor_contract_has_no_privileged_truth(self) -> None:
        forbidden = {
            "target_world_pose", "robot_world_pose", "ground_truth_bbox",
            "scheduled_target_bearing", "route_progress", "semantic_segmentation",
            "simulator_object_id",
        }
        self.assertTrue(forbidden.isdisjoint(ActorObservation.keys()))

    def test_nominal_profile_disables_every_randomization(self) -> None:
        profile = load_profile("sim2real_nominal", seed=1)
        self.assertFalse(any(profile.enabled.values()))
        plan = build_episode_randomization(profile, root=ROOT, distractor_range=(3, 8))
        self.assertEqual(plan.distractor_count, 0)
        self.assertEqual(plan.mount_translation_bias_m, (0.0, 0.0, 0.0))
        self.assertEqual(plan.exposure_ev, 0.0)


if __name__ == "__main__":
    unittest.main()
