import unittest

from isaac_track_lab.task.motion import (
    OcclusionEvent,
    RouteFollower,
    inside_route_corridor,
    scheduled_visibility,
)
from isaac_track_lab.task.runtime import (
    EpisodeState,
    FailureReason,
    StepObservation,
    TaskEvaluator,
    TerminationThresholds,
)
from isaac_track_lab.task.scenario import build_scenario_plan


THRESHOLDS = TerminationThresholds(
    target_distance_m=12.0,
    target_distance_grace_s=2.0,
    target_absent_grace_s=3.0,
    minimum_bbox_area_ratio=0.001,
    base_contact_impulse_ns=120.0,
    maximum_tilt_deg=60.0,
    minimum_base_height_m=0.28,
    fallen_grace_s=0.5,
)


def observation(elapsed_s: float, **overrides: object) -> StepObservation:
    values: dict[str, object] = {
        "elapsed_s": elapsed_s,
        "target_distance_m": 3.0,
        "target_bbox_area_ratio": 0.01,
        "base_contact_impulse_ns": 0.0,
        "base_height_m": 0.42,
        "base_tilt_deg": 0.0,
        "in_bounds": True,
    }
    values.update(overrides)
    return StepObservation(**values)  # type: ignore[arg-type]


class TaskRuntimeTest(unittest.TestCase):
    def test_success_and_reacquisition_metrics(self) -> None:
        evaluator = TaskEvaluator(THRESHOLDS, horizon_s=4.0)
        evaluator.update(observation(1.0))
        evaluator.update(observation(2.0, target_bbox_area_ratio=0.0))
        evaluator.update(observation(3.0))
        result = evaluator.update(observation(4.0))
        self.assertEqual(result.state, EpisodeState.SUCCESS)
        self.assertAlmostEqual(result.tracking_coverage, 0.75)
        self.assertEqual(result.absence_events, 1)
        self.assertEqual(result.reacquisitions, 1)
        self.assertAlmostEqual(result.mean_reacquisition_time_s or 0.0, 1.0)

    def test_target_absent_failure(self) -> None:
        evaluator = TaskEvaluator(THRESHOLDS, horizon_s=20.0)
        result = evaluator.update(observation(3.0, target_bbox_area_ratio=0.0))
        self.assertEqual(result.failure_reason, FailureReason.TARGET_ABSENT)

    def test_target_too_far_requires_grace(self) -> None:
        evaluator = TaskEvaluator(THRESHOLDS, horizon_s=20.0)
        self.assertEqual(
            evaluator.update(observation(1.0, target_distance_m=13.0)).state,
            EpisodeState.RUNNING,
        )
        result = evaluator.update(observation(2.0, target_distance_m=13.0))
        self.assertEqual(result.failure_reason, FailureReason.TARGET_TOO_FAR)

    def test_immediate_safety_failures(self) -> None:
        cases = (
            ({"base_contact_impulse_ns": 120.0}, FailureReason.STRONG_COLLISION),
            ({"in_bounds": False}, FailureReason.OUT_OF_BOUNDS),
        )
        for values, reason in cases:
            with self.subTest(reason=reason):
                evaluator = TaskEvaluator(THRESHOLDS, horizon_s=20.0)
                result = evaluator.update(observation(0.1, **values))
                self.assertEqual(result.failure_reason, reason)

    def test_fall_requires_grace(self) -> None:
        evaluator = TaskEvaluator(THRESHOLDS, horizon_s=20.0)
        evaluator.update(observation(0.25, base_height_m=0.2))
        result = evaluator.update(observation(0.5, base_height_m=0.2))
        self.assertEqual(result.failure_reason, FailureReason.FALLEN)
        self.assertEqual(result.fall_count, 1)

    def test_failure_wins_at_horizon(self) -> None:
        evaluator = TaskEvaluator(THRESHOLDS, horizon_s=1.0)
        result = evaluator.update(observation(1.0, in_bounds=False))
        self.assertEqual(result.state, EpisodeState.FAILURE)


class MotionTest(unittest.TestCase):
    def test_route_follower_and_ping_pong(self) -> None:
        route = RouteFollower([(0, 0, 0), (2, 0, 0)], 1.0, ping_pong=True)
        self.assertEqual(route.position(1.0), (1.0, 0.0, 0.0))
        self.assertEqual(route.position(3.0), (1.0, 0.0, 0.0))

    def test_occlusion_schedule(self) -> None:
        events = [OcclusionEvent(2.0, 1.0)]
        self.assertTrue(scheduled_visibility(1.9, events))
        self.assertFalse(scheduled_visibility(2.5, events))
        self.assertTrue(scheduled_visibility(3.0, events))

    def test_route_corridor(self) -> None:
        waypoints = [(0, 0, 0), (10, 0, 0), (10, 10, 0)]
        self.assertTrue(inside_route_corridor((5, 2), waypoints, 3.0))
        self.assertFalse(inside_route_corridor((5, 4), waypoints, 3.0))

    def test_scenario_plan_is_deterministic_and_within_ranges(self) -> None:
        scenario = {
            "id": "short_distractors_occlusion",
            "duration_s": 300,
            "interference": "distractors_occlusion",
            "distractors": [3, 8],
            "forced_occlusions": [2, 6],
        }
        occlusion = {"duration_s": [1.0, 6.0]}
        first = build_scenario_plan(scenario, occlusion, seed=7)
        second = build_scenario_plan(scenario, occlusion, seed=7)
        self.assertEqual(first, second)
        self.assertIn(first.distractor_count, range(3, 9))
        self.assertIn(len(first.occlusion_events), range(2, 7))
        self.assertTrue(
            all(1.0 <= event.duration_s <= 6.0 for event in first.occlusion_events)
        )

    def test_one_hour_plan_has_checkpoints_and_reproducible_disturbances(self) -> None:
        scenario = {
            "id": "hour_distractors_occlusion",
            "duration_s": 3600,
            "interference": "distractors_occlusion",
            "distractors": [4, 12],
            "forced_occlusions": [12, 24],
            "appearance_changes": [4, 8],
            "lighting_changes": [6, 12],
            "sensor_degradations": [3, 8],
        }
        occlusion = {
            "duration_s": [1.0, 6.0],
            "types": ["static_occluder", "building_corner"],
            "reappearance": ["same_side", "opposite_side"],
        }
        disturbances = {
            "appearance_change": {"duration_s": [10.0, 45.0], "types": ["outerwear_change"]},
            "lighting_change": {"duration_s": [20.0, 90.0], "types": ["backlight"]},
            "sensor_degradation": {"duration_s": [2.0, 12.0], "types": ["motion_blur"]},
        }
        first = build_scenario_plan(scenario, occlusion, 41, disturbances)
        second = build_scenario_plan(scenario, occlusion, 41, disturbances)
        self.assertEqual(first, second)
        self.assertEqual(first.checkpoint_count, 12)
        kinds = {event.kind for event in first.disturbance_events}
        self.assertEqual(
            kinds,
            {"occlusion", "appearance_change", "lighting_change", "sensor_degradation"},
        )


if __name__ == "__main__":
    unittest.main()
