import unittest

import numpy as np

from isaac_track_lab.envs import (
    ActorObservation,
    BackendTransition,
    EpisodeRequest,
    PlanarVelocityAction,
    PrivilegedSignals,
    StandardTrackingEnv,
)
from isaac_track_lab.task import TerminationThresholds


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


def actor(token: int = 0) -> ActorObservation:
    return ActorObservation(
        rgb=np.full((8, 8, 3), token, dtype=np.uint8),
        target_template_rgb=np.zeros((4, 4, 3), dtype=np.uint8),
        joint_position=(0.0,) * 12,
        joint_velocity=(0.0,) * 12,
        base_angular_velocity=(0.0, 0.0, 0.0),
        projected_gravity=(0.0, 0.0, -1.0),
        foot_contacts=(1.0, 1.0, 1.0, 1.0),
    )


def privileged(**overrides: object) -> PrivilegedSignals:
    values: dict[str, object] = {
        "target_distance_m": 3.0,
        "target_bbox_area_ratio": 0.01,
        "base_contact_impulse_ns": 0.0,
        "base_height_m": 0.42,
        "base_tilt_deg": 0.0,
        "in_bounds": True,
    }
    values.update(overrides)
    return PrivilegedSignals(**values)  # type: ignore[arg-type]


class DeterministicBackend:
    def __init__(self) -> None:
        self.request: EpisodeRequest | None = None
        self.elapsed_s = 0.0

    def reset(self, request: EpisodeRequest) -> BackendTransition:
        self.request = request
        self.elapsed_s = 0.0
        return BackendTransition(actor(request.seed % 256), privileged(), 0.0)

    def step(self, action: PlanarVelocityAction) -> BackendTransition:
        self.elapsed_s += 1.0
        assert self.request is not None
        return BackendTransition(
            actor((self.request.seed + round(self.elapsed_s)) % 256),
            privileged(),
            self.elapsed_s,
        )


class EnvironmentContractTest(unittest.TestCase):
    def test_actor_type_contains_no_privileged_fields(self) -> None:
        self.assertTrue({
            "target_world_pose",
            "robot_world_pose",
            "ground_truth_bbox",
            "scheduled_target_bearing",
        }.isdisjoint(ActorObservation.keys()))

    def test_reset_is_seed_reproducible_and_returns_no_truth(self) -> None:
        request = EpisodeRequest("baseline_305m", seed=17, horizon_s=2.0)
        first, first_info = StandardTrackingEnv(
            DeterministicBackend(), THRESHOLDS
        ).reset(request)
        second, second_info = StandardTrackingEnv(
            DeterministicBackend(), THRESHOLDS
        ).reset(request)
        np.testing.assert_array_equal(first.rgb, second.rgb)
        self.assertEqual(first_info, second_info)
        self.assertNotIn("privileged", first_info)

    def test_step_uses_truth_only_for_reward_and_evaluation(self) -> None:
        env = StandardTrackingEnv(DeterministicBackend(), THRESHOLDS)
        env.reset(EpisodeRequest("baseline_305m", seed=4, horizon_s=2.0))
        action = PlanarVelocityAction(1.0, 0.0, 0.0)
        _, reward, terminated, truncated, info = env.step(action)
        self.assertEqual(reward, 1.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        _, _, terminated, truncated, info = env.step(action)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info["episode_result"]["state"], "success")
        self.assertNotIn("privileged", info)

    def test_step_before_reset_is_rejected(self) -> None:
        env = StandardTrackingEnv(DeterministicBackend(), THRESHOLDS)
        with self.assertRaises(RuntimeError):
            env.step(PlanarVelocityAction(0.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
