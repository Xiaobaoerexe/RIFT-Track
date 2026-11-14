import math
import unittest

from isaac_track_lab.control import (
    BBoxVelocityPolicy,
    PlanarVelocitySurrogate,
    Pose2D,
    VelocityCommand,
    VisualFollowConfig,
    TrackingControlMode,
    compose_tracking_command,
)


class VisualControlTest(unittest.TestCase):
    def test_centered_small_target_moves_forward(self) -> None:
        policy = BBoxVelocityPolicy(VisualFollowConfig(desired_area_ratio=0.02))
        command = policy.command((300, 180, 340, 300), 640, 480)
        self.assertGreater(command.vx_mps, 0.0)
        self.assertAlmostEqual(command.yaw_rate_rps, 0.0)

    def test_right_target_turns_right_and_lost_searches_same_way(self) -> None:
        policy = BBoxVelocityPolicy(VisualFollowConfig(desired_area_ratio=0.02))
        command = policy.command((450, 180, 500, 300), 640, 480)
        self.assertLess(command.yaw_rate_rps, 0.0)
        search = policy.command(None, 640, 480)
        self.assertLess(search.yaw_rate_rps, 0.0)
        self.assertGreater(search.vx_mps, 0.0)

    def test_planar_surrogate_acceleration_and_heading(self) -> None:
        adapter = PlanarVelocitySurrogate(Pose2D(0.0, 0.0, math.pi / 2.0))
        adapter.apply(VelocityCommand(0.5, 0.0, 0.0))
        pose = adapter.step(0.1)
        self.assertAlmostEqual(pose.x_m, 0.0, places=6)
        self.assertGreater(pose.y_m, 0.0)
        self.assertAlmostEqual(adapter.velocity.vx_mps, 0.1)

    def test_pure_e2e_never_uses_scheduled_target_bearing(self) -> None:
        visual = VelocityCommand(1.1, 0.0, -0.2)
        command = compose_tracking_command(
            mode=TrackingControlMode.PURE_E2E,
            visual_command=visual,
            robot_heading_rad=2.0,
            scheduled_target_bearing_rad=None,
            target_visible=False,
        )
        self.assertEqual(command, visual)
        self.assertFalse(TrackingControlMode.PURE_E2E.allows_privileged_recovery)

    def test_showcase_hybrid_requires_and_uses_route_bearing(self) -> None:
        visual = VelocityCommand(1.1, 0.0, 0.0)
        with self.assertRaises(ValueError):
            compose_tracking_command(
                mode=TrackingControlMode.SHOWCASE_HYBRID,
                visual_command=visual,
                robot_heading_rad=0.0,
                scheduled_target_bearing_rad=None,
                target_visible=False,
            )
        command = compose_tracking_command(
            mode=TrackingControlMode.SHOWCASE_HYBRID,
            visual_command=visual,
            robot_heading_rad=0.0,
            scheduled_target_bearing_rad=math.pi / 2.0,
            target_visible=False,
        )
        self.assertAlmostEqual(command.yaw_rate_rps, 0.9)
        self.assertTrue(TrackingControlMode.SHOWCASE_HYBRID.allows_privileged_recovery)


if __name__ == "__main__":
    unittest.main()
