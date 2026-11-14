import unittest

import numpy as np

from isaac_track_lab.control.velocity import VelocityCommand
from isaac_track_lab.tracking.recovery import RecoveryState, VisualRecoveryController
from isaac_track_lab.tracking.mast import MaSTTemplateTracker
from isaac_track_lab.tracking.appearance import TemplateHistogramVerifier


class VisualRecoveryTest(unittest.TestCase):
    def test_template_appearance_rejects_different_object(self) -> None:
        red = np.zeros((32, 16, 3), dtype=np.uint8)
        red[:, :, 0] = 220
        blue = np.zeros((64, 64, 3), dtype=np.uint8)
        blue[:, :, 2] = 220
        verifier = TemplateHistogramVerifier(red)
        same = verifier.similarity(red, (0, 0, 16, 32))
        different = verifier.similarity(blue, (0, 0, 64, 64))
        self.assertGreater(same, 0.99)
        self.assertLess(different, 0.2)

    def test_template_derived_locator_finds_matching_colour_without_fixed_hue(self) -> None:
        template = np.zeros((60, 24, 3), dtype=np.uint8)
        template[:, :] = (220, 25, 40)
        frame = np.full((160, 240, 3), 95, dtype=np.uint8)
        frame[50:125, 150:178] = (220, 25, 40)
        locator = TemplateHistogramVerifier(template)
        box = locator.locate(frame)
        self.assertIsNotNone(box)
        assert box is not None
        self.assertLessEqual(box[0], 150)
        self.assertGreaterEqual(box[2], 178)

    def test_template_derived_locator_rejects_unrelated_colour(self) -> None:
        template = np.zeros((60, 24, 3), dtype=np.uint8)
        template[:, :] = (220, 25, 40)
        frame = np.full((160, 240, 3), 95, dtype=np.uint8)
        frame[50:125, 150:178] = (20, 40, 220)
        self.assertIsNone(TemplateHistogramVerifier(template).locate(frame))

    def test_mast_geometry_allows_far_target_after_occlusion(self) -> None:
        tracker = object.__new__(MaSTTemplateTracker)
        tracker._initial_area = 8000.0
        tracker._initial_aspect = 0.4
        tracker._frame_shape = (480, 640)
        tracker._last_accepted_box = (280.0, 120.0, 320.0, 320.0)
        tracker._lost_frames = 3
        self.assertTrue(tracker._geometry_reliable((289.0, 222.5, 311.0, 277.5)))

    def test_mast_large_center_jump_is_recovery_only(self) -> None:
        tracker = object.__new__(MaSTTemplateTracker)
        tracker._initial_area = 8000.0
        tracker._initial_aspect = 0.4
        tracker._frame_shape = (480, 640)
        tracker._last_accepted_box = (40.0, 120.0, 80.0, 320.0)
        candidate = (430.0, 120.0, 470.0, 320.0)
        tracker._lost_frames = 0
        self.assertFalse(tracker._geometry_reliable(candidate))
        tracker._lost_frames = 3
        self.assertTrue(tracker._geometry_reliable(candidate))

    def test_loss_search_and_confirmed_reacquisition(self) -> None:
        recovery = VisualRecoveryController()
        self.assertEqual(recovery.update(0.1, False, 0.2), RecoveryState.TRACKING)
        self.assertEqual(recovery.update(0.2, False, None), RecoveryState.SEARCHING)
        search = recovery.command(VelocityCommand(1.2, 0.0, 0.0), 0.2)
        self.assertEqual(search.vx_mps, 0.0)
        self.assertLess(search.yaw_rate_rps, 0.0)
        self.assertEqual(recovery.update(1.0, True, -0.1), RecoveryState.VERIFYING)
        self.assertEqual(recovery.update(1.1, True, -0.1), RecoveryState.VERIFYING)
        self.assertEqual(recovery.update(1.2, True, -0.1), RecoveryState.TRACKING)
        metrics = recovery.metrics(1.2)
        self.assertEqual(metrics.loss_events, 1)
        self.assertEqual(metrics.reacquisitions, 1)
        self.assertAlmostEqual(metrics.total_lost_s, 1.0)

    def test_search_sweep_reverses_without_world_state(self) -> None:
        recovery = VisualRecoveryController()
        recovery.update(0.1, False, -0.3)
        recovery.update(0.2, False, None)
        visual = VelocityCommand(1.0, 0.0, 0.0)
        first = recovery.command(visual, 0.3)
        second = recovery.command(visual, 1.8)
        self.assertGreater(first.yaw_rate_rps, 0.0)
        self.assertLess(second.yaw_rate_rps, 0.0)
        self.assertEqual(first.vx_mps, 0.0)
        self.assertEqual(second.vx_mps, 0.0)

    def test_image_right_maps_to_negative_go2_yaw_during_search(self) -> None:
        recovery = VisualRecoveryController()
        recovery.update(0.1, True, 0.4)
        recovery.update(0.2, False, None)
        recovery.update(0.3, False, None)
        command = recovery.command(VelocityCommand(0.0, 0.0, 0.0), 0.3)
        self.assertLess(command.yaw_rate_rps, 0.0)


if __name__ == "__main__":
    unittest.main()
