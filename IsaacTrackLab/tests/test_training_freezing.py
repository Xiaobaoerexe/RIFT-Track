import json
from pathlib import Path
import unittest

from isaac_track_lab.training import (
    MODULE_NAMES,
    TrainingPhase,
    apply_freeze_plan,
    load_training_phases,
    optimizer_parameters,
)


ROOT = Path(__file__).resolve().parents[1]


class FakeParameter:
    def __init__(self, elements: int) -> None:
        self.elements = elements
        self.requires_grad = True

    def requires_grad_(self, value: bool) -> "FakeParameter":
        self.requires_grad = value
        return self

    def numel(self) -> int:
        return self.elements


class FakeModule:
    def __init__(self, *sizes: int) -> None:
        self.items = [FakeParameter(size) for size in sizes]

    def parameters(self):
        return iter(self.items)


def fake_modules():
    return {
        "visual_encoder": FakeModule(10, 20),
        "algorithm_constrained_navigation": FakeModule(5),
        "locomotion": FakeModule(7, 8),
    }


class TrainingFreezingTest(unittest.TestCase):
    def test_every_configured_phase_is_complete_and_valid(self) -> None:
        phases = load_training_phases(ROOT / "configs/training_phases.json")
        self.assertEqual(
            set(phases),
            {"visual_only", "navigation_only", "locomotion_only", "visual_and_navigation", "end_to_end"},
        )
        for phase in phases.values():
            self.assertEqual(
                set(phase.trainable_modules) | set(phase.frozen_modules), set(MODULE_NAMES)
            )

    def test_freeze_plan_and_optimizer_selection_match(self) -> None:
        modules = fake_modules()
        phase = load_training_phases(ROOT / "configs/training_phases.json")["navigation_only"]
        audit = apply_freeze_plan(modules, phase)
        self.assertFalse(modules["visual_encoder"].items[0].requires_grad)
        self.assertTrue(modules["algorithm_constrained_navigation"].items[0].requires_grad)
        self.assertFalse(modules["locomotion"].items[0].requires_grad)
        self.assertEqual(optimizer_parameters(modules, phase), modules["algorithm_constrained_navigation"].items)
        self.assertEqual(audit.parameter_elements["visual_encoder"], 30)
        self.assertEqual(audit.module_trainable["locomotion"], False)

    def test_invalid_overlap_and_missing_module_are_rejected(self) -> None:
        overlap = TrainingPhase(
            "bad", ("visual_encoder",), ("visual_encoder", "locomotion")
        )
        with self.assertRaises(ValueError):
            overlap.validate()
        valid = load_training_phases(ROOT / "configs/training_phases.json")["visual_only"]
        modules = fake_modules()
        del modules["locomotion"]
        with self.assertRaises(ValueError):
            apply_freeze_plan(modules, valid)

    def test_training_phase_chain_matches_environment_contract(self) -> None:
        phase_config = json.loads(
            (ROOT / "configs/training_phases.json").read_text(encoding="utf-8")
        )
        environment = json.loads(
            (ROOT / "configs/training_environment.json").read_text(encoding="utf-8")
        )
        self.assertEqual(phase_config["interface_chain"], environment["interfaces"])


if __name__ == "__main__":
    unittest.main()
