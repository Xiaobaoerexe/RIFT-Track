"""Framework-light freeze plans for the three-part tracking policy.

The helpers intentionally depend only on the small ``parameters`` and
``requires_grad`` surface shared by common tensor frameworks.  They configure
training ownership; they do not choose a network architecture, loss or
optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping


MODULE_NAMES = (
    "visual_encoder",
    "algorithm_constrained_navigation",
    "locomotion",
)
_MODULE_SET = frozenset(MODULE_NAMES)


@dataclass(frozen=True)
class TrainingPhase:
    name: str
    trainable_modules: tuple[str, ...]
    frozen_modules: tuple[str, ...]

    def validate(self) -> None:
        if not self.name:
            raise ValueError("training phase name cannot be empty")
        trainable = frozenset(self.trainable_modules)
        frozen = frozenset(self.frozen_modules)
        if len(trainable) != len(self.trainable_modules):
            raise ValueError(f"{self.name}: trainable_modules contains duplicates")
        if len(frozen) != len(self.frozen_modules):
            raise ValueError(f"{self.name}: frozen_modules contains duplicates")
        unknown = (trainable | frozen) - _MODULE_SET
        if unknown:
            raise ValueError(f"{self.name}: unknown modules: {sorted(unknown)}")
        overlap = trainable & frozen
        if overlap:
            raise ValueError(f"{self.name}: modules cannot be both trainable and frozen: {sorted(overlap)}")
        missing = _MODULE_SET - (trainable | frozen)
        if missing:
            raise ValueError(f"{self.name}: every module needs an explicit state: {sorted(missing)}")

    @classmethod
    def from_mapping(cls, name: str, data: Mapping[str, Any]) -> "TrainingPhase":
        phase = cls(
            name=name,
            trainable_modules=tuple(data.get("trainable_modules", ())),
            frozen_modules=tuple(data.get("frozen_modules", ())),
        )
        phase.validate()
        return phase


@dataclass(frozen=True)
class FreezeAudit:
    phase: str
    module_trainable: Mapping[str, bool]
    parameter_tensors: Mapping[str, int]
    parameter_elements: Mapping[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "module_trainable": dict(self.module_trainable),
            "parameter_tensors": dict(self.parameter_tensors),
            "parameter_elements": dict(self.parameter_elements),
        }


def _validate_modules(modules: Mapping[str, Any]) -> None:
    supplied = frozenset(modules)
    if supplied != _MODULE_SET:
        missing = sorted(_MODULE_SET - supplied)
        extra = sorted(supplied - _MODULE_SET)
        raise ValueError(f"module registry mismatch; missing={missing}, extra={extra}")
    owners: dict[int, str] = {}
    for module_name, module in modules.items():
        parameters = getattr(module, "parameters", None)
        if not callable(parameters):
            raise TypeError(f"{module_name} must provide parameters()")
        for parameter in parameters():
            parameter_id = id(parameter)
            if parameter_id in owners:
                raise ValueError(
                    f"parameter is shared by {owners[parameter_id]} and {module_name}; "
                    "freeze ownership would be ambiguous"
                )
            owners[parameter_id] = module_name


def _set_requires_grad(parameter: Any, trainable: bool) -> None:
    setter = getattr(parameter, "requires_grad_", None)
    if callable(setter):
        setter(trainable)
    elif hasattr(parameter, "requires_grad"):
        parameter.requires_grad = trainable
    else:
        raise TypeError("parameter must expose requires_grad_() or requires_grad")


def _parameter_elements(parameter: Any) -> int:
    numel = getattr(parameter, "numel", None)
    return int(numel()) if callable(numel) else 0


def apply_freeze_plan(
    modules: Mapping[str, Any], phase: TrainingPhase
) -> FreezeAudit:
    """Apply one explicit plan and return checkpoint-friendly audit data."""
    phase.validate()
    _validate_modules(modules)
    trainable_names = frozenset(phase.trainable_modules)
    module_trainable: dict[str, bool] = {}
    parameter_tensors: dict[str, int] = {}
    parameter_elements: dict[str, int] = {}
    for module_name in MODULE_NAMES:
        trainable = module_name in trainable_names
        parameters = tuple(modules[module_name].parameters())
        for parameter in parameters:
            _set_requires_grad(parameter, trainable)
        module_trainable[module_name] = trainable
        parameter_tensors[module_name] = len(parameters)
        parameter_elements[module_name] = sum(_parameter_elements(item) for item in parameters)
    return FreezeAudit(
        phase=phase.name,
        module_trainable=module_trainable,
        parameter_tensors=parameter_tensors,
        parameter_elements=parameter_elements,
    )


def optimizer_parameters(
    modules: Mapping[str, Any], phase: TrainingPhase
) -> list[Any]:
    """Return only parameters owned by modules trainable in ``phase``."""
    phase.validate()
    _validate_modules(modules)
    selected: list[Any] = []
    for module_name in MODULE_NAMES:
        if module_name not in phase.trainable_modules:
            continue
        selected.extend(
            parameter
            for parameter in modules[module_name].parameters()
            if bool(getattr(parameter, "requires_grad", True))
        )
    return selected


def load_training_phases(path: str | Path) -> dict[str, TrainingPhase]:
    """Load and validate all named phase plans from JSON."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_phases = document.get("phases")
    if not isinstance(raw_phases, dict) or not raw_phases:
        raise ValueError("training phase config must contain a non-empty phases object")
    return {
        name: TrainingPhase.from_mapping(name, value)
        for name, value in raw_phases.items()
    }
