"""Validated training profiles and the immutable measured D455 baseline."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class D455Nominal:
    width: int = 640
    height: int = 480
    fx: float = 391.18
    fy: float = 391.18
    cx: float = 321.78
    cy: float = 239.94
    body_relative_translation_m: tuple[float, float, float] = (0.32, 0.0, 0.32)
    body_relative_quaternion_xyzw_ros: tuple[float, float, float, float] = (
        0.54167522, -0.54167522, 0.45451948, -0.45451948,
    )
    optical_center_height_m: float = 0.66
    pitch_down_deg: float = 10.0
    rgb_hz: float = 30.0
    depth_hz: float = 30.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "image_size": [self.width, self.height],
            "fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy,
            "body_relative_translation_m": list(self.body_relative_translation_m),
            "body_relative_quaternion_xyzw_ros": list(
                self.body_relative_quaternion_xyzw_ros
            ),
            "optical_center_height_m": self.optical_center_height_m,
            "pitch_down_deg": self.pitch_down_deg,
            "rgb_hz": self.rgb_hz,
            "depth_hz": self.depth_hz,
            "status": "measured_and_frozen",
        }

    @classmethod
    def from_calibration(cls, path: Path | str) -> "D455Nominal":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        intr = data["color_aligned_intrinsics"]
        mount = data["mount"]
        size = data["image_size"]
        result = cls(
            width=int(size[0]), height=int(size[1]),
            fx=float(intr["fx"]), fy=float(intr["fy"]),
            cx=float(intr["cx"]), cy=float(intr["cy"]),
            body_relative_translation_m=tuple(map(float, mount["body_relative_translation_m"])),
            body_relative_quaternion_xyzw_ros=tuple(
                map(float, mount["body_relative_quaternion_xyzw_ros"])
            ),
            optical_center_height_m=float(mount["optical_center_height_m"]),
            pitch_down_deg=float(mount["pitch_down_deg"]),
            rgb_hz=float(data["nominal_frequency_hz"]["rgb"]),
            depth_hz=float(data["nominal_frequency_hz"]["depth"]),
        )
        if result != cls():
            raise ValueError(
                "D455 calibration differs from the measured frozen project baseline; "
                "nominal intrinsics/extrinsics must not be randomized or replaced"
            )
        return result


@dataclass(frozen=True)
class Sim2RealProfile:
    name: str
    seed: int
    raw: dict[str, Any]
    d455: D455Nominal

    @property
    def enabled(self) -> dict[str, bool]:
        return {key: bool(value) for key, value in self.raw["enabled"].items()}

    def section(self, name: str) -> dict[str, Any]:
        value = self.raw.get(name, {})
        if not isinstance(value, dict):
            raise TypeError(f"profile section {name!r} must be an object")
        return value

    def with_seed(self, seed: int) -> "Sim2RealProfile":
        return Sim2RealProfile(self.name, int(seed), self.raw, self.d455)

    def with_disabled(self, names: set[str]) -> "Sim2RealProfile":
        unknown = names - REQUIRED_SWITCHES
        if unknown:
            raise ValueError(f"unknown ablation switches: {sorted(unknown)}")
        raw = _deep_merge(self.raw, {"enabled": {name: False for name in names}})
        return Sim2RealProfile(self.name, self.seed, raw, self.d455)


REQUIRED_SWITCHES = {
    "rgb_degradation", "depth_degradation", "camera_vibration",
    "lighting_randomization", "material_randomization",
    "person_appearance_randomization", "distractors_occlusion",
    "sensor_latency", "action_latency",
}


def load_profile(
    name_or_path: str | Path,
    *,
    seed: int = 0,
    root: Path = ROOT,
) -> Sim2RealProfile:
    path = Path(name_or_path)
    if not path.suffix:
        path = root / "configs" / f"{path.name}.json"
    elif not path.is_absolute():
        path = root / path
    data = json.loads(path.read_text(encoding="utf-8"))
    if "inherits" in data:
        parent_path = Path(data["inherits"])
        if not parent_path.is_absolute():
            parent_path = root / parent_path
        parent = json.loads(parent_path.read_text(encoding="utf-8"))
        data = _deep_merge(parent, data)
    enabled = data.get("enabled")
    if not isinstance(enabled, dict) or set(enabled) != REQUIRED_SWITCHES:
        raise ValueError(
            f"profile enabled switches must be exactly {sorted(REQUIRED_SWITCHES)}"
        )
    calibration = root / data.get("d455_calibration", "configs/d455_sim_calibration.json")
    d455 = D455Nominal.from_calibration(calibration)
    return Sim2RealProfile(str(data["profile"]), int(seed), data, d455)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
