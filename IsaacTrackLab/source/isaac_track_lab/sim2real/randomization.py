"""Seeded episode randomization and time-correlated camera vibration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import pi, sin
from pathlib import Path
import random
from typing import Any

from .config import Sim2RealProfile


def _uniform(rng: random.Random, values: list[float]) -> float:
    return rng.uniform(float(values[0]), float(values[1]))


def _vec_uniform(rng: random.Random, values: list[list[float]]) -> tuple[float, ...]:
    return tuple(_uniform(rng, pair) for pair in values)


@dataclass(frozen=True)
class EpisodeRandomization:
    seed: int
    profile: str
    mount_translation_bias_m: tuple[float, float, float]
    mount_rpy_bias_deg: tuple[float, float, float]
    hdri: str
    hdri_rotation_deg: float
    hdri_intensity: float
    sun_azimuth_deg: float
    sun_elevation_deg: float
    sun_intensity: float
    light_rgb: tuple[float, float, float]
    shadow_strength: float
    exposure_ev: float
    gamma: float
    white_balance_rgb: tuple[float, float, float]
    materials: dict[str, dict[str, Any]]
    target_asset: str
    target_scale: float
    target_clothing_rgb: tuple[float, float, float]
    target_roughness: float
    target_speed_mps: float
    target_pause_start_fraction: float
    target_pause_duration_s: float
    target_lateral_shift_m: float
    target_turn_deg: float
    distractor_count: int
    distractor_assets: tuple[str, ...]
    skipped_assets: tuple[dict[str, str], ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _existing_assets(root: Path, candidates: list[str]) -> tuple[list[str], list[dict[str, str]]]:
    existing, skipped = [], []
    for item in candidates:
        path = Path(item)
        if not path.is_absolute():
            path = root / path
        if path.is_file():
            existing.append(str(path))
        else:
            skipped.append({"asset": item, "reason": "asset_not_found"})
    return existing, skipped


def build_episode_randomization(
    profile: Sim2RealProfile,
    *,
    root: Path,
    distractor_range: tuple[int, int] = (0, 0),
    map_id: str | None = None,
) -> EpisodeRandomization:
    rng = random.Random(profile.seed)
    enabled = profile.enabled
    camera = profile.section("camera_vibration")
    lighting = profile.section("lighting")
    materials_cfg = profile.section("materials")
    people = profile.section("people")

    if enabled["camera_vibration"]:
        translation = _vec_uniform(rng, camera["episode_mount_translation_m"])
        rpy = _vec_uniform(rng, camera["episode_mount_rpy_deg"])
    else:
        translation, rpy = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)

    hdri_candidates = lighting.get("hdri_by_map", {}).get(map_id, lighting["hdri_assets"])
    hdri_assets, skipped_hdri = _existing_assets(root, hdri_candidates)
    if not hdri_assets:
        raise FileNotFoundError("no configured local HDRI asset exists")
    light_on = enabled["lighting_randomization"]
    hdri = rng.choice(hdri_assets) if light_on else hdri_assets[0]

    semantic_materials: dict[str, dict[str, Any]] = {}
    for semantic, cfg in materials_cfg["semantic_groups"].items():
        variants, skipped = _existing_assets(root, cfg["albedo_assets"])
        skipped_hdri.extend(skipped)
        if not variants:
            raise FileNotFoundError(f"no local material asset for semantic group {semantic}")
        semantic_materials[semantic] = {
            "albedo": rng.choice(variants) if enabled["material_randomization"] else variants[0],
            "tint_rgb": _vec_uniform(rng, cfg["tint_rgb"])
            if enabled["material_randomization"] else (1.0, 1.0, 1.0),
            "roughness": _uniform(rng, cfg["roughness"])
            if enabled["material_randomization"] else float(cfg["roughness"][0]),
            "texture_scale": _uniform(rng, cfg["texture_scale"])
            if enabled["material_randomization"] else 1.0,
            "texture_rotation_deg": _uniform(rng, cfg["texture_rotation_deg"])
            if enabled["material_randomization"] else 0.0,
            "wetness": _uniform(rng, cfg["wetness"])
            if enabled["material_randomization"] else 0.0,
        }

    people_assets, skipped_people = _existing_assets(root, people["assets"])
    if not people_assets:
        raise FileNotFoundError("no configured local person asset exists")
    people_on = enabled["person_appearance_randomization"]
    distractors_on = enabled["distractors_occlusion"]
    count = rng.randint(*map(int, distractor_range)) if distractors_on else 0
    distractor_assets = tuple(rng.choice(people_assets) for _ in range(count))
    return EpisodeRandomization(
        seed=profile.seed,
        profile=profile.name,
        mount_translation_bias_m=translation,
        mount_rpy_bias_deg=rpy,
        hdri=hdri,
        hdri_rotation_deg=_uniform(rng, lighting["hdri_rotation_deg"]) if light_on else 0.0,
        hdri_intensity=_uniform(rng, lighting["hdri_intensity"]) if light_on else 1.0,
        sun_azimuth_deg=_uniform(rng, lighting["sun_azimuth_deg"]) if light_on else 135.0,
        sun_elevation_deg=_uniform(rng, lighting["sun_elevation_deg"]) if light_on else 45.0,
        sun_intensity=_uniform(rng, lighting["sun_intensity"]) if light_on else 1.0,
        light_rgb=_vec_uniform(rng, lighting["light_rgb"]) if light_on else (1.0, 1.0, 1.0),
        shadow_strength=_uniform(rng, lighting["shadow_strength"]) if light_on else 1.0,
        exposure_ev=_uniform(rng, lighting["exposure_ev"]) if light_on else 0.0,
        gamma=_uniform(rng, lighting["gamma"]) if light_on else 1.0,
        white_balance_rgb=_vec_uniform(rng, lighting["white_balance_rgb"])
        if light_on else (1.0, 1.0, 1.0),
        materials=semantic_materials,
        target_asset=rng.choice(people_assets) if people_on else people_assets[0],
        target_scale=_uniform(rng, people["scale"]) if people_on else 1.0,
        target_clothing_rgb=_vec_uniform(rng, people["clothing_rgb"])
        if people_on else (1.0, 1.0, 1.0),
        target_roughness=_uniform(rng, people["roughness"]) if people_on else 0.6,
        target_speed_mps=_uniform(rng, people["speed_mps"]) if people_on else 1.25,
        target_pause_start_fraction=rng.uniform(0.25, 0.70) if people_on else 2.0,
        target_pause_duration_s=_uniform(rng, people["pause_duration_s"]) if people_on else 0.0,
        target_lateral_shift_m=_uniform(rng, people["lateral_shift_m"]) if people_on else 0.0,
        target_turn_deg=_uniform(rng, people["turn_deg"]) if people_on else 0.0,
        distractor_count=count,
        distractor_assets=distractor_assets,
        skipped_assets=tuple(skipped_hdri + skipped_people),
    )


class TemporalCameraVibration:
    """OU vibration plus gait harmonics; never independent frame noise."""

    def __init__(self, config: dict[str, Any], seed: int, enabled: bool = True) -> None:
        self.config = config
        self.enabled = enabled
        self.rng = random.Random(seed ^ 0xC0FFEE)
        self.state = [0.0] * 6
        self.time_s = 0.0

    def reset(self) -> None:
        self.state[:] = [0.0] * 6
        self.time_s = 0.0

    def step(
        self,
        dt: float,
        *,
        base_angular_speed_rps: float = 0.0,
        foot_contact_fraction: float = 1.0,
        gait_phase: float | None = None,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        if not self.enabled:
            return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
        self.time_s += dt
        theta = float(self.config["ou_theta"])
        sigma = self.config["ou_sigma_translation_m"] + self.config["ou_sigma_rpy_deg"]
        coupling = 1.0 + float(self.config["angular_velocity_gain"]) * base_angular_speed_rps
        coupling += float(self.config["foot_contact_gain"]) * max(0.0, 1.0 - foot_contact_fraction)
        for i in range(6):
            self.state[i] += theta * (0.0 - self.state[i]) * dt
            self.state[i] += float(sigma[i]) * coupling * (dt ** 0.5) * self.rng.gauss(0.0, 1.0)
        phase = gait_phase if gait_phase is not None else (
            self.time_s * float(self.config["gait_frequency_hz"])
        ) % 1.0
        harmonic = sin(2.0 * pi * phase)
        translation = tuple(
            self.state[i] + float(self.config["gait_translation_amplitude_m"][i]) * harmonic
            for i in range(3)
        )
        rotation = tuple(
            self.state[i + 3] + float(self.config["gait_rpy_amplitude_deg"][i]) * harmonic
            for i in range(3)
        )
        return translation, rotation


def smoothstep(value: float) -> float:
    value = min(1.0, max(0.0, value))
    return value * value * (3.0 - 2.0 * value)


def event_weight(now_s: float, start_s: float, duration_s: float, ramp_fraction: float = 0.2) -> float:
    if now_s < start_s or now_s >= start_s + duration_s:
        return 0.0
    local = (now_s - start_s) / duration_s
    if local < ramp_fraction:
        return smoothstep(local / ramp_fraction)
    if local > 1.0 - ramp_fraction:
        return smoothstep((1.0 - local) / ramp_fraction)
    return 1.0
