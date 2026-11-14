"""Python deployment adapter for MaSTrack's existing Go2 locomotion policy."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import onnxruntime as ort
import yaml


# The deployed policy's ``joint_ids_map`` converts Unitree SDK order into this
# policy order: front-left, front-right, rear-left, rear-right.
POLICY_JOINT_NAMES = (
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
)


@dataclass(frozen=True)
class Go2PolicyOutput:
    joint_position_targets: np.ndarray
    raw_actions: np.ndarray
    latency_ms: float


class Go2LocomotionPolicy:
    """Reproduce the observation/history/action contract used on the real Go2."""

    def __init__(
        self,
        model_path: str | Path,
        deploy_config_path: str | Path,
        providers: tuple[str, ...] = ("CPUExecutionProvider",),
    ) -> None:
        self.model_path = Path(model_path)
        self.deploy_config_path = Path(deploy_config_path)
        with self.deploy_config_path.open("r", encoding="utf-8-sig") as stream:
            self.config = yaml.safe_load(stream)
        self.session = ort.InferenceSession(str(self.model_path), providers=list(providers))
        self.providers = tuple(self.session.get_providers())
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or inputs[0].shape != [1, 225]:
            raise ValueError(f"unexpected Go2 policy input: {[(x.name, x.shape) for x in inputs]}")
        self.input_name = inputs[0].name
        action_outputs = [item.name for item in outputs if item.name == "actions"]
        if action_outputs != ["actions"]:
            raise ValueError(f"Go2 policy has no unique actions output: {[x.name for x in outputs]}")

        self.observation_config = self.config["observations"]
        self.term_order = tuple(self.observation_config.keys())
        expected_terms = (
            "base_ang_vel", "projected_gravity", "velocity_commands",
            "joint_pos_rel", "joint_vel_rel", "last_action",
        )
        if self.term_order != expected_terms:
            raise ValueError(f"unexpected observation order: {self.term_order}")
        self.default_joint_positions = np.asarray(
            self.config["default_joint_pos"], dtype=np.float32
        )
        self.action_scale = np.asarray(
            self.config["actions"]["JointPositionAction"]["scale"], dtype=np.float32
        )
        self.action_offset = np.asarray(
            self.config["actions"]["JointPositionAction"]["offset"], dtype=np.float32
        )
        self.step_dt = float(self.config["step_dt"])
        if self.default_joint_positions.shape != (12,) or self.action_scale.shape != (12,):
            raise ValueError("Go2 policy joint configuration must contain 12 values")
        self._history: dict[str, deque[np.ndarray]] = {}
        self._last_action = np.zeros(12, dtype=np.float32)

    def reset(self) -> None:
        self._history.clear()
        self._last_action.fill(0.0)

    @staticmethod
    def _vector(value: np.ndarray | tuple[float, ...], size: int, name: str) -> np.ndarray:
        vector = np.asarray(value, dtype=np.float32).reshape(-1)
        if vector.shape != (size,):
            raise ValueError(f"{name} must contain {size} values, got {vector.shape}")
        if not np.isfinite(vector).all():
            raise ValueError(f"{name} contains a non-finite value")
        return vector

    def _scaled(self, name: str, value: np.ndarray) -> np.ndarray:
        config = self.observation_config[name]
        clip = config.get("clip")
        if clip is not None:
            value = np.clip(value, float(clip[0]), float(clip[1]))
        scale = config.get("scale")
        if scale is not None:
            value = value * np.asarray(scale, dtype=np.float32)
        return value.astype(np.float32, copy=False)

    def _push(self, name: str, value: np.ndarray) -> np.ndarray:
        length = int(self.observation_config[name].get("history_length", 1))
        history = self._history.get(name)
        if history is None:
            history = deque((value.copy() for _ in range(length)), maxlen=length)
            self._history[name] = history
        else:
            history.append(value.copy())
        return np.concatenate(tuple(history), dtype=np.float32)

    def build_observation(
        self,
        base_angular_velocity: np.ndarray | tuple[float, ...],
        projected_gravity: np.ndarray | tuple[float, ...],
        velocity_command: np.ndarray | tuple[float, ...],
        joint_positions: np.ndarray,
        joint_velocities: np.ndarray,
    ) -> np.ndarray:
        values = {
            "base_ang_vel": self._vector(base_angular_velocity, 3, "base_angular_velocity"),
            "projected_gravity": self._vector(projected_gravity, 3, "projected_gravity"),
            "velocity_commands": self._vector(velocity_command, 3, "velocity_command"),
            "joint_pos_rel": self._vector(joint_positions, 12, "joint_positions")
            - self.default_joint_positions,
            "joint_vel_rel": self._vector(joint_velocities, 12, "joint_velocities"),
            "last_action": self._last_action.copy(),
        }
        observation = np.concatenate(
            [self._push(name, self._scaled(name, values[name])) for name in self.term_order],
            dtype=np.float32,
        )
        if observation.shape != (225,):
            raise RuntimeError(f"Go2 observation is {observation.shape}, expected (225,)")
        return observation

    def step(
        self,
        base_angular_velocity: np.ndarray | tuple[float, ...],
        projected_gravity: np.ndarray | tuple[float, ...],
        velocity_command: np.ndarray | tuple[float, ...],
        joint_positions: np.ndarray,
        joint_velocities: np.ndarray,
    ) -> Go2PolicyOutput:
        observation = self.build_observation(
            base_angular_velocity,
            projected_gravity,
            velocity_command,
            joint_positions,
            joint_velocities,
        )
        started = perf_counter()
        actions = self.session.run(["actions"], {self.input_name: observation[None]})[0]
        latency_ms = (perf_counter() - started) * 1000.0
        actions = np.asarray(actions, dtype=np.float32).reshape(-1)
        if actions.shape != (12,) or not np.isfinite(actions).all():
            raise RuntimeError("Go2 policy returned an invalid 12-joint action")
        self._last_action = actions.copy()
        targets = actions * self.action_scale + self.action_offset
        return Go2PolicyOutput(targets, actions, latency_ms)

