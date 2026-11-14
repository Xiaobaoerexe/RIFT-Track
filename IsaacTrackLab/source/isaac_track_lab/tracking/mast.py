"""Leak-resistant adapter from RGB plus one initial box to MaST ONNX."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class MaSTPrediction:
    bbox_xyxy: tuple[float, float, float, float] | None
    confidence: float | None
    visible: bool
    latency_ms: float
    mode: str

class MaSTTemplateTracker:
    """MaST adapter that cannot consume post-initialization ground truth."""

    def __init__(
        self,
        mastrack_root: Path | str,
        *,
        model_name: str = "MaST_nano",
        minimum_score: float = 0.15,
        recovery_score: float = 0.20,
        recovery_search_factor: float = 8.0,
        device_id: int = 0,
        require_cuda: bool = True,
    ) -> None:
        root = Path(mastrack_root).expanduser().resolve()
        if not (root / "mastrack/mast_backend.py").is_file():
            raise FileNotFoundError(f"MaSTrack package not found: {root}")
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        module = importlib.import_module("mastrack.mast_backend")
        self._backend = module.MaSTOnnxBackend(
            model_name=model_name,
            device_id=device_id,
            require_cuda=require_cuda,
        )
        self.minimum_score = float(minimum_score)
        self.recovery_score = float(recovery_score)
        self.recovery_search_factor = float(recovery_search_factor)
        self._normal_search_factor = float(self._backend.tracker.search_factor)
        self._initialized = False
        self._lost_frames = 0
        self._frame_shape: tuple[int, int] | None = None
        self._initial_area: float | None = None
        self._initial_aspect: float | None = None
        self._last_accepted_box: tuple[float, float, float, float] | None = None

    @property
    def providers(self) -> tuple[str, ...]:
        return self._backend.providers

    def initialize(self, rgb: np.ndarray, bbox_xyxy: Sequence[float]) -> None:
        image = self._rgb(rgb)
        x1, y1, x2, y2 = self._xyxy(bbox_xyxy, image.shape)
        self._backend.initialize(image, (x1, y1, x2 - x1, y2 - y1))
        self._frame_shape = image.shape[:2]
        self._initial_area = (x2 - x1) * (y2 - y1)
        self._initial_aspect = (x2 - x1) / (y2 - y1)
        self._last_accepted_box = (x1, y1, x2, y2)
        self._lost_frames = 0
        self._initialized = True

    def track(self, rgb: np.ndarray) -> MaSTPrediction:
        if not self._initialized or self._frame_shape is None:
            raise RuntimeError("MaST tracker has not been initialized")
        image = self._rgb(rgb)
        started = time.perf_counter()
        self._backend.tracker.search_factor = (
            self.recovery_search_factor if self._lost_frames else self._normal_search_factor
        )
        try:
            result = self._backend.track(image)
        finally:
            self._backend.tracker.search_factor = self._normal_search_factor
        latency_ms = (time.perf_counter() - started) * 1000.0
        score = result.score
        score_reliable = score is not None and score >= (
            self.recovery_score if self._lost_frames else self.minimum_score
        )
        if not score_reliable:
            self._backend.reject()
            self._lost_frames += 1
            return MaSTPrediction(None, score, False, latency_ms, "search")

        height, width = self._frame_shape
        x, y, box_width, box_height = result.bbox
        x1 = max(0.0, min(float(width), x))
        y1 = max(0.0, min(float(height), y))
        x2 = max(0.0, min(float(width), x + box_width))
        y2 = max(0.0, min(float(height), y + box_height))
        if x2 <= x1 or y2 <= y1:
            self._backend.reject()
            self._lost_frames += 1
            return MaSTPrediction(None, score, False, latency_ms, "search")
        candidate = (x1, y1, x2, y2)
        if not self._geometry_reliable(candidate):
            self._backend.reject()
            self._lost_frames += 1
            return MaSTPrediction(None, score, False, latency_ms, "search")
        self._backend.accept((x1, y1, x2 - x1, y2 - y1))
        self._last_accepted_box = candidate
        mode = "reacquired" if self._lost_frames else "tracking"
        self._lost_frames = 0
        return MaSTPrediction(candidate, score, True, latency_ms, mode)

    def _geometry_reliable(self, box: tuple[float, float, float, float]) -> bool:
        if (
            self._initial_area is None
            or self._initial_aspect is None
            or self._frame_shape is None
        ):
            return False
        x1, y1, x2, y2 = box
        width, height = x2 - x1, y2 - y1
        area = width * height
        aspect = width / height
        initial_area_ratio = area / self._initial_area
        aspect_ratio = aspect / self._initial_aspect
        image_area = float(self._frame_shape[0] * self._frame_shape[1])
        # During a real loss the robot stops while the target keeps walking.
        # A target that recedes from 3 m to roughly 10 m occupies about 9% of
        # its initial area.  The previous 20% floor therefore rejected a clear,
        # centered target after physical occlusion.  Template score, aspect,
        # absolute image area and center-jump checks remain active.
        if not (
            0.02 <= initial_area_ratio <= 2.75
            and 0.40 <= aspect_ratio <= 2.50
            and 0.0003 <= area / image_area <= 0.12
        ):
            return False
        if self._last_accepted_box is None:
            return True
        previous = self._last_accepted_box
        previous_area = (previous[2] - previous[0]) * (previous[3] - previous[1])
        # ``previous`` can predate a multi-second occlusion, so allow a large
        # scale reduction on reacquisition. Normal frame-to-frame stability is
        # still enforced by the tracker score and center-jump bound.
        if not 0.06 <= area / max(previous_area, 1.0) <= 2.50:
            return False
        previous_center = ((previous[0] + previous[2]) * 0.5, (previous[1] + previous[3]) * 0.5)
        center = ((x1 + x2) * 0.5, (y1 + y2) * 0.5)
        center_jump = (
            ((center[0] - previous_center[0]) / self._frame_shape[1]) ** 2
            + ((center[1] - previous_center[1]) / self._frame_shape[0]) ** 2
        ) ** 0.5
        # A recovery scan deliberately rotates the camera until the target can
        # re-enter on the opposite side of the frame.  Retaining the normal
        # 25% frame-diagonal gate after a confirmed loss makes that valid
        # re-entry impossible to accept.  Keep the strict gate during normal
        # tracking and widen it only while the tracker is already lost.
        maximum_center_jump = 0.75 if self._lost_frames else 0.25
        return center_jump <= maximum_center_jump

    @staticmethod
    def _rgb(rgb: np.ndarray) -> np.ndarray:
        image = np.asarray(rgb)
        if image.ndim != 3 or image.shape[2] < 3 or image.dtype != np.uint8:
            raise ValueError("MaST input must be HxWx3 uint8 RGB")
        return np.ascontiguousarray(image[:, :, :3])

    @staticmethod
    def _xyxy(
        bbox: Sequence[float], image_shape: Sequence[int]
    ) -> tuple[float, float, float, float]:
        if len(bbox) != 4:
            raise ValueError("initial bbox must be [x1, y1, x2, y2]")
        height, width = int(image_shape[0]), int(image_shape[1])
        x1, y1, x2, y2 = map(float, bbox)
        if not np.isfinite((x1, y1, x2, y2)).all():
            raise ValueError("initial bbox must be finite")
        x1, x2 = max(0.0, x1), min(float(width), x2)
        y1, y2 = max(0.0, y1), min(float(height), y2)
        if x2 <= x1 or y2 <= y1:
            raise ValueError("initial bbox is empty or outside the image")
        return x1, y1, x2, y2
