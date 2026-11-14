"""Template-only appearance verification for rejecting tracker drift."""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np


class TemplateHistogramVerifier:
    """Compare a predicted crop to the initial RGB template in HSV space."""

    def __init__(self, template_rgb: np.ndarray) -> None:
        self._template = self._histogram(template_rgb)
        self._template_hues = self._dominant_hues(template_rgb)

    def similarity(
        self, rgb: np.ndarray, bbox_xyxy: Sequence[float] | None
    ) -> float:
        if bbox_xyxy is None or len(bbox_xyxy) != 4:
            return 0.0
        height, width = rgb.shape[:2]
        x1, y1, x2, y2 = bbox_xyxy
        left, top = max(0, int(x1)), max(0, int(y1))
        right, bottom = min(width, int(np.ceil(x2))), min(height, int(np.ceil(y2)))
        if right - left < 4 or bottom - top < 4:
            return 0.0
        candidate = self._histogram(rgb[top:bottom, left:right, :3])
        distance = cv2.compareHist(self._template, candidate, cv2.HISTCMP_BHATTACHARYYA)
        return float(np.clip(1.0 - distance, 0.0, 1.0))

    def locate(self, rgb: np.ndarray) -> tuple[float, float, float, float] | None:
        """Find a full-frame recovery candidate using template-derived hues.

        This is intentionally used only after loss.  It contains no fixed
        target colour and consumes the same RGB/template pair available at
        deployment.
        """
        image = np.asarray(rgb)
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError("rgb must be an HxWx3 image")
        hsv = cv2.cvtColor(np.ascontiguousarray(image[:, :, :3]), cv2.COLOR_RGB2HSV)
        hue = hsv[:, :, 0].astype(np.int16)
        mask = np.zeros(hue.shape, dtype=np.uint8)
        for template_hue in self._template_hues:
            distance = np.abs(hue - template_hue)
            circular_distance = np.minimum(distance, 180 - distance)
            mask[circular_distance <= 9] = 255
        mask[(hsv[:, :, 1] < 45) | (hsv[:, :, 2] < 30)] = 0
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        best_box = None
        best_score = -1.0
        height, width = image.shape[:2]
        for component in range(1, count):
            x, y, box_width, box_height, pixels = map(int, stats[component])
            if pixels < 10 or box_height < 8:
                continue
            left = max(0, int(x - 0.45 * box_width))
            right = min(width, int(x + 1.45 * box_width))
            top = max(0, int(y - 0.20 * box_height))
            bottom = min(height, int(y + 1.45 * box_height))
            if bottom - top < 12 or right - left < 4:
                continue
            aspect = (right - left) / max(bottom - top, 1)
            if not 0.15 <= aspect <= 1.15:
                continue
            box = (float(left), float(top), float(right), float(bottom))
            similarity = self.similarity(image, box)
            score = similarity + min(pixels / 500.0, 0.25)
            if similarity >= 0.08 and score > best_score:
                best_box, best_score = box, score
        return best_box

    @staticmethod
    def _dominant_hues(rgb: np.ndarray) -> tuple[int, ...]:
        image = np.asarray(rgb)
        if image.ndim != 3 or image.shape[2] < 3 or image.size == 0:
            raise ValueError("template must be a non-empty HxWx3 image")
        hsv = cv2.cvtColor(np.ascontiguousarray(image[:, :, :3]), cv2.COLOR_RGB2HSV)
        valid = (hsv[:, :, 1] >= 45) & (hsv[:, :, 2] >= 30)
        values = hsv[:, :, 0][valid]
        if values.size == 0:
            return (0,)
        histogram = np.bincount(values, minlength=180).astype(np.float64)
        selected: list[int] = []
        for _ in range(3):
            index = int(np.argmax(histogram))
            if histogram[index] <= 0.0:
                break
            selected.append(index)
            for offset in range(-10, 11):
                histogram[(index + offset) % 180] = 0.0
        return tuple(selected)

    @staticmethod
    def _histogram(rgb: np.ndarray) -> np.ndarray:
        image = np.asarray(rgb)
        if image.ndim != 3 or image.shape[2] < 3 or image.size == 0:
            raise ValueError("template/candidate must be a non-empty HxWx3 image")
        hsv = cv2.cvtColor(np.ascontiguousarray(image[:, :, :3]), cv2.COLOR_RGB2HSV)
        # Ignore low-saturation road, wall and black vehicle pixels.  The mask
        # is derived identically from template and candidate; it does not encode
        # a target-specific colour.  A soft central ellipse also reduces the
        # background included by a loose full-body box.
        foreground = np.zeros(hsv.shape[:2], dtype=np.uint8)
        cv2.ellipse(
            foreground,
            (hsv.shape[1] // 2, hsv.shape[0] // 2),
            (max(1, int(hsv.shape[1] * 0.46)), max(1, int(hsv.shape[0] * 0.48))),
            0.0, 0.0, 360.0, 255, -1,
        )
        saturated = ((hsv[:, :, 1] >= 55) & (hsv[:, :, 2] >= 35)).astype(np.uint8) * 255
        mask = cv2.bitwise_and(foreground, saturated)
        histogram = cv2.calcHist([hsv], [0, 1], mask, [24, 16], [0, 180, 0, 256])
        if float(histogram.sum()) <= 0.0:
            return histogram
        cv2.normalize(histogram, histogram, alpha=1.0, norm_type=cv2.NORM_L1)
        return histogram
