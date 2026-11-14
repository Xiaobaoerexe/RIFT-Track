"""D455 observation degradation applied to tensors consumed by the actor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .config import Sim2RealProfile


@dataclass(frozen=True)
class ProcessedRGBD:
    rgb: torch.Tensor
    depth_m: torch.Tensor
    depth_valid_mask: torch.Tensor
    metadata: dict[str, Any]


def _sample_uniform(generator: torch.Generator, low: float, high: float, device: torch.device) -> float:
    value = torch.rand((), generator=generator, device=device)
    return float(low + (high - low) * value)


class D455PostProcessor:
    """Stateful, seeded RGB-D degradation with GPU-resident image operations."""

    def __init__(self, profile: Sim2RealProfile, device: str | torch.device = "cpu") -> None:
        self.profile = profile
        self.device = torch.device(device)
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(profile.seed)
        self.depth_cfg = profile.section("depth_degradation")
        self.rgb_cfg = profile.section("rgb_degradation")
        self.temporal_hole_mask: torch.Tensor | None = None
        self.previous_rgb: torch.Tensor | None = None
        self.previous_depth: torch.Tensor | None = None
        self.frame_index = 0

    def reset(self) -> None:
        self.generator.manual_seed(self.profile.seed)
        self.temporal_hole_mask = None
        self.previous_rgb = None
        self.previous_depth = None
        self.frame_index = 0

    @staticmethod
    def _as_rgb(value: torch.Tensor, device: torch.device) -> torch.Tensor:
        tensor = torch.as_tensor(value, device=device)
        if tensor.ndim == 4:
            tensor = tensor[0]
        if tensor.shape[-1] > 3:
            tensor = tensor[..., :3]
        if tensor.ndim != 3 or tensor.shape[-1] != 3:
            raise ValueError("RGB must have HxWx3 shape")
        return tensor.to(torch.float32)

    @staticmethod
    def _as_depth(value: torch.Tensor, device: torch.device) -> torch.Tensor:
        tensor = torch.as_tensor(value, device=device)
        if tensor.ndim == 4:
            tensor = tensor[0]
        if tensor.ndim == 3 and tensor.shape[-1] == 1:
            tensor = tensor[..., 0]
        if tensor.ndim != 2:
            raise ValueError("Depth must have HxW shape")
        return tensor.to(torch.float32)

    def process(
        self,
        rgb: torch.Tensor,
        depth_m: torch.Tensor,
        *,
        motion_intensity: float = 0.0,
        underexposure_weight: float = 0.0,
        force_depth_dropout: bool = False,
        shot_noise_weight: float = 0.0,
    ) -> ProcessedRGBD:
        rgb_t = self._as_rgb(rgb, self.device)
        depth_t = self._as_depth(depth_m, self.device)
        rgb_out, rgb_meta = self._process_rgb(
            rgb_t, motion_intensity, underexposure_weight, shot_noise_weight
        )
        depth_out, mask, depth_meta = self._process_depth(depth_t, force_depth_dropout)
        self.previous_rgb = rgb_out.clone()
        self.previous_depth = depth_out.clone()
        self.frame_index += 1
        return ProcessedRGBD(
            rgb=rgb_out,
            depth_m=depth_out,
            depth_valid_mask=mask,
            metadata={**rgb_meta, **depth_meta, "frame_index": self.frame_index - 1},
        )

    def _process_depth(
        self, depth: torch.Tensor, force_dropout: bool
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        cfg = self.depth_cfg
        enabled = self.profile.enabled["depth_degradation"]
        minimum, maximum = map(float, cfg["valid_range_m"])
        valid = torch.isfinite(depth)
        output = depth.clone()
        if enabled:
            valid &= (depth >= minimum) & (depth <= maximum)
            sigma = float(cfg["noise_base_sigma_m"]) + float(cfg["noise_quadratic_sigma_m"]) * output.square()
            output = output + torch.randn(
                output.shape, generator=self.generator, device=self.device
            ) * sigma
            quantum = float(cfg["quantization_m"])
            output = torch.round(output / quantum) * quantum
            dx = F.pad(torch.abs(output[:, 1:] - output[:, :-1]), (0, 1))
            dy = F.pad(torch.abs(output[1:, :] - output[:-1, :]), (0, 0, 0, 1))
            edges = torch.maximum(dx, dy) > float(cfg["edge_threshold_m"])
            edge_holes = edges & (
                torch.rand(output.shape, generator=self.generator, device=self.device)
                < float(cfg["edge_hole_probability"])
            )
            flying = edges & ~edge_holes & (
                torch.rand(output.shape, generator=self.generator, device=self.device)
                < float(cfg["flying_pixel_probability"])
            )
            if torch.any(flying):
                shifted = torch.roll(output, shifts=1, dims=1)
                mix = torch.rand(output.shape, generator=self.generator, device=self.device)
                output = torch.where(flying, output * mix + shifted * (1.0 - mix), output)
            valid &= ~edge_holes

            local = torch.zeros_like(valid)
            count_range = cfg["local_hole_count"]
            count = int(_sample_uniform(
                self.generator, int(count_range[0]), int(count_range[1]) + 0.999, self.device
            ))
            height, width = output.shape
            radius_range = cfg["local_hole_radius_px"]
            yy, xx = torch.meshgrid(
                torch.arange(height, device=self.device),
                torch.arange(width, device=self.device), indexing="ij",
            )
            for _ in range(count):
                cx = int(_sample_uniform(self.generator, 0, width, self.device))
                cy = int(_sample_uniform(self.generator, 0, height, self.device))
                radius = _sample_uniform(
                    self.generator, float(radius_range[0]), float(radius_range[1]), self.device
                )
                local |= (xx - cx).square() + (yy - cy).square() <= radius * radius
            persistence = float(cfg["temporal_hole_persistence"])
            if self.temporal_hole_mask is None:
                self.temporal_hole_mask = local
            else:
                keep = torch.rand(output.shape, generator=self.generator, device=self.device) < persistence
                drift_x = int(_sample_uniform(self.generator, -1.5, 1.5, self.device))
                drift_y = int(_sample_uniform(self.generator, -1.5, 1.5, self.device))
                persisted = torch.roll(self.temporal_hole_mask, (drift_y, drift_x), (0, 1)) & keep
                self.temporal_hole_mask = persisted | local
            valid &= ~self.temporal_hole_mask

            shift = cfg["rgb_depth_subpixel_shift_px"]
            shift_x = _sample_uniform(self.generator, float(shift[0][0]), float(shift[0][1]), self.device)
            shift_y = _sample_uniform(self.generator, float(shift[1][0]), float(shift[1][1]), self.device)
            output, valid = self._subpixel_shift(output, valid, shift_x, shift_y)
            # Registration interpolation is followed by the device's metric
            # depth quantization; actor-visible valid samples remain quantized.
            output = torch.round(output / quantum) * quantum
            dropout = force_dropout or (
                _sample_uniform(self.generator, 0.0, 1.0, self.device)
                < float(cfg["full_frame_dropout_probability"])
            )
            if dropout:
                valid[:] = False
        else:
            shift_x = shift_y = 0.0
            dropout = force_dropout
            if dropout:
                valid[:] = False
        output = torch.where(valid, output, torch.full_like(output, float("nan")))
        return output, valid, {
            "depth_dropout": bool(dropout),
            "depth_invalid_fraction": float((~valid).to(torch.float32).mean()),
            "rgb_depth_shift_px": [shift_x, shift_y],
            "depth_quantization_m": float(cfg["quantization_m"]) if enabled else 0.0,
        }

    def _subpixel_shift(
        self, depth: torch.Tensor, valid: torch.Tensor, shift_x: float, shift_y: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        height, width = depth.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=self.device),
            torch.linspace(-1.0, 1.0, width, device=self.device), indexing="ij",
        )
        grid = torch.stack((
            xx - 2.0 * shift_x / max(1, width - 1),
            yy - 2.0 * shift_y / max(1, height - 1),
        ), dim=-1)[None]
        shifted_depth = F.grid_sample(
            depth.nan_to_num(0.0)[None, None], grid, mode="bilinear",
            padding_mode="zeros", align_corners=True,
        )[0, 0]
        shifted_valid = F.grid_sample(
            valid.to(torch.float32)[None, None], grid, mode="nearest",
            padding_mode="zeros", align_corners=True,
        )[0, 0] > 0.5
        return shifted_depth, shifted_valid

    def _process_rgb(
        self,
        rgb: torch.Tensor,
        motion_intensity: float,
        underexposure_weight: float,
        shot_noise_weight: float,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        cfg = self.rgb_cfg
        enabled = self.profile.enabled["rgb_degradation"]
        value = rgb.clamp(0, 255) / 255.0
        if not enabled:
            return rgb.to(torch.uint8), {
                "exposure_ev": 0.0, "gamma": 1.0,
                "white_balance_rgb": [1.0, 1.0, 1.0], "motion_blur_px": 0,
            }
        median_kernel = int(cfg.get("renderer_firefly_median_kernel", 0))
        if median_kernel >= 3:
            if median_kernel % 2 == 0:
                raise ValueError("renderer_firefly_median_kernel must be odd")
            height, width = value.shape[:2]
            chw = value.permute(2, 0, 1)[None]
            padded = F.pad(
                chw, (median_kernel // 2,) * 4, mode="reflect"
            )
            patches = F.unfold(padded, kernel_size=median_kernel).view(
                1, 3, median_kernel * median_kernel, height * width
            )
            value = patches.median(dim=2).values.view(3, height, width).permute(1, 2, 0)
        exposure = _sample_uniform(self.generator, *map(float, cfg["exposure_ev"]), self.device)
        exposure += underexposure_weight * float(cfg["temporary_underexposure_ev"])
        gamma = _sample_uniform(self.generator, *map(float, cfg["gamma"]), self.device)
        wb = torch.tensor([
            _sample_uniform(self.generator, *map(float, pair), self.device)
            for pair in cfg["white_balance_rgb"]
        ], device=self.device)
        value = value * (2.0 ** exposure) * wb
        gray = value.mean(dim=-1, keepdim=True)
        saturation = _sample_uniform(self.generator, *map(float, cfg["saturation"]), self.device)
        value = gray + saturation * (value - gray)
        value = value.clamp(0.0, 1.0).pow(1.0 / gamma)
        shot_scale = float(cfg["shot_noise_scale"]) * (1.0 + 2.0 * shot_noise_weight)
        shot_sigma = shot_scale * torch.sqrt(value.clamp_min(1e-4))
        read_sigma = float(cfg["read_noise_sigma"])
        noise = torch.randn(value.shape, generator=self.generator, device=self.device)
        value = value + noise * torch.sqrt(shot_sigma.square() + read_sigma ** 2)
        max_blur = int(cfg["motion_blur_max_px"])
        blur_px = min(max_blur, int(round(max_blur * max(0.0, motion_intensity))))
        if blur_px >= 2:
            kernel = blur_px + (1 - blur_px % 2)
            chw = value.permute(2, 0, 1)[None]
            value = F.avg_pool2d(chw, kernel_size=(1, kernel), stride=1, padding=(0, kernel // 2))[0].permute(1, 2, 0)
        return (value.clamp(0, 1) * 255.0).round().to(torch.uint8), {
            "exposure_ev": exposure, "gamma": gamma,
            "white_balance_rgb": wb.tolist(), "saturation": saturation,
            "motion_blur_px": blur_px,
            "renderer_firefly_median_kernel": median_kernel,
            "shot_noise_event_weight": shot_noise_weight,
        }
