"""Autocast only neural operators; physical state and reductions stayFP32."""
import torch


def network_autocast(reference,enabled=True):
    # CPU and explicit deployment/export paths remainFP32. Parameters and Adam
    # moments stayFP32; BF16 is an execution dtype, not a checkpoint conversion.
    return torch.autocast(device_type=reference.device.type,dtype=torch.bfloat16,
                          enabled=bool(enabled and reference.is_cuda))
