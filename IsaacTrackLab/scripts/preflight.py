#!/usr/bin/env python3
from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def command(*args: str) -> str | None:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL, timeout=10).strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        return None


def status(ok: bool, label: str, detail: str) -> bool:
    print(f"[{'OK' if ok else 'MISSING'}] {label}: {detail}")
    return ok


def main() -> None:
    checks: list[bool] = []
    checks.append(status(platform.machine() == "x86_64", "architecture", platform.machine()))
    project_python = ROOT / ".venv/bin/python"
    project_version = command(str(project_python), "--version") if project_python.exists() else None
    checks.append(status(project_version is not None and "3.12." in project_version, "project Python", project_version or "create .venv with Python 3.12"))

    gpu = command("nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader")
    checks.append(status(gpu is not None, "NVIDIA GPU", gpu or "nvidia-smi unavailable"))

    memory_kib = 0
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemTotal:"):
            memory_kib = int(line.split()[1])
    checks.append(status(memory_kib >= 32 * 1024 * 1024, "RAM", f"{memory_kib / 1024 / 1024:.1f} GiB"))

    free_gib = shutil.disk_usage(ROOT).free / 1024**3
    checks.append(status(free_gib >= 100, "free storage", f"{free_gib:.1f} GiB"))
    checks.append(status(shutil.which("uv") is not None, "uv", command("uv", "--version") or "not installed"))

    urdf = ROOT / "third_party/unitree_ros/robots/as2_description/urdf/as2.urdf"
    checks.append(status(urdf.is_file(), "As2 URDF", str(urdf)))
    lab = ROOT / "third_party/IsaacLab/source/isaaclab"
    checks.append(status(lab.is_dir(), "Isaac Lab v3 source", str(lab)))

    sim_path = os.environ.get("ISAACSIM_PATH")
    wheel_sim = ROOT / "third_party/IsaacLab/.venv/bin/isaacsim"
    sim_detail = sim_path if sim_path else str(wheel_sim) if wheel_sim.exists() else "not installed"
    checks.append(status(bool((sim_path and Path(sim_path).exists()) or wheel_sim.exists()), "Isaac Sim 6.1", sim_detail))

    as2_usd = ROOT / "assets/robots/as2/as2/as2.usda"
    checks.append(status(as2_usd.is_file(), "As2 USD", str(as2_usd)))

    required_ready = all(checks)
    print("\nPreparation status:", "Isaac Sim/Lab and As2 base are ready" if required_ready else "resolve missing checks")
    raise SystemExit(0 if required_ready else 1)


if __name__ == "__main__":
    main()
