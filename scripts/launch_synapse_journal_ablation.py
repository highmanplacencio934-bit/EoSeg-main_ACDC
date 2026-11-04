"""Shared journal process and metric utilities; inference-specific stages removed."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASE = ROOT / "outputs/ablations/journal_20261025"
PYTHON = "/Disk/HDD/Sk/miniconda3/envs/DeepLearning_xiakun/bin/python"
ENV = {"HF_HUB_OFFLINE": "1", "TORCH_HOME": "/Disk/HDD/Sk/.cache/torch",
       "TIMM_USE_OLD_CACHE": "1", "WANDB_MODE": "offline"}


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n")
    temp.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def environment():
    result = {**os.environ, **ENV}
    result.pop("CUDA_VISIBLE_DEVICES", None)
    return result


def identity(pid):
    try:
        path = Path("/proc") / str(pid)
        raw = (path / "stat").read_text()
        tail = raw[raw.rfind(")") + 2:].split()
        return {"pid": int(pid), "state": tail[0], "start_ticks": int(tail[19]),
                "command": (path / "cmdline").read_bytes().replace(b"\0", b" ").decode(),
                "cwd": os.readlink(path / "cwd")}
    except (FileNotFoundError, ProcessLookupError):
        return None


def wait_identity(pid, command_fragment):
    for _ in range(40):
        current = identity(pid)
        if current and command_fragment in current["command"]:
            return current
        time.sleep(0.05)
    raise RuntimeError("Could not verify freshly launched process identity")


def alive(owner):
    if not owner:
        return False
    current = identity(owner["pid"])
    return bool(current and current["state"] != "Z" and
                current["start_ticks"] == owner["start_ticks"] and
                current["command"] == owner["command"] and current["cwd"] == str(ROOT))






def set_plan_status(experiment, status, evidence):
    path = ROOT / ".codex/synapse_journal_ablation_plan.md"
    lines = path.read_text().splitlines()
    inside = False
    found = False
    for index, line in enumerate(lines):
        if line.startswith("## "):
            inside = line.startswith("## 11.")
        if not inside or not line.startswith("|"):
            continue
        columns = line.split("|")
        if len(columns) == 7 and columns[2].strip() == experiment:
            columns[4], columns[5] = " " + status + " ", " " + evidence + " "
            lines[index] = "|".join(columns)
            found = True
    if not found:
        raise ValueError("Experiment row missing from section 11: " + experiment)
    path.write_text("\n".join(lines) + "\n")


def voxel_mode(folder, physical=None):
    """Absent flags mean the historical dual-unit protocol; False is explicit."""
    units_path = folder / "metric_units.json"
    units = read(units_path) if units_path.is_file() else {}
    enabled = units.get("hd95_voxel_enabled", True)
    if type(enabled) is not bool:
        raise RuntimeError("Invalid HD95 voxel mode metadata")
    if physical is not None:
        reported = physical.get("hd95_voxel_enabled", True)
        if type(reported) is not bool or reported != enabled:
            raise RuntimeError("HD95 voxel mode mismatch between units and summary")
        if not enabled and ("mean_hd95_voxel" not in physical or physical["mean_hd95_voxel"] is not None):
            raise RuntimeError("Disabled voxel summary must be null")
    return enabled


def validate_voxel_value(value, enabled):
    if not enabled:
        if value not in ("", None):
            raise RuntimeError("Disabled voxel metric must be empty")
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError("Invalid numeric metric") from error
    if not math.isfinite(number) or number < 0:
        raise RuntimeError("Invalid numeric metric")
    return number


def validate_metric_row(row, voxel_enabled):
    for key in ("dice", "hd95_mm"):
        try:
            value = float(row[key])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("Invalid numeric metric") from error
        if not math.isfinite(value) or value < 0 or (key == "dice" and value > 1):
            raise RuntimeError("Invalid numeric metric")
    if "hd95_voxel" not in row:
        raise RuntimeError("Missing voxel metric column")
    validate_voxel_value(row["hd95_voxel"], voxel_enabled)
    validate_voxel_row_mode(row, voxel_enabled)


def validate_voxel_row_mode(row, voxel_enabled):
    if "hd95_voxel_enabled" in row:
        value = row["hd95_voxel_enabled"]
        row_mode = value if type(value) is bool else {"True": True, "False": False,
                    "true": True, "false": False}.get(value)
        if type(row_mode) is not bool or row_mode != voxel_enabled:
            raise RuntimeError("HD95 voxel mode mismatch in metric row")


def format_voxel(value):
    return "未计算" if value in (None, "") else f"{float(value):.6f}"








