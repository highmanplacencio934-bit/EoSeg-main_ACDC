"""Validate ROI reports independently of removed inference-specific publication stages."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import fcntl
import hashlib
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/ablations/journal_20261025"
OUT = BASE / "optimization"
SOURCES = ["training/synapse_roi_metrics.py", "training/synapse_roi_evaluation.py",
           "scripts/run_synapse_roi.py", "scripts/validate_synapse_roi_acceleration.py",
           "scripts/finalize_synapse_roi_readiness.py",
           "tests/test_synapse_roi_metrics.py", "tests/test_synapse_roi_evaluation.py",
           "tests/test_synapse_roi_readiness.py"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name):
    return json.loads((OUT / name).read_text())


def write(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def _finite_positive(value, name):
    import math
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise RuntimeError("Invalid positive finite timing: " + name)


def validate_reports(cpu, gpu, fit, expected_cases):
    import math
    import statistics
    expected = set(expected_cases)
    if len(expected_cases) != 12 or len(expected) != 12:
        raise RuntimeError("Expected fixed twelve-case list")
    if cpu.get("cases") != expected_cases or cpu.get("repeats", 0) < 2:
        raise RuntimeError("CPU repeat/list contract mismatch")
    if cpu.get("num_cpu_workers") != 1 or cpu.get("gpu_used") is not False:
        raise RuntimeError("CPU validation was not the selected single-worker CPU path")
    comparisons = cpu.get("comparisons", [])
    if len(comparisons) != 12 * cpu["repeats"]:
        raise RuntimeError("Incomplete CPU comparisons")
    totals = []
    valid_statuses = {"nonempty", "empty_gt_legacy_zero", "empty_pred_legacy_zero", "both_empty_legacy_zero"}
    for repeat in range(cpu["repeats"]):
        items = [x for x in comparisons if x["repeat"] == repeat]
        if len(items) != 12 or {x["case"] for x in items} != expected:
            raise RuntimeError("Missing or duplicate CPU case")
        sums = {"original": 0.0, "roi": 0.0}
        for item in items:
            if item.get("exact") is not True:
                raise RuntimeError("A CPU comparison was not exact")
            by_mode = {}
            for mode in sums:
                entry = item[mode]
                _finite_positive(entry["cached_case_total_seconds"], mode)
                _finite_positive(entry["all_organs_seconds"], mode)
                values = entry["organs"]
                if len(values) != 8 or {x["label_id"] for x in values} != set(range(1, 9)):
                    raise RuntimeError("Missing or duplicate CPU organ")
                by_mode[mode] = {x["label_id"]: x["metrics"] for x in values}
                for metrics in by_mode[mode].values():
                    if metrics.get("status") not in valid_statuses:
                        raise RuntimeError("Unknown metric status")
                    for key in ("dice", "hd95_mm", "hd95_voxel"):
                        if not math.isfinite(metrics[key]) or metrics[key] < 0:
                            raise RuntimeError("Nonfinite or negative metric")
                    if metrics["dice"] > 1:
                        raise RuntimeError("Dice outside 0..1")
                sums[mode] += entry["cached_case_total_seconds"]
            if by_mode["original"] != by_mode["roi"]:
                raise RuntimeError("CPU report contains unequal actual metric dictionaries")
        totals.append(sums)
    medians = {mode: statistics.median(x[mode] for x in totals) for mode in ("original", "roi")}
    if cpu.get("median_seconds") != medians:
        raise RuntimeError("CPU aggregate timing does not match individual complete cases")
    speedup = medians["original"] / medians["roi"]
    _finite_positive(cpu["speedup"], "speedup")
    if cpu["speedup"] != speedup or speedup <= 1:
        raise RuntimeError("Invalid ROI timing benefit")
    if cpu.get("compared_case_classes") != len(comparisons) * 8:
        raise RuntimeError("Incorrect numeric comparison count")
    results = gpu.get("cases", [])
    if len(results) != 12 or [x["case"] for x in results] != expected_cases:
        raise RuntimeError("Incomplete or out-of-order GPU cases")
    if gpu.get("precision") != "16-mixed" or gpu.get("compile_test") is not True or gpu.get("compiled_forward_observed") is not True:
        raise RuntimeError("Wrong GPU precision/compile path")
    if gpu.get("num_cases") != 12:
        raise RuntimeError("Wrong GPU case count")
    _finite_positive(gpu["seconds"], "GPU total")
    for row in results:
        for key in ("dice_per_class", "hd95_mm_per_class", "hd95_voxel_per_class"):
            values = row.get(key, [])
            if len(values) != 8 or any(not math.isfinite(v) or v < 0 for v in values):
                raise RuntimeError("Incomplete/nonfinite GPU metrics")
        if any(v > 1 for v in row["dice_per_class"]):
            raise RuntimeError("GPU Dice outside 0..1")
    if fit.get("attempted_train_batches") != 3 or len(fit.get("finite_losses", [])) != 3 or any(not math.isfinite(x) for x in fit["finite_losses"]):
        raise RuntimeError("Incomplete/nonfinite fit smoke")
    for key, value in {"batch_size": 16, "num_workers": 8, "img_size": [384, 384], "patch_size": 16, "precision": "16-mixed", "compile_disabled": False, "evaluation_cases": 12}.items():
        if fit.get(key) != value:
            raise RuntimeError("Fit contract changed: " + key)
    _finite_positive(fit["seconds"], "fit smoke time")




