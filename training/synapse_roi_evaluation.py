"""Instance-scoped ROI metrics for independent Synapse fit/test entry points.

The original inference functions, wrappers, default CLI and metric remain intact.
Only CPU metric execution changes; geometry and the legacy empty policy persist.
"""
from contextlib import contextmanager
from pathlib import Path
from types import MethodType
import json
import logging
import time

import numpy as np

from training.synapse_physical_semantic import SynapsePhysicalHD95Semantic
from training.synapse_roi_metrics import binary_metrics_mm_roi

_MISSING = object()


def _set_back(model, name, original):
    if original is _MISSING:
        if name in model.__dict__:
            delattr(model, name)
    else:
        setattr(model, name, original)


@contextmanager
def roi_metrics_scope(model, capture_dir=None):
    """Bind ROI metrics to one model, restoring every binding even on failure.

    Capture reconstructs the exact original integer prediction from the eight
    class masks passed by the unchanged 1-view evaluator. It never reruns the
    forward or derives predictions from ground-truth labels.
    """
    if not isinstance(model, SynapsePhysicalHD95Semantic):
        raise TypeError("ROI evaluation requires an explicit physical Synapse model wrapper")
    geometry = getattr(model, "physical_geometry", None)
    if geometry is None or not callable(getattr(geometry, "for_case", None)):
        raise TypeError("ROI evaluation requires verified physical_geometry.for_case")
    if any(c.__module__ == "training.synapse_teacher_semantic" for c in type(model).__mro__):
        raise TypeError("The independent teacher evaluator is not supported by ROI scope")
    if getattr(model, "_synapse_roi_scope_active", False):
        raise RuntimeError("Nested ROI scopes on the same model are unsupported")
    capture_root = Path(capture_dir) if capture_dir is not None else None
    if capture_root is not None:
        capture_root.mkdir(parents=True, exist_ok=True)

    names = ("calculate_metric_percase", "evaluate_synapse_case", "_synapse_roi_scope_active", "roi_evaluation_contract",
             "roi_evaluation_case_timings")
    old = {name: model.__dict__.get(name, _MISSING) for name in names}
    original_evaluate = model.evaluate_synapse_case
    active_capture = {}
    timings = []
    contract = {"implementation": "union_foreground_bbox_halo1_original_medpy",
                "cpu_workers": 1,
                "geometry": "full-case shape and label fingerprint before ROI",
                "metric_policy": ("original Dice, HD95_mm, HD95_voxel and legacy empty status"
                                  if model.compute_hd95_voxel else "original Dice, HD95_mm; voxel disabled; legacy empty status"),
                "hd95_voxel_enabled": model.compute_hd95_voxel}
    try:
        model._synapse_roi_scope_active = True
        model.roi_evaluation_contract = contract
        model.roi_evaluation_case_timings = timings

        def calculate(self, pred, gt):
            context = self._physical_case_context
            if context is None:
                raise RuntimeError("ROI metrics require verified full-case geometry context")
            spacing, records = context
            start = time.perf_counter()
            result = binary_metrics_mm_roi(pred, gt, spacing, compute_hd95_voxel=self.compute_hd95_voxel)
            elapsed = time.perf_counter() - start
            if active_capture:
                active_capture["metrics_seconds"] += elapsed
                class_id = active_capture["calls"] + 1
                if class_id >= self.num_classes:
                    raise RuntimeError("Too many class metric calls during capture")
                pred_mask, gt_mask = np.asarray(pred, dtype=bool), np.asarray(gt, dtype=bool)
                expected = active_capture["gt"]
                if pred_mask.shape != expected.shape or not np.array_equal(gt_mask, expected == class_id):
                    raise ValueError("Capture mask does not match the verified full-case class label")
                prediction = active_capture["prediction"]
                if np.any(pred_mask & (prediction != 0)):
                    raise ValueError("Original class predictions overlap")
                prediction[pred_mask] = class_id
                active_capture["calls"] += 1
            records.append(result)
            return result["dice"], result["hd95_mm"]

        model.calculate_metric_percase = MethodType(calculate, model)

        def evaluate(self, batch):
            if active_capture:
                raise RuntimeError("Concurrent/reentrant prediction capture is unsupported")
            started = time.perf_counter()
            case = batch["case_name"]
            if capture_root is not None:
                # Validate against complete original GT before any metric ROI.
                label = batch["label"].detach().cpu().numpy()
                if label.ndim == 4 and label.shape[0] == 1:
                    label = label[0]
                geometry = self.physical_geometry.for_case(case, label)
                target = capture_root / (case + ".npz")
                metadata_path = capture_root / (case + ".json")
                if target.exists() or metadata_path.exists():
                    raise FileExistsError("Refusing to overwrite prediction capture for " + case)
                active_capture.update(gt=label, prediction=np.zeros(label.shape, dtype=np.uint8),
                                      calls=0, metrics_seconds=0.0)
            try:
                result = original_evaluate(batch)
                total = time.perf_counter() - started
                if capture_root is not None:
                    if active_capture["calls"] != self.num_classes - 1:
                        raise RuntimeError("Incomplete original foreground prediction capture")
                    # Recheck the unmodified full GT, not the ROI.
                    self.physical_geometry.for_case(case, active_capture["gt"])
                    with target.open("xb") as handle:
                        np.savez_compressed(handle, prediction=active_capture["prediction"],
                                            gt=active_capture["gt"].astype(np.uint8))
                    row = {"case": case, "shape_zyx": list(label.shape),
                           "spacing_zyx_mm": list(geometry.spacing_zyx_mm),
                           "label_sha256": geometry.label_sha256,
                           "checkpoint_epoch": self.physical_checkpoint_epoch,
                           "checkpoint_global_step": self.physical_checkpoint_global_step,
                           "case_total_seconds": total,
                           "roi_metrics_seconds": active_capture["metrics_seconds"],
                           "prediction_file": target.name, "contract": contract}
                    with metadata_path.open("x", encoding="utf-8") as handle:
                        json.dump(row, handle, ensure_ascii=False, indent=2)
                    timings.append(row)
                return result
            finally:
                active_capture.clear()
        model.evaluate_synapse_case = MethodType(evaluate, model)

        logging.info("Independent ROI evaluation enabled: %s", contract)
        yield model
    finally:
        for name in names:
            _set_back(model, name, old[name])
