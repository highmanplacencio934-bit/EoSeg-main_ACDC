"""ACDC single-view Dice-only inference with optional metric ROI."""
from contextlib import contextmanager, nullcontext
from types import MethodType

import numpy as np
import torch

from training.acdc_mask_classification_semantic import ACDCMaskClassificationSemantic

_MISSING = object()


@contextmanager
def evaluation_scope(model, roi_metrics=True):
    model = getattr(model, "_orig_mod", model)
    if not isinstance(model, ACDCMaskClassificationSemantic):
        raise TypeError("ACDC evaluation requires ACDCMaskClassificationSemantic")
    if not isinstance(roi_metrics, bool):
        raise TypeError("roi_metrics must be boolean")
    if getattr(model, "_acdc_evaluation_active", False):
        raise RuntimeError("Nested ACDC evaluation scopes are unsupported")
    names = ("calculate_metric_percase_baseline",
             "roi_metrics_enabled", "acdc_evaluation_contract",
             "_acdc_evaluation_active")
    previous = {name: model.__dict__.get(name, _MISSING) for name in names}
    original_metric = model.calculate_metric_percase_baseline
    try:
        model._acdc_evaluation_active = True
        model.roi_metrics_enabled = roi_metrics
        model.acdc_evaluation_contract = {
            "dataset": "ACDC", "inference": "single_view",
            "roi_metrics": roi_metrics, "foreground_ids": [1, 2, 3],
            "hd95_mm": False, "hd95_voxel": False,
        }
        if roi_metrics:
            def metric(this, pred, gt):
                pred, gt = np.asarray(pred), np.asarray(gt)
                if pred.shape != gt.shape or pred.ndim not in (2, 3):
                    raise ValueError("Matching 2D/3D ACDC masks required")
                foreground = (pred != 0) | (gt != 0)
                if foreground.any():
                    coordinates = np.where(foreground)
                    bounds = tuple(slice(int(axis.min()), int(axis.max()) + 1)
                                   for axis in coordinates)
                    pred, gt = pred[bounds], gt[bounds]
                return original_metric(pred, gt)
            model.calculate_metric_percase_baseline = MethodType(metric, model)
        yield model.acdc_evaluation_contract
    finally:
        for name, value in previous.items():
            if value is _MISSING:
                model.__dict__.pop(name, None)
            else:
                setattr(model, name, value)


@contextmanager
def compiled_test_scope(model):
    # Fit compiles validation_step too; compile its metric helper in independent tests.
    names = ("forward", "_compute_case_metrics")
    previous = {name: model.__dict__.get(name, _MISSING) for name in names}
    try:
        with torch._dynamo.config.patch(capture_scalar_outputs=True, suppress_errors=True):
            model.forward = torch.compile(model.forward)
            metric = torch.compile(model._compute_case_metrics)

            def case_metrics(prediction, target):
                # NumPy CPU tensors need normal dispatch keys for metric compilation.
                with torch.inference_mode(False):
                    return metric(prediction, target)

            model._compute_case_metrics = case_metrics
            yield
    finally:
        for name, value in previous.items():
            if value is _MISSING:
                model.__dict__.pop(name, None)
            else:
                setattr(model, name, value)


@contextmanager
def inference_context(model, precision="16-mixed", compile_test=True,
                      roi_metrics=True):
    if precision not in ("16-mixed", "32-true"):
        raise ValueError("Supported inference precision is 16-mixed or 32-true")
    if precision == "16-mixed" and model.device.type != "cuda":
        raise ValueError("16-mixed inference requires CUDA; select 32-true for CPU")
    torch.set_float32_matmul_precision("medium")
    was_training = model.training
    try:
        model.eval()
        autocast = (torch.autocast("cuda", dtype=torch.float16)
                    if precision == "16-mixed" else nullcontext())
        with torch.inference_mode(), autocast, evaluation_scope(model, roi_metrics) as contract:
            with compiled_test_scope(model) if compile_test else nullcontext():
                yield dict(contract, precision=precision, compile_test=bool(compile_test))
    finally:
        model.train(was_training)
