"""Regressions for complete ACDC manifests and isolated evaluation/V2 interfaces."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import yaml
from timm.models.vision_transformer import VisionTransformer

from datasets.acdc_splits import validate_splits
from main import LightningCLI
from training.acdc_test_logger import ACDCTestLogger
from training.acdc_evaluation import evaluation_scope, inference_context
from training.acdc_deterministic import source_hashes, require_readiness
from scripts import visualize_acdc_results as visualization

ROOT = Path(__file__).resolve().parents[1]


def tiny_factory(name, **kwargs):
    return VisionTransformer(img_size=kwargs["img_size"], patch_size=kwargs["patch_size"],
                             embed_dim=32, depth=4, num_heads=4, num_classes=0)


def small_config():
    cfg = yaml.safe_load((ROOT / "configs/acdc/vit_query_mul_scale_fusion.yaml").read_text())
    cfg["data"]["init_args"]["img_size"] = [32, 32]
    cfg["model"]["init_args"]["img_size"] = [32, 32]
    cfg["model"]["init_args"]["network"]["init_args"]["encoder"]["init_args"]["img_size"] = [32, 32]
    cfg["model"]["init_args"]["num_points"] = 32
    return cfg


def small_model():
    cfg = visualization.apply_cli_style_links(small_config(), Path("fixture.ckpt"))
    with patch("models.vit.timm.create_model", side_effect=tiny_factory):
        return visualization.instantiate_from_config(cfg["model"])


class ACDCAdaptationFixTests(unittest.TestCase):
    def test_missing_evaluation_frame_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "train.txt").write_bytes((ROOT / "ACDC/lists_ACDC/train.txt").read_bytes())
            names = (ROOT / "ACDC/lists_ACDC/test.txt").read_text().splitlines()[1:]
            for split in ("val", "test"):
                (directory / (split + ".txt")).write_text("\n".join(names) + "\n")
            with self.assertRaisesRegex(ValueError, "40 evaluation volumes"):
                validate_splits(ROOT / "ACDC", directory, "fit")

    def test_missing_training_slice_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            for split in ("val", "test"):
                (directory / (split + ".txt")).write_bytes((ROOT / "ACDC/lists_ACDC" / (split + ".txt")).read_bytes())
            names = (ROOT / "ACDC/lists_ACDC/train.txt").read_text().splitlines()[1:]
            (directory / "train.txt").write_text("\n".join(names) + "\n")
            with self.assertRaisesRegex(ValueError, "1512 training slices"):
                validate_splits(ROOT / "ACDC", directory, "fit")

    def test_visualizer_rejects_missing_query(self):
        cfg = small_config()
        model = small_model()
        state = model.state_dict()
        del state["network.query_embed"]
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "missing.ckpt"
            torch.save({"state_dict": state}, p)
            with patch("models.vit.timm.create_model", side_effect=tiny_factory):
                with self.assertRaisesRegex(RuntimeError, "network.query_embed"):
                    visualization.load_model(cfg, p, torch.device("cpu"))

    def test_complete_visualization_checkpoint_exact(self):
        cfg = small_config()
        model = small_model()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "complete.ckpt"
            torch.save({"state_dict": model.state_dict()}, p)
            with patch("models.vit.timm.create_model", side_effect=tiny_factory):
                restored = visualization.load_model(cfg, p, torch.device("cpu"))
            for name, tensor in model.state_dict().items():
                torch.testing.assert_close(restored.state_dict()[name], tensor, rtol=0, atol=0)

    def test_roi_dice_matches_full_masks_and_restores_model(self):
        model = small_model()
        rng = np.random.default_rng(9)
        masks = [(rng.random((5, 21, 23)) > .8, rng.random((5, 21, 23)) > .8),
                 (np.zeros((3, 4, 5)), np.zeros((3, 4, 5))),
                 (np.ones((3, 4, 5)), np.zeros((3, 4, 5))),
                 (np.zeros((3, 4, 5)), np.ones((3, 4, 5)))]
        expected = [model.calculate_metric_percase_baseline(p, g) for p, g in masks]
        original = model.calculate_metric_percase_baseline
        with evaluation_scope(model, roi_metrics=True):
            self.assertEqual([model.calculate_metric_percase_baseline(p, g) for p, g in masks], expected)
        self.assertEqual(model.calculate_metric_percase_baseline, original)
        self.assertFalse(hasattr(model, "_acdc_evaluation_active"))


    def test_identity_evaluation_is_exact(self):
        model = small_model().eval()
        volume = np.random.default_rng(3).random((1, 32, 32), dtype=np.float32)
        with torch.inference_mode():
            expected = model._run_case_inference(volume)
        with inference_context(model, precision="32-true", compile_test=False):
            actual = model._run_case_inference(volume)
        np.testing.assert_array_equal(actual, expected)

    def test_compiled_independent_metrics_match_fit_compilation_and_restore(self):
        model = small_model().eval()
        prediction = np.zeros((2, 7, 9), dtype=np.int32)
        target = np.zeros_like(prediction)
        prediction[0, 1:4, 2:5] = 1
        target[0, 2:5, 3:6] = 1
        prediction[1, 1:5, 1:5] = 2
        target[1, 2:6, 1:5] = 2
        target[1, 0, 0] = 3
        old_capture = torch._dynamo.config.capture_scalar_outputs
        torch._dynamo.config.capture_scalar_outputs = True
        original_forward, original_metric = model.forward, model._compute_case_metrics
        try:
            eager = original_metric(prediction, target)
            expected = torch.compile(original_metric)(prediction, target)
            self.assertNotEqual(eager, expected)
            def evaluate(module, **kwargs):
                with torch.inference_mode():
                    result = module._compute_case_metrics(prediction, target)
                    self.assertTrue(torch.is_inference_mode_enabled())
                    return result

            caller = SimpleNamespace(
                config={"subcommand": "test", "test": {"compile_test": True, "compile_disabled": False}},
                trainer=SimpleNamespace(test=evaluate),
            )
            self.assertEqual(LightningCLI.test(caller, model), expected)
            self.assertEqual(model.forward, original_forward)
            self.assertEqual(model._compute_case_metrics, original_metric)
            self.assertEqual(model._compute_case_metrics(prediction, target), eager)
        finally:
            torch._dynamo.config.capture_scalar_outputs = old_capture

    def test_test_summary_matches_fit_logged_scalar_representation(self):
        model = small_model()
        rows = []
        for scores in ((1 / 3, .127, .71), (.21, 4 / 7, .063)):
            values = {f"dice_class_{i}": float(np.float32(v)) for i, v in enumerate(scores, 1)}
            values["mean_dice"] = float(np.mean(list(values.values())))
            rows.append(values)
        model.eval_epoch_metrics = rows
        logged = {}

        def log(name, value, **kwargs):
            logged[name] = torch.tensor(value, dtype=torch.get_default_dtype()).item()

        with patch.object(model, "log", side_effect=log):
            model.on_validation_epoch_end()
        model.test_case_results = [{"case": str(i), "metrics": values} for i, values in enumerate(rows)]
        with tempfile.TemporaryDirectory() as tmp:
            ACDCTestLogger().on_test_end(SimpleNamespace(log_dir=tmp, default_root_dir=tmp), model)
            summary = json.loads((Path(tmp) / "acdc_independent_test_summary.json").read_text())
        self.assertEqual(summary["mean_dice"], logged["metrics/test_mean_dice"])
        for c in range(1, 4):
            self.assertEqual(summary[f"dice_class_{c}"], logged[f"metrics/test_dice_class_{c}"])

    def test_fit_summary_records_actual_roi_switch(self):
        for enabled in (False, True):
            with tempfile.TemporaryDirectory() as tmp:
                values = {"mean_dice": .5, **{f"dice_class_{i}": .5 for i in (1, 2, 3)}}
                module = SimpleNamespace(roi_metrics_enabled=enabled,
                    eval_case_results=[{"case": str(i), "metrics": values} for i in range(40)])
                metrics = {"metrics/test_mean_dice": .5,
                           **{f"metrics/test_dice_class_{i}": .5 for i in (1, 2, 3)}}
                trainer = SimpleNamespace(log_dir=tmp, default_root_dir=tmp, sanity_checking=False,
                    callback_metrics=metrics, datamodule=SimpleNamespace(eval_dataset=range(40)),
                    current_epoch=0, global_step=0, max_epochs=400, limit_val_batches=1.0, fast_dev_run=False)
                ACDCTestLogger().on_validation_end(trainer, module)
                summary = json.loads((Path(tmp) / "selection_summary.json").read_text())
                self.assertEqual(summary["roi_metrics"], enabled)

    def test_acdc_v2_sources_and_wrong_dataset_gate(self):
        hashes = source_hashes(ROOT)
        self.assertIn("configs/acdc/deterministic_v2/ccps.yaml", hashes)
        self.assertIn("ACDC/lists_ACDC/train.txt", hashes)
        self.assertFalse(any("lists_Synapse" in name for name in hashes))
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "readiness.json"
            p.write_text(json.dumps({"dataset": "Synapse", "passed": True}))
            with self.assertRaisesRegex(RuntimeError, "own dataset"):
                require_readiness(p, ROOT, "baseline")

    def test_innovation_configs_keep_acdc_science(self):
        for filename, expected in (("vit_query_mul_scale_fusion_asee.yaml", "EoSegMultiQSegFusionASEE"),
                                   ("vit_query_mul_scale_fusion_asee_ccps.yaml", "EoSegMultiQSegFusionASEECCPS")):
            cfg = yaml.safe_load((ROOT / "configs/acdc" / filename).read_text())
            self.assertEqual(cfg["model"]["class_path"], "training.acdc_mask_classification_semantic.ACDCMaskClassificationSemantic")
            self.assertTrue(cfg["model"]["init_args"]["network"]["class_path"].endswith(expected))
            self.assertEqual(cfg["model"]["init_args"]["network"]["init_args"]["num_q"], 6)
            self.assertEqual(cfg["trainer"]["max_epochs"], 400)
            self.assertEqual(float(cfg["model"]["init_args"]["lr"]), 5e-4)
            self.assertEqual(cfg["data"]["init_args"]["img_size"], [224, 224])


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
