"""Dice-only ACDC records; use post-module hooks so epochs are not shifted."""
import csv
import json
from pathlib import Path

import numpy as np
import torch

from lightning.pytorch.callbacks import Callback
from training.run_artifacts import write_run_manifest


class ACDCTestLogger(Callback):
    def __init__(self, history_filename="acdc_test_history.csv", best_filename="acdc_best_result.txt"):
        super().__init__()
        self.history_filename = history_filename
        self.best_filename = best_filename
        self.best_score = float("-inf")
        self.best_epoch = None

    def _run_dir(self, trainer):
        return Path(trainer.log_dir or trainer.default_root_dir)

    @staticmethod
    def _append(path, rows):
        if not rows:
            return
        exists = path.exists()
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            if not exists:
                writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _write_json(path, value):
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")

    def on_fit_start(self, trainer, pl_module):
        run_dir = self._run_dir(trainer)
        run_dir.mkdir(parents=True, exist_ok=True)
        if (run_dir / self.history_filename).exists():
            raise FileExistsError("Use a new output/logger version; never append a second run")
        write_run_manifest(trainer, pl_module, run_dir)

    def _case_rows(self, pl_module, common, stage):
        results = pl_module.eval_case_results if stage == "fit" else pl_module.test_case_results
        return [dict(common, case=result["case"], **result["metrics"], hd95_mm=None, hd95_voxel=None)
                for result in results]

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking or not pl_module.eval_case_results:
            return
        metrics = trainer.callback_metrics
        required = ["metrics/test_mean_dice"] + [f"metrics/test_dice_class_{c}" for c in range(1, 4)]
        if not all(key in metrics for key in required):
            raise RuntimeError("Missing current-epoch ACDC Dice metrics")
        data = getattr(trainer, "datamodule", None)
        dataset = getattr(data, "eval_dataset", None)
        expected = len(dataset) if dataset is not None else 40
        limit = getattr(trainer, "limit_val_batches", 1.0)
        partial_requested = ((isinstance(limit, int) and limit < expected)
                             or (isinstance(limit, float) and limit < 1.0)
                             or bool(getattr(trainer, "fast_dev_run", False)))
        complete = len(pl_module.eval_case_results) == expected
        if dataset is not None and not partial_requested and not complete:
            raise RuntimeError("ACDC evaluation is incomplete for this epoch")
        epoch = int(trainer.current_epoch) + 1
        values = {c: float(metrics[f"metrics/test_dice_class_{c}"]) for c in range(1, 4)}
        score = float(metrics["metrics/test_mean_dice"])
        improved = score > self.best_score
        if improved:
            self.best_score, self.best_epoch = score, epoch
        if getattr(trainer, "is_global_zero", True):
            print(f"[EPOCH-BEST] epoch={epoch}/{trainer.max_epochs} "
                  f"best_epoch={self.best_epoch} best_dice={100 * self.best_score:.6f}%",
                  flush=True)
        run_dir = self._run_dir(trainer)
        common = {"epoch": epoch, "global_step": trainer.global_step}
        self._append(run_dir / self.history_filename, [dict(common, avg_dsc=score, rv=values[1],
                     myo=values[2], lv=values[3], hd95_mm=None, hd95_voxel=None, is_best=int(improved))])
        self._append(run_dir / "acdc_test_case_metrics.csv", self._case_rows(pl_module, common, "fit"))
        summary = {
            "epoch_numbering": "one-based", "max_epochs": trainer.max_epochs,
            "last_evaluated_epoch": epoch, "best_epoch": self.best_epoch,
            "best_mean_dice": self.best_score, "evaluation_split": "test",
            "last_evaluated_volumes": len(pl_module.eval_case_results),
            "test_used_for_selection": True,
            "evaluation_complete": complete,
            "selection_policy": "every epoch, complete test volumes; test used for model selection",
            "roi_metrics": bool(getattr(pl_module, "roi_metrics_enabled", False)), "hd95_mm": None, "hd95_voxel": None,
            "hd95_computed": False,
            "class_names_source": "priors/acdc_label_map.verified.json; official legend and sampled training-H5 semantics; raw geometry unverified",
        }
        self._write_json(run_dir / "selection_summary.json", summary)
        if improved:
            text = (f"best_epoch={epoch}\nbest_avg_dsc={score:.12g}\n"
                    f"best_rv={values[1]:.12g}\nbest_myo={values[2]:.12g}\nbest_lv={values[3]:.12g}\n"
                    "dice_unit=0..1\nhd95_mm=not_computed\nhd95_voxel=not_computed\n")
            (run_dir / self.best_filename).write_text(text, encoding="utf-8")

    def on_fit_end(self, trainer, pl_module):
        run_dir = self._run_dir(trainer)
        path = run_dir / "selection_summary.json"
        if path.is_file():
            summary = json.loads(path.read_text())
            cb = trainer.checkpoint_callback
            summary.update(best_checkpoint=cb.best_model_path if cb is not None else None,
                           last_checkpoint=cb.last_model_path if cb is not None else None)
            self._write_json(path, summary)

    def on_test_end(self, trainer, pl_module):
        results = pl_module.test_case_results
        if not results:
            return
        run_dir = self._run_dir(trainer)
        run_dir.mkdir(parents=True, exist_ok=True)
        # A second independent test needs its own output directory as well.
        if (run_dir / "acdc_independent_test_summary.json").exists():
            raise FileExistsError("Independent test output already exists")
        self._append(run_dir / "acdc_independent_test_cases.csv", self._case_rows(pl_module, {}, "test"))
        def logged_average(key):
            value = np.mean([result["metrics"][key] for result in results])
            # Match the scalar conversion performed by Lightning's fit-time log.
            return torch.tensor(value, dtype=torch.get_default_dtype(), device="cpu").item()

        summary = {f"dice_class_{c}": logged_average(f"dice_class_{c}") for c in range(1, 4)}
        summary.update(mean_dice=logged_average("mean_dice"), volumes=len(results),
                       evaluation_split="test", shares_cases_with_model_selection=True,
                       hd95_mm=None, hd95_voxel=None, hd95_computed=False,
                       roi_metrics=getattr(pl_module, "roi_metrics_enabled", False),
                       inference_contract=getattr(pl_module, "acdc_evaluation_contract", None))
        self._write_json(run_dir / "acdc_independent_test_summary.json", summary)
