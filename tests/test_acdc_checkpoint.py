"""Real Lightning regression: an early best must not freeze the rolling last."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset
from lightning.pytorch import LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint

if os.environ.get("ACDC_TEST_LEGACY_CHECKPOINT") == "1":
    ACDCModelCheckpoint = ModelCheckpoint
else:
    from training.acdc_checkpoint import ACDCModelCheckpoint
from training.acdc_test_logger import ACDCTestLogger


class TinyRun(LightningModule):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.trace = []

    def training_step(self, batch, batch_idx):
        loss = (self.weight * batch[0]).square().mean()
        self.trace.append((self.current_epoch, batch_idx, float(self.weight.detach()), float(loss.detach())))
        return loss

    def validation_step(self, batch, batch_idx):
        self.log("score", 0.9 - 0.1 * self.current_epoch)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


def run(callback_type, directory, at_train_end=False, save_last=True, every=1, epochs=3):
    seed_everything(0, workers=True)
    model = TinyRun()
    cb = callback_type(dirpath=directory, monitor="score", mode="max",
        filename="best_test", auto_insert_metric_name=False, enable_version_counter=False,
        save_top_k=1, save_last=save_last, save_weights_only=True,
        every_n_epochs=every, save_on_train_epoch_end=at_train_end)
    trainer = Trainer(max_epochs=epochs, accelerator="cpu", devices=1, logger=False,
        callbacks=[cb], enable_progress_bar=False, enable_model_summary=False,
        num_sanity_val_steps=0, deterministic=True)
    batches = DataLoader(TensorDataset(torch.ones(2, 1)), batch_size=1)
    trainer.fit(model, batches, batches)
    return model, cb, torch.random.get_rng_state()


class ACDCCheckpointTests(unittest.TestCase):
    def test_last_advances_after_best_for_both_hook_locations(self):
        for at_train_end in (False, True):
            with self.subTest(at_train_end=at_train_end), tempfile.TemporaryDirectory() as tmp:
                model, cb, _ = run(ACDCModelCheckpoint, tmp, at_train_end)
                best = torch.load(cb.best_model_path, weights_only=True)
                last = torch.load(cb.last_model_path, weights_only=True)
                self.assertEqual(best["epoch"], 0)
                self.assertEqual(last["epoch"], 2)
                self.assertEqual(last["global_step"], 6)
                self.assertTrue(torch.equal(last["state_dict"]["weight"], model.weight.detach()))
                self.assertFalse(torch.equal(last["state_dict"]["weight"], best["state_dict"]["weight"]))
                self.assertEqual({q.name for q in Path(tmp).glob("*.ckpt")}, {"best_test.ckpt", "last.ckpt"})

    def test_saving_change_does_not_change_training_or_rng(self):
        with tempfile.TemporaryDirectory() as tmp:
            old, _, old_rng = run(ModelCheckpoint, str(Path(tmp) / "old"))
            new, _, new_rng = run(ACDCModelCheckpoint, str(Path(tmp) / "new"))
            self.assertEqual(new.trace, old.trace)
            self.assertTrue(torch.equal(new.weight, old.weight))
            self.assertTrue(torch.equal(new_rng, old_rng))

    def test_save_last_false_and_epoch_interval_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, cb, _ = run(ACDCModelCheckpoint, str(Path(tmp) / "off"), save_last=False)
            self.assertEqual(cb.last_model_path, "")
            self.assertFalse((Path(tmp) / "off/last.ckpt").exists())
            _, cb, _ = run(ACDCModelCheckpoint, str(Path(tmp) / "interval"), every=2, epochs=4)
            self.assertEqual(torch.load(cb.best_model_path, weights_only=True)["epoch"], 1)
            self.assertEqual(torch.load(cb.last_model_path, weights_only=True)["epoch"], 3)

    def test_epoch_best_output_uses_running_best_not_current_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            scores = (.9, .8, .95, .95)
            callback = ACDCTestLogger()
            model = SimpleNamespace(roi_metrics_enabled=True, eval_case_results=[
                {"case": str(i), "metrics": {"mean_dice": .5, "dice_class_1": .5,
                 "dice_class_2": .5, "dice_class_3": .5}} for i in range(40)])
            trainer = SimpleNamespace(log_dir=tmp, default_root_dir=tmp, sanity_checking=False,
                callback_metrics={}, datamodule=SimpleNamespace(eval_dataset=range(40)),
                current_epoch=0, global_step=0, max_epochs=4, limit_val_batches=1.0,
                fast_dev_run=False, is_global_zero=True)
            lines = []
            for e, score in enumerate(scores):
                trainer.current_epoch = e
                trainer.global_step = 2 * (e + 1)
                trainer.callback_metrics = {"metrics/test_mean_dice": score,
                    **{f"metrics/test_dice_class_{i}": score for i in (1, 2, 3)}}
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    callback.on_validation_end(trainer, model)
                lines.append(output.getvalue().strip())
            for line, epoch, best_epoch, score in zip(lines, (1,2,3,4), (1,1,3,3), (90,90,95,95)):
                self.assertIn(f"epoch={epoch}/4 ", line)
                self.assertIn(f"best_epoch={best_epoch} ", line)
                self.assertIn(f"best_dice={score:.6f}%", line)
            summary = json.loads((Path(tmp) / "selection_summary.json").read_text())
            self.assertEqual(summary["best_epoch"], 3)
            self.assertEqual(summary["best_mean_dice"], .95)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
