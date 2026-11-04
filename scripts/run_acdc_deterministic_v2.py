"""ACDC execution V2 with the existing deterministic loss and CCPS boundary."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lightning.pytorch.callbacks import LearningRateMonitor
from scripts.evaluate_acdc import ACDCEvaluationCLI
from training.lightning_module import LightningModule
from datasets.lightning_data_module import LightningDataModule
from training.acdc_evaluation import evaluation_scope
from training.acdc_deterministic import PROTOCOL_ID, NETWORKS, require_readiness, ACDCProtocolRecord
from training.deterministic_execution_v2 import (
    backend_state, configure_strict, restore_backend, install_criterion,
    execution_context, install_execution_boundary,
)
from main import LightningCLI


class ACDCExecutionV2CLI(ACDCEvaluationCLI):
    def __init__(self, *args, **kwargs):
        previous = backend_state()
        try:
            super().__init__(*args, **kwargs)
        except BaseException:
            restore_backend(previous)
            raise

    def add_arguments_to_parser(self, parser):
        super().add_arguments_to_parser(parser)
        parser.add_argument("--deterministic_enabled", type=bool, default=True)
        parser.add_argument("--deterministic_protocol", default=PROTOCOL_ID)
        parser.add_argument("--deterministic_profile", choices=tuple(NETWORKS), default="baseline")
        parser.add_argument("--deterministic_probe_batches", type=int, default=0)
        parser.add_argument("--deterministic_output_dir", default="outputs/determinism/" + PROTOCOL_ID)
        parser.add_argument("--deterministic_readiness", default="outputs/determinism/" + PROTOCOL_ID + "/readiness.json")

    def _options(self):
        stage = self.config.get("subcommand")
        return self.config[stage] if stage else self.config

    def before_instantiate_classes(self):
        super().before_instantiate_classes()
        options = self._options()
        if options["deterministic_protocol"] != PROTOCOL_ID:
            raise ValueError("Unknown ACDC execution protocol")
        batches = options["deterministic_probe_batches"]
        if not 0 <= batches <= 100:
            raise ValueError("ACDC probes must use 1..100 batches, or zero for admitted full training")
        self._previous_backend = configure_strict(enabled=options["deterministic_enabled"])
        if options["deterministic_enabled"] and self.config.get("subcommand") == "fit":
            if options["compile_disabled"]:
                raise ValueError("ACDC V2 fit preserves default outer compile")
            if batches == 0:
                require_readiness(options["deterministic_readiness"], ROOT, options["deterministic_profile"])

    def after_instantiate_classes(self):
        super().after_instantiate_classes()
        options = self._options()
        if options["deterministic_enabled"]:
            if type(self.model.network).__name__ != NETWORKS[options["deterministic_profile"]]:
                raise ValueError("ACDC V2 profile/network mismatch")
            install_criterion(self.model)
            install_execution_boundary(self.model, options["deterministic_profile"])
            if self.config.get("subcommand") == "fit":
                self.trainer.callbacks.append(ACDCProtocolRecord(
                    options["deterministic_output_dir"], ROOT,
                    options["deterministic_probe_batches"], options["deterministic_profile"]))

    def _run_acdc(self, stage, model, kwargs):
        options = self._options()
        try:
            with execution_context(options["deterministic_profile"], enabled=options["deterministic_enabled"]):
                with evaluation_scope(model, options["roi_metrics"]):
                    return getattr(LightningCLI, stage)(self, model, **kwargs)
        finally:
            restore_backend(self._previous_backend)

    def fit(self, model, **kwargs):
        return self._run_acdc("fit", model, kwargs)

    def test(self, model, **kwargs):
        return self._run_acdc("test", model, kwargs)


if __name__ == "__main__":
    ACDCExecutionV2CLI(
        LightningModule, LightningDataModule, subclass_mode_model=True,
        subclass_mode_data=True, save_config_callback=None, seed_everything_default=0,
        trainer_defaults={"precision": "16-mixed", "devices": 1,
                          "callbacks": [LearningRateMonitor(logging_interval="epoch")],
                          "gradient_clip_val": 0.01, "gradient_clip_algorithm": "norm"},
    )
