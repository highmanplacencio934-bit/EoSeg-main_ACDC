"""Independent ACDC single-view Dice-only test with optional metric ROI."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from main import LightningCLI
from training.lightning_module import LightningModule
from datasets.lightning_data_module import LightningDataModule
from training.acdc_evaluation import evaluation_scope


class ACDCEvaluationCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        super().add_arguments_to_parser(parser)
        parser.add_argument("--roi_metrics", type=bool, default=True)

    def test(self, model, **kwargs):
        options = self.config[self.config["subcommand"]]
        with evaluation_scope(model, options["roi_metrics"]):
            return super().test(model, **kwargs)


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] != "test":
        raise SystemExit("Use evaluate_acdc.py test with an ACDC config and checkpoint")
    ACDCEvaluationCLI(
        LightningModule, LightningDataModule, subclass_mode_model=True,
        subclass_mode_data=True, save_config_callback=None, seed_everything_default=0,
        trainer_defaults={"precision": "16-mixed", "devices": 1},
    )
