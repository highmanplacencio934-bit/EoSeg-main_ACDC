"""ACDC-specific provenance and admission checks for execution V2."""
from pathlib import Path
import hashlib
import json
import time
import uuid

import torch
import torch._functorch.config as aot_config

from training.deterministic_protocol import (
    DeterministicProtocolRecord as BaseRecord, backend_state, write_json,
)
from training.deterministic_mask_loss import DeterministicMaskClassificationLoss

PROTOCOL_ID = "acdc_ce2d_stableseg_execution_v2"
NETWORKS = {
    "baseline": "EoSegMultiQSegFusion",
    "asee": "EoSegMultiQSegFusionASEE",
    "ccps": "EoSegMultiQSegFusionASEECCPS",
}


def source_hashes(root):
    root = Path(root)
    paths = set()
    for directory in ("models", "datasets", "training", "configs/acdc"):
        paths.update(p for p in (root / directory).rglob("*")
                     if p.suffix in (".py", ".yaml", ".yml"))
    for name in ("main.py", "requirements.txt", "sitecustomize.py",
                 "scripts/run_acdc_deterministic_v2.py", "scripts/validate_acdc_deterministic.py", "scripts/evaluate_acdc.py",
                 "scripts/visualize_acdc_results.py", "scripts/build_acdc_asee_prior.py",
                 "scripts/build_synapse_asee_prior.py",
                 "ACDC/lists_ACDC/train.txt", "ACDC/lists_ACDC/val.txt", "ACDC/lists_ACDC/test.txt",
                 "priors/acdc_asee_80_20_v1.pt", "priors/acdc_prior_manifest.json",
                 "priors/acdc_label_map.verified.json", "pretrained/acdc_pretrained_manifest.json"):
        paths.add(root / name)
    pretrained = json.loads((root / "pretrained/acdc_pretrained_manifest.json").read_text())
    paths.add(Path(pretrained["vision_encoder"]["path"]))
    encoder = root / pretrained["text_encoder"]["path"]
    paths.update(encoder / name for name in pretrained["text_encoder"]["file_sha256"])
    result = {}
    for path in sorted(paths):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(block)
        name = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
        result[name] = digest.hexdigest()
    return result


def require_readiness(path, root, profile):
    path = Path(path)
    if not path.is_file():
        raise RuntimeError("ACDC V2 readiness is missing; run the bounded validation first")
    readiness = json.loads(path.read_text())
    if readiness.get("dataset") != "ACDC" or readiness.get("protocol") != PROTOCOL_ID:
        raise RuntimeError("ACDC V2 requires its own dataset/protocol readiness")
    if readiness.get("passed") is not True or profile not in readiness.get("validated_profiles", []):
        raise RuntimeError("ACDC V2 model profile has not passed validation")
    if readiness.get("sources") != source_hashes(root):
        raise RuntimeError("ACDC V2 sources or assets changed after validation")
    gates = readiness.get("profile_gates", {}).get(profile, {})
    if any(gates.get("G" + str(i)) != "passed" for i in range(6)):
        raise RuntimeError("ACDC V2 requires G0-G5 for the selected profile")


class ACDCProtocolRecord(BaseRecord):
    def on_fit_start(self, trainer, pl_module):
        raw = getattr(pl_module, "_orig_mod", pl_module)
        if type(raw.network).__name__ != NETWORKS[self.profile]:
            raise ValueError("ACDC V2 profile does not match the actual network")
        if not torch.are_deterministic_algorithms_enabled() or torch.is_deterministic_algorithms_warn_only_enabled():
            raise RuntimeError("ACDC V2 requires strict deterministic algorithms")
        if not isinstance(raw.criterion, DeterministicMaskClassificationLoss):
            raise RuntimeError("ACDC deterministic criterion is missing")
        if raw.num_classes != 4 or raw.network.num_q != 6:
            raise ValueError("ACDC execution expects four semantic classes and six queries")
        if self.profile == "ccps":
            if not getattr(raw, "_ccps_execution_v2_boundary", False) or aot_config.backward_pass_autocast != "off":
                raise RuntimeError("CCPS V2 execution boundary is missing")
        split = trainer.datamodule.split_summary
        if split["counts"] != {"train": 1512, "val": 40, "test": 40}:
            raise ValueError("ACDC V2 requires the verified 80/20 dataset")
        run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "_" + uuid.uuid4().hex[:8]
        self.run_dir = self.output_dir / run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)
        write_json(self.run_dir / "protocol_manifest.json", {
            "dataset": "ACDC", "protocol": PROTOCOL_ID, "profile": self.profile,
            "probe_batches": self.probe_batches, "sources": source_hashes(self.root),
            "backend": backend_state(), "split_summary": split,
            "scientific_max_epochs": trainer.max_epochs,
            "estimated_stepping_batches": trainer.estimated_stepping_batches,
            "resolved_config": getattr(trainer, "baseline_resolved_config", None),
            "ccps_local_bn_eager_boundary": bool(getattr(raw, "_ccps_execution_v2_boundary", False)),
            "compiler_backward_assumption": aot_config.backward_pass_autocast,
            "hd95_mm": False, "hd95_voxel": False,
        })
