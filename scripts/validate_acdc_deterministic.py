"""Bounded GPU validation of the ACDC execution protocol, never a formal training run."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import pickle
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import yaml
from lightning.pytorch import Callback
from training.acdc_deterministic import PROTOCOL_ID, source_hashes
from training.deterministic_protocol import configure_strict, restore_backend, write_json


def digest(value):
    if torch.is_tensor(value):
        tensor = value.detach().contiguous()
        raw = tensor.reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
        return hashlib.sha256(str((tensor.dtype, tuple(tensor.shape))).encode() + raw).hexdigest()
    if isinstance(value, dict):
        return {str(k): digest(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [digest(v) for v in value]
    return value


def rng_state():
    return {
        "python": hashlib.sha256(pickle.dumps(random.getstate())).hexdigest(),
        "numpy": hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest(),
        "torch": digest(torch.get_rng_state()),
        "cuda": [digest(value) for value in torch.cuda.get_rng_state_all()],
    }


class Trace(Callback):
    def __init__(self, output, batches):
        self.output = Path(output)
        self.batches = batches
        self.rows = []
        self.current = {}

    def on_fit_start(self, trainer, pl_module):
        raw = getattr(pl_module, "_orig_mod", pl_module)
        self.names = {p: name for name, p in raw.named_parameters()}
        self.initial = {
            "state": digest(raw.state_dict()), "rng": rng_state(),
            "optimizer_groups": [{k: v for k, v in group.items() if k != "params"}
                                 for group in trainer.optimizers[0].param_groups],
            "network": type(raw.network).__name__,
            "classes": raw.num_classes, "queries": raw.network.num_q,
        }
        torch.cuda.reset_peak_memory_stats()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.current = {"input": digest(batch), "rng_before": rng_state()}

    def on_before_backward(self, trainer, pl_module, loss):
        self.current["loss"] = float(loss.detach())
        if not math.isfinite(self.current["loss"]):
            raise FloatingPointError("Nonfinite ACDC probe loss")

    def on_after_backward(self, trainer, pl_module):
        raw = getattr(pl_module, "_orig_mod", pl_module)
        self.current["scaled_gradient"] = {name: digest(p.grad) for name, p in raw.named_parameters() if p.grad is not None}

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        raw = getattr(pl_module, "_orig_mod", pl_module)
        self.current["unscaled_gradient"] = {name: digest(p.grad) for name, p in raw.named_parameters() if p.grad is not None}

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        raw = getattr(pl_module, "_orig_mod", pl_module)
        optimizer = trainer.optimizers[0]
        states = {self.names[p]: state for p, state in optimizer.state.items()}
        self.current.update(state=digest(raw.state_dict()), optimizer=digest(states), rng_after=rng_state())
        self.rows.append(self.current)
        if len(self.rows) >= self.batches:
            trainer.should_stop = True

    def on_fit_end(self, trainer, pl_module):
        optimizer = trainer.optimizers[0]
        updates = max((int(state.get("step", 0)) for state in optimizer.state.values()), default=0)
        if len(self.rows) != self.batches or updates < 2:
            raise RuntimeError("Probe must cover native batches and at least two real optimizer updates")
        write_json(self.output, {
            "initial": self.initial, "rows": self.rows,
            "successful_optimizer_updates": updates,
            "peak_allocated": torch.cuda.max_memory_allocated(),
            "peak_reserved": torch.cuda.max_memory_reserved(),
            "evaluation_cases": len(pl_module.eval_case_results),
        })


def operator_checks(output):
    from training.deterministic_point_sampling import sample_point
    from training.deterministic_mask_loss import DeterministicMaskClassificationLoss
    from datasets.acdc_semantic import ACDCSemantic
    previous = configure_strict()
    try:
        torch.manual_seed(0)
        x = torch.randn(16, 6, 5, device="cuda")
        target = torch.randint(0, 5, (16, 6), device="cuda")
        weight = torch.tensor([1., 1., 1., 1., .1], device="cuda")
        image = torch.randn(2, 3, 7, 9, device="cuda")
        points = torch.rand(2, 64, 2, device="cuda")
        references = {}
        for name in ("ce", "point"):
            for index in range(30):
                leaf = (x if name == "ce" else image).clone().requires_grad_()
                result = (torch.nn.functional.cross_entropy(leaf.reshape(-1, 5), target.reshape(-1), weight)
                          if name == "ce" else sample_point(leaf, points, align_corners=False).square().mean())
                gradient, = torch.autograd.grad(result, leaf)
                row = [digest(result), digest(gradient)]
                if index == 0:
                    references[name] = row
                elif row != references[name]:
                    raise AssertionError("ACDC GPU operator replay differs: " + name)
        random.seed(0); np.random.seed(0); torch.manual_seed(0)
        data = ACDCSemantic(base_dir=str(ROOT / "ACDC"), list_dir=str(ROOT / "ACDC/lists_ACDC"),
                            img_size=(224, 224), batch_size=16, num_workers=0)
        data.setup("fit")
        _, targets = data.train_collate([data.train_dataset[i] for i in range(16)])
        targets = [{key: value.cuda() if torch.is_tensor(value) else value
                    for key, value in target.items()} for target in targets]
        criterion = DeterministicMaskClassificationLoss(
            num_labels=4, num_points=12544, oversample_ratio=3.0,
            importance_sample_ratio=.75, mask_coefficient=5., dice_coefficient=5.,
            class_coefficient=2., no_object_coefficient=.1).cuda()
        masks = torch.randn(16, 6, 224, 224, device="cuda")
        logits = torch.randn(16, 6, 5, device="cuda")
        reference = None
        for index in range(3):
            torch.manual_seed(11)
            m, c = masks.clone().requires_grad_(), logits.clone().requires_grad_()
            losses = criterion(masks_queries_logits=m, class_queries_logits=c, targets=targets)
            loss = criterion.loss_total(losses, lambda *args, **kwargs: None)
            gradients = torch.autograd.grad(loss, (m, c))
            row = digest({"losses": losses, "gradients": gradients})
            if reference is None: reference = row
            elif row != reference: raise AssertionError("ACDC full criterion replay differs")
        write_json(output, {"passed": True, "ce_repeats": 30, "point_repeats": 30,
                            "criterion_repeats": 3, "semantic_classes": 4})
    finally:
        restore_backend(previous)


def run_child(command, log, env):
    with Path(log).open("w") as stream:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError("Validation command failed; see " + str(log))


def validate(output, batches):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONHASHSEED="0", OMP_NUM_THREADS="1")
    env.pop("CUDA_VISIBLE_DEVICES", None)
    cpu_code = """import unittest,sys
suite=unittest.TestSuite()
for pattern in ['test_acdc_baseline.py','test_acdc_priors.py','test_acdc_adaptation_fixes.py']:
 suite.addTests(unittest.TestLoader().discover('tests',pattern=pattern))
result=unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if result.wasSuccessful() else 1)
"""
    run_child([sys.executable, "-c", cpu_code], output / "cpu.log", dict(env, CUDA_VISIBLE_DEVICES=""))
    run_child([sys.executable, __file__, "--operators", str(output / "operators.json")],
              output / "operators.log", env)
    all_gates = {}
    for profile in ("baseline", "asee", "ccps"):
        traces = []
        cache = output / "cache" / profile
        cache_counts = []
        for repeat in (1, 2):
            free = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free",
                    "--format=csv,noheader,nounits"], text=True)
            if int(free.splitlines()[0]) < 22000:
                raise RuntimeError("Need 22000 MiB free for the unchanged ACDC GPU probe")
            run = output / profile / str(repeat)
            run.mkdir(parents=True)
            cfg = yaml.safe_load((ROOT / "configs/acdc/deterministic_v2" / (profile + ".yaml")).read_text())
            cfg["trainer"]["enable_checkpointing"] = False
            cfg["trainer"]["callbacks"] = [c for c in cfg["trainer"]["callbacks"]
                                            if not c["class_path"].endswith("ModelCheckpoint")]
            cfg["trainer"]["callbacks"].append({"class_path": "scripts.validate_acdc_deterministic.Trace",
                                                "init_args": {"output": str(run / "trace.json"), "batches": batches}})
            cfg["trainer"]["enable_progress_bar"] = False
            cfg["trainer"]["logger"]["init_args"].update(save_dir=str(run), name="gpu_probe")
            cfg["trainer"]["default_root_dir"] = str(run)
            cfg["deterministic_probe_batches"] = batches
            cfg["deterministic_output_dir"] = str(run / "protocol")
            path = run / "config.yaml"; path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            child_env = dict(env, TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"),
                             TRITON_CACHE_DIR=str(cache / "triton"))
            cache_counts.append(sum(1 for p in cache.rglob("*") if p.is_file()) if cache.exists() else 0)
            write_json(output / "status.json", {"phase": "running", "profile": profile, "repeat": repeat})
            print("START", profile, repeat, flush=True)
            run_child([sys.executable, "scripts/run_acdc_deterministic_v2.py", "fit", "--config", str(path)],
                      run / "train.log", child_env)
            traces.append(json.loads((run / "trace.json").read_text()))
            print("DONE", profile, repeat, flush=True)
        checks = {
            "initial": traces[0]["initial"] == traces[1]["initial"],
            "trajectory": traces[0]["rows"] == traces[1]["rows"],
            "complete_evaluation": all(t["evaluation_cases"] == 40 for t in traces),
            "cold_warm_cache": cache_counts[0] == 0 and cache_counts[1] > 0,
        }
        write_json(output / profile / "comparison.json", {"checks": checks, "cache_file_counts": cache_counts})
        if not all(checks.values()):
            raise AssertionError("ACDC GPU repeat mismatch: " + profile + " " + repr(checks))
        all_gates[profile] = {"G" + str(i): "passed" for i in range(6)}
    readiness = {"dataset": "ACDC", "protocol": PROTOCOL_ID, "passed": True,
                 "validated_profiles": list(all_gates), "profile_gates": all_gates,
                 "sources": source_hashes(ROOT), "validation_output": str(output),
                 "scope": "CPU/operator/native-batch GPU short traces and cold/warm cache; full 400-epoch repeats pending"}
    write_json(ROOT / "outputs/determinism" / PROTOCOL_ID / "readiness.json", readiness)
    write_json(output / "status.json", {"phase": "passed", "readiness": str(ROOT / "outputs/determinism" / PROTOCOL_ID / "readiness.json")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir")
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--operators")
    args = parser.parse_args()
    if args.operators:
        operator_checks(args.operators)
    elif args.output_dir and 2 <= args.batches <= 100:
        validate(args.output_dir, args.batches)
    else:
        parser.error("Use --output-dir with 2..100 batches, or --operators")
