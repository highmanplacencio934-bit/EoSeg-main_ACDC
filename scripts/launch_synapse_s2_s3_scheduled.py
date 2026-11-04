"""Isolated S2/S3 scheduled supervisor: fixed predecessor and memory-based admission.

Original training variables remain fixed; one explicit component flag changes.
"""
from __future__ import annotations
import argparse
import copy
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "main.py").is_file() and (parent / "AGENTS.md").is_file())
sys.path.insert(0, str(ROOT))
from scripts import launch_synapse_journal_ablation as shared
BASE, PYTHON, ENV = shared.BASE, shared.PYTHON, shared.ENV
read, write, sha, now = shared.read, shared.write, shared.sha, shared.now
voxel_mode, validate_voxel_value = shared.voxel_mode, shared.validate_voxel_value
validate_metric_row, format_voxel = shared.validate_metric_row, shared.format_voxel
validate_voxel_row_mode = shared.validate_voxel_row_mode
identity, wait_identity, alive = shared.identity, shared.wait_identity, shared.alive
environment, set_plan_status = shared.environment, shared.set_plan_status

ASEE = "models.eoseg_asee.EoSegMultiQSegFusionASEE"
CCPS = "models.eoseg_asee_ccps.EoSegMultiQSegFusionASEECCPS"
PROFILES = {
    "S1": {"config": "s1_asee_text_only.yaml", "flags": {"use_text": True, "use_probability": False}},
    "S2": {"config": "s2_asee_probability_only.yaml", "flags": {"use_text": False, "use_probability": True}},
    "S3": {"config": "s3_asee_patch_only.yaml", "flags": {"enhance_patch": True, "enhance_query": False}},
    "S4": {"config": "s4_asee_query_only.yaml", "flags": {"enhance_patch": False, "enhance_query": True}},
    "A1": {"config": "a1_global.yaml", "flags": {"selector_mode": "global", "selector_on_patch": True, "selector_on_query": False, "patch_gate_effect": "both"}},
    "A2": {"config": "a2_visual.yaml", "flags": {"selector_mode": "visual", "selector_on_patch": True, "selector_on_query": False, "patch_gate_effect": "both"}},
    "A10": {"config": "a10_context_only.yaml", "flags": {"selector_mode": "visual_prior", "selector_on_patch": True, "selector_on_query": False, "patch_gate_effect": "context_only"}},
    "A11": {"config": "a11_residual_only.yaml", "flags": {"selector_mode": "visual_prior", "selector_on_patch": True, "selector_on_query": False, "patch_gate_effect": "residual_only"}},
}
for _key, _profile in PROFILES.items():
    _profile.update(network=CCPS if _key.startswith("A") else ASEE, ccps=_key.startswith("A"), asee=True)
    _profile["configuration"] = "configs/synapse/ablations_journal/" + _profile["config"]


def checked_run(run):
    run = Path(run).resolve()
    if not run.is_relative_to(BASE.resolve()) or run.parent.name not in PROFILES:
        raise ValueError("Component run outside allowed roots")
    return run


# Same-model training processes measured around20GiB; budget24GiB including headroom.
# Same-best8-view measured around3.2GiB; budget6GiB including temporary allocations.
TRAIN_FREE_MIB = 24576
TEST_FREE_MIB = 6144
BASELINE_2_RUN = ROOT / "outputs/baseline-2/20261026T130228Z"
ALLOWED_EXPERIMENTS = {"S2", "S3"}


def gpu_free(required_mib=TEST_FREE_MIB):
    """Other users may run; only available memory determines resource admission."""
    raw = subprocess.check_output(
        ["nvidia-smi", "--id=0", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        text=True).strip()
    return int(raw) >= required_mib


def require_baseline_2_complete(run=None):
    """Refuse to start a successor before the fixed current baseline fully finishes."""
    run = Path(run) if run is not None else BASELINE_2_RUN
    status = read(run / "status.json")
    if status.get("state") != "completed" or status.get("roi_enabled") is not True:
        raise RuntimeError("ROI baseline-2 has not fully completed; keep its training running")
    if read(run / "training_exit.json").get("exit_code") != 0:
        raise RuntimeError("baseline-2 did not exit0; no automatic retraining")
    summary = read(run / "summary.json")
    if summary.get("max_epochs") != 220 or summary.get("test_epoch_records") != 50 or summary.get("test_case_records") != 600 or summary.get("case_organ_records") != 4800:
        raise RuntimeError("baseline-2 completion summary is incomplete")
    folders = list((run / "synapse_test_selection").glob("*"))
    if len(folders) != 1:
        raise RuntimeError("baseline-2 must have one selection directory")
    folder = folders[0]
    selection = read(folder / "selection_summary.json")
    if selection.get("max_epochs") != 220 or selection.get("last_evaluated_epoch") != 220:
        raise RuntimeError("baseline-2 final selection incomplete")
    expected = set((ROOT / "project_TransUNet/TransUNet/lists/lists_Synapse/test_vol.txt").read_text().splitlines())
    with (folder / "test_epoch_metrics.csv").open(newline="") as stream:
        epochs = list(csv.DictReader(stream))
    with (folder / "test_case_metrics.csv").open(newline="") as stream:
        cases = list(csv.DictReader(stream))
    with (folder / "physical_case_organ_metrics.csv").open(newline="") as stream:
        organs = list(csv.DictReader(stream))
    window = set(range(171, 221))
    if len(epochs) != 50 or {int(row["epoch"]) for row in epochs} != window or len(cases) != 600 or len(organs) != 4800:
        raise RuntimeError("baseline-2 raw CSV coverage incomplete")
    for epoch in window:
        case_rows = [row for row in cases if int(row["epoch"]) == epoch]
        organ_rows = [row for row in organs if int(row["epoch"]) == epoch]
        if len(case_rows) != 12 or {row["case"] for row in case_rows} != expected:
            raise RuntimeError("baseline-2 missing/duplicate test case")
        if len(organ_rows) != 96 or {(row["case"], int(row["label_id"])) for row in organ_rows} != {(case, label) for case in expected for label in range(1, 9)}:
            raise RuntimeError("baseline-2 missing/duplicate organ case")
    checkpoint = Path(summary["checkpoint"]).resolve()
    if not checkpoint.is_relative_to(run.resolve()) or sha(checkpoint) != summary["checkpoint_sha256"]:
        raise RuntimeError("baseline-2 best checkpoint provenance mismatch")
    import torch
    last = torch.load(folder / "last.ckpt", map_location="cpu", weights_only=False, mmap=True)
    if last["epoch"] + 1 != 220 or last["global_step"] != 30360:
        raise RuntimeError("baseline-2 did not finish all training steps")
    return {"run": str(run), "summary_sha256": sha(run / "summary.json"),
            "best_checkpoint_sha256": summary["checkpoint_sha256"], "max_epochs": 220}



def check_no_checkpoint(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "ckpt_path" and item is not None:
                raise RuntimeError("Component fit must not load a trained checkpoint")
            check_no_checkpoint(item)
    elif isinstance(value, list):
        for item in value:
            check_no_checkpoint(item)


def preflight(experiment, require_gpu=True):
    if experiment not in ALLOWED_EXPERIMENTS:
        raise RuntimeError("This scheduled entry only authorizes S2/S3")
    baseline_completion = require_baseline_2_complete()
    import yaml
    profile = PROFILES[experiment]
    state = read(BASE / "orchestration_state.json")
    registry = read(BASE / "experiment_registry.json")
    if any(next(x for x in registry["queue"] if x["id"] == key)["status"] != "completed" for key in ("M0", "M1")):
        raise RuntimeError("M0 and M1 must complete first")
    if state["stop"]["status"] != "stopped_by_user":
        raise RuntimeError("Original augmentation run not confirmed stopped")
    ready_path = BASE / "optimization/readiness.json"
    ready = read(ready_path)
    if ready.get("status") != "passed" or ready.get("passed") is not True:
        raise RuntimeError("ROI readiness not passed")
    for name, expected in {**ready["hashes"], **ready["validation_report_hashes"]}.items():
        if sha(ROOT / name) != expected:
            raise RuntimeError("ROI evidence/source changed: " + name)
    preparation_path = BASE / "preparation/component_configs.json"
    preparation = read(preparation_path)
    rows = [x for x in preparation["configurations"] if x["id"] == experiment]
    if len(rows) != 1:
        raise RuntimeError("Component configuration is not prepared")
    record = rows[0]
    if record.get("cpu_parse_status") != "passed" or record.get("print_config_exit_code") != 0 or record.get("data_dictionary_identical") is not True:
        raise RuntimeError("Component CPU configuration preparation not passed")
    config_path = ROOT / profile["configuration"]
    if record["configuration"] != profile["configuration"] or sha(config_path) != record["sha256"]:
        raise RuntimeError("Component configuration/hash mismatch")
    source_path = ROOT / record["source_configuration"]
    if sha(source_path) != record["source_sha256"]:
        raise RuntimeError("Original comparison configuration changed")
    config, original = yaml.safe_load(config_path.read_text()), yaml.safe_load(source_path.read_text())
    check_no_checkpoint(config)
    if config.get("seed_everything") != 0 or config.get("compile_disabled", False):
        raise RuntimeError("Seed/view/compile training contract changed")
    trainer = config["trainer"]
    if trainer["max_epochs"] != 220 or trainer["precision"] != "16-mixed":
        raise RuntimeError("Training budget/precision changed")
    ignore = {"default_root_dir", "logger", "callbacks"}
    if {k: v for k, v in trainer.items() if k not in ignore} != {k: v for k, v in original["trainer"].items() if k not in ignore}:
        raise RuntimeError("Unexpected scientific trainer override")
    if config["data"] != original["data"] or config["data"]["class_path"] != "datasets.synapse_semantic.SynapseSemantic":
        raise RuntimeError("Original augmentation/data configuration changed")
    for key, expected in {"batch_size": 16, "num_workers": 8, "img_size": [384, 384], "num_classes": 9}.items():
        if config["data"]["init_args"].get(key) != expected:
            raise RuntimeError("Wrong data setting: " + key)
    model = config["model"]
    expected_wrapper = "training.mask_classification_ccps.MaskClassificationSemanticCCPS" if profile["ccps"] else "training.synapse_physical_semantic.SynapsePhysicalHD95Semantic"
    if model["class_path"] != expected_wrapper:
        raise RuntimeError("Wrong explicit physical training wrapper")
    args, source_args = model["init_args"], original["model"]["init_args"]
    for key, expected in {"num_classes": 9, "img_size": [384, 384]}.items():
        if args.get(key) != expected:
            raise RuntimeError("Wrong model dimensions")
    if float(args["lr"]) != 2e-4 or args.get("presence_loss_weight", 0.0) != 0.0:
        raise RuntimeError("Unexpected LR/presence supervision")
    excluded = {"network", "spacing_manifest"}
    if {k: v for k, v in args.items() if k not in excluded} != {k: v for k, v in source_args.items() if k not in excluded}:
        raise RuntimeError("Unexpected model/loss/optimizer setting")
    expected_network = copy.deepcopy(source_args["network"])
    expected_network["init_args"].update(record["requested_component_flags"])
    if args["network"] != expected_network or args["network"]["class_path"] != profile["network"]:
        raise RuntimeError("Network differs beyond declared component")
    for key, expected in profile["flags"].items():
        if args["network"]["init_args"].get(key) != expected:
            raise RuntimeError("Incorrect study flag: " + key)
    parsed = record["parsed_scientific_values"]
    for key, expected in {"lr": 2e-4, "llrd": 0.8, "weight_decay": 0.05, "num_points": 12544, "max_epochs": 220,
                          "precision": "16-mixed", "gradient_clip_val": 0.01, "patch_size": 16,
                          "scale_range": [0.5, 2.0], "color_jitter_enabled": True, "rotation_present": False}.items():
        if parsed.get(key) != expected:
            raise RuntimeError("Resolved scientific setting mismatch: " + key)
    prior = args["network"]["init_args"]["prior_path"]
    if prior != "priors/synapse_asee_v1.pt" or sha(ROOT / prior) != ready["hashes"][prior]:
        raise RuntimeError("Verified training-only prior changed")
    hardware = None
    if require_gpu:
        if not gpu_free(TRAIN_FREE_MIB):
            raise RuntimeError("Insufficient free GPU memory for training; never stop other users")
        probe = "import json,sys,torch; assert torch.cuda.is_available(); p=torch.cuda.get_device_properties(0); print(json.dumps({'python':sys.executable,'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':p.name,'total_memory_bytes':p.total_memory}))"
        hardware = json.loads(subprocess.check_output([PYTHON, "-c", probe], env=environment(), cwd=ROOT, text=True))
        if "5090" not in hardware["gpu"]:
            raise RuntimeError("Unexpected GPU")
    if shutil.disk_usage(ROOT).free < 15 * 1024**3:
        raise RuntimeError("Insufficient component-run disk")
    return {"configuration": profile["configuration"], "configuration_sha256": sha(config_path),
            "component_record": record, "preparation_sha256": sha(preparation_path),
            "readiness_sha256": sha(ready_path), "prior_sha256": sha(ROOT / prior),
            "launcher_sha256": sha(Path(__file__)), "shared_launcher_sha256": sha(Path(shared.__file__)), "network_class": profile["network"],
            "model_class": model["class_path"], "seed": 0, "max_epochs": 220,
            "initialization": "original offline timm pretrained flow; no trained segmentation checkpoint",
            "disk_free_bytes": shutil.disk_usage(ROOT).free, "environment": ENV,
            "hardware": hardware, "hostname": subprocess.check_output(["hostname"], text=True).strip(),
            "baseline_2_completion": baseline_completion}


def fit_command(experiment, run, proof):
    callback = [{"class_path": "training.synapse_physical_selection.SynapsePhysicalHD95Selection",
                 "init_args": {"last_n_epochs": 50, "expected_test_cases": 12, "output_dir": str(run / "training")}}]
    return [PYTHON, "-u", "scripts/run_synapse_roi.py", "fit", "--config", proof["configuration"],
            "--seed_everything=0", "--trainer.accelerator=gpu", "--trainer.devices=1",
            "--trainer.default_root_dir=" + str(run / "training"), "--trainer.callbacks=" + json.dumps(callback),
            "--trainer.logger.init_args.save_dir=" + str(run / "wandb"),
            "--trainer.logger.init_args.name=journal_" + experiment.lower() + "_seed0_" + run.name]


def validate_training(experiment, run):
    import torch
    profile = PROFILES[experiment]
    training = read(run / "training_exit.json")
    if training["exit_code"] != 0:
        raise RuntimeError("Training did not exit successfully")
    folders = list((run / "training" / "synapse_test_selection").glob("*"))
    if len(folders) != 1:
        raise RuntimeError("Expected one training selection output")
    folder = folders[0]
    voxel_enabled = voxel_mode(folder)
    selection = read(folder / "selection_summary.json")
    if selection["max_epochs"] != 220 or selection["first_evaluation_epoch"] != 171 or selection["last_evaluated_epoch"] != 220:
        raise RuntimeError("Incomplete 220/final50 training budget")
    with (folder / "test_epoch_metrics.csv").open(newline="", encoding="utf-8") as stream:
        epochs = list(csv.DictReader(stream))
    with (folder / "test_case_metrics.csv").open(newline="", encoding="utf-8") as stream:
        cases = list(csv.DictReader(stream))
    with (folder / "physical_epoch_organ_metrics.csv").open(newline="", encoding="utf-8") as stream:
        organs = list(csv.DictReader(stream))
    with (folder / "physical_case_organ_metrics.csv").open(newline="", encoding="utf-8") as stream:
        case_organs = list(csv.DictReader(stream))
    expected = set((ROOT / "project_TransUNet/TransUNet/lists/lists_Synapse/test_vol.txt").read_text().splitlines())
    if [int(x["epoch"]) for x in epochs] != list(range(171, 221)):
        raise RuntimeError("Missing or duplicate candidate epochs")
    if len(cases) != 600 or len(organs) != 400 or len(case_organs) != 4800:
        raise RuntimeError("Incomplete final50 CSV records")
    for epoch in range(171, 221):
        cr = [x for x in cases if int(x["epoch"]) == epoch]
        oc = [x for x in organs if int(x["epoch"]) == epoch]
        co = [x for x in case_organs if int(x["epoch"]) == epoch]
        if len(cr) != 12 or {x["case"] for x in cr} != expected:
            raise RuntimeError("Incomplete/duplicate epoch cases")
        if len(oc) != 8 or {int(x["label_id"]) for x in oc} != set(range(1, 9)):
            raise RuntimeError("Incomplete/duplicate epoch organs")
        if len(co) != 96 or {(x["case"], int(x["label_id"])) for x in co} != {(c, i) for c in expected for i in range(1, 9)}:
            raise RuntimeError("Incomplete case/organ epoch pairs")
    for row in cases + organs + case_organs:
        validate_voxel_row_mode(row, voxel_enabled)
        for key, value in row.items():
            if key == "hd95_voxel":
                validate_voxel_value(value, voxel_enabled)
                continue
            if key in {"mean_dice", "mean_hd95", "dice", "hd95_mm"} or key.startswith("dice_class_") or key.startswith("hd95_class_"):
                number = float(value)
                if not math.isfinite(number) or number < 0 or ((key == "dice" or key == "mean_dice" or key.startswith("dice_class_")) and number > 1):
                    raise RuntimeError("Nonfinite/out-of-range training selection metrics")
    best = folder / "best_test.ckpt"
    last = folder / "last.ckpt"
    if not best.is_file() or not last.is_file():
        raise RuntimeError("Missing original best/last weights")
    final = torch.load(last, map_location="cpu", mmap=True, weights_only=False)
    if final["epoch"] != 219 or final["global_step"] != 30360:
        raise RuntimeError("Last checkpoint did not complete 220 full epochs")
    del final
    ckpt = torch.load(best, map_location="cpu", mmap=True, weights_only=False)
    hp = ckpt["hyper_parameters"]
    if hp.get("compute_hd95_voxel", True) != voxel_enabled:
        raise RuntimeError("Checkpoint HD95 voxel mode differs from selection metadata")
    if hp["network"]["class_path"] != profile["network"] or ckpt["epoch"] + 1 != selection["best_epoch"]:
        raise RuntimeError("Selected checkpoint model/epoch mismatch")
    saved_init = hp["network"]["init_args"]
    for flag, value in profile["flags"].items():
        if saved_init.get(flag) != value:
            raise RuntimeError("Saved checkpoint study flag mismatch: " + flag)
    if hp.get("_class_path") != ("training.mask_classification_ccps.MaskClassificationSemanticCCPS" if profile["ccps"] else "training.synapse_physical_semantic.SynapsePhysicalHD95Semantic"):
        raise RuntimeError("Saved training wrapper mismatch")
    first_best = max(epochs, key=lambda x: float(x["mean_dice"]))
    if int(first_best["epoch"]) != selection["best_epoch"] or float(first_best["mean_dice"]) != selection["best_mean_dice"]:
        raise RuntimeError("Checkpoint was not chosen by original mean Dice")
    checkpoint = {"path": str(best), "sha256": sha(best), "epoch": ckpt["epoch"] + 1,
                  "global_step": ckpt["global_step"], "network_class": profile["network"],
                  "model_state_entry_count": len(ckpt["state_dict"]),
                  "model_tensor_count": sum(torch.is_tensor(x) for x in ckpt["state_dict"].values())}
    del ckpt
    manifest = read(folder / "run_manifest.json")
    if str(manifest.get("device", "")).split(":")[0] != "cuda":
        raise RuntimeError("Training did not use CUDA")
    if read(folder / "metric_units.json")["hd95_unit"] != "mm":
        raise RuntimeError("Training selection did not preserve physical HD95")
    proof = {"selection_folder": str(folder.relative_to(ROOT)), "selection": selection,
             "checkpoint": checkpoint, "last_checkpoint_sha256": sha(last),
             "manifest_sha256": sha(folder / "run_manifest.json"),
             "candidate_epochs": 50, "case_records": 600, "case_organ_records": 4800,
             "hd95_voxel_enabled": voxel_enabled, "validated_at_utc": now()}
    write(run / "training_completion.json", proof)
    return proof






def save_runtime(registry, state, experiment, run, status):
    entry = next(x for x in registry["queue"] if x["id"] == experiment)
    entry.update(status=status["state"], output_dir=str(run.relative_to(ROOT)))
    if status.get("state") in ("completed", "failed"):
        entry["finished_at_utc"] = status.get("finished_at_utc")
    write(BASE / "experiment_registry.json", registry)
    state.update(phase=status["state"], active_experiment=None if status["state"] in ("completed",) else
                 {"id": experiment, "run_dir": str(run.relative_to(ROOT)), "supervisor_pid": os.getpid(), "status": status["state"]},
                 last_update_utc=now())
    write(BASE / "orchestration_state.json", state)
    write(run / "status.json", status)


def run_child(command, run, filename, status, state, registry, experiment):
    with (run / filename).open("w") as log:
        child = subprocess.Popen(command, cwd=ROOT, env=environment(), stdout=log, stderr=subprocess.STDOUT)
        status.update(child_pid=child.pid, child_identity=wait_identity(child.pid, "scripts/run_synapse_roi.py"),
                      supervisor_identity=identity(os.getpid()))
        save_runtime(registry, state, experiment, run, status)
        return child.wait()


def supervise(run):
    run = checked_run(run)
    with (BASE / "queue.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        launch, status = read(run / "launch.json"), read(run / "status.json")
        experiment = launch["experiment"]
        registry, state = read(BASE / "experiment_registry.json"), read(BASE / "orchestration_state.json")
        entry = next(x for x in registry["queue"] if x["id"] == experiment)
        active = state.get("active_experiment") or {}
        required = "starting"
        receipt = read(BASE / experiment / "component_launch_receipt.json")
        if (status.get("state") != required or entry.get("status") != required or
                active.get("id") != experiment or active.get("run_dir") != str(run.relative_to(ROOT)) or
                active.get("supervisor_pid") != os.getpid() or receipt.get("run_dir") != str(run.relative_to(ROOT))):
            raise RuntimeError("Supervisor intent rejected duplicate/stale run")
        try:
            current = preflight(experiment, require_gpu=True)
            for key in ("configuration_sha256", "prior_sha256", "readiness_sha256", "launcher_sha256", "shared_launcher_sha256", "preparation_sha256"):
                if current[key] != launch["preflight"][key]:
                    raise RuntimeError("Source changed after launch intent: " + key)
            status.update(state="training", stage="fit", training_started_at_utc=now())
            write(run / "fit_launch.json", {"command": launch["fit_command"], "preflight": current, "started_at_utc": now()})
            code = run_child(launch["fit_command"], run, "train.log", status, state, registry, experiment)
            write(run / "training_exit.json", {"exit_code": code, "finished_at_utc": now()})
            status["training_exit_code"] = code
            if code != 0:
                raise RuntimeError("Training exited nonzero; no automatic retraining")
            training = validate_training(experiment, run)
            status["training_completed"] = True
            registry, state = read(BASE / "experiment_registry.json"), read(BASE / "orchestration_state.json")
            entry = next(x for x in registry["queue"] if x["id"] == experiment)
            entry.update(status="completed", completed_at_utc=now(),
                         checkpoint_sha256=training["checkpoint"]["sha256"], summary=str((run / "training_completion.json").relative_to(ROOT)))
            status.update(state="completed", stage="finished", exit_code=0, finished_at_utc=now())
            save_runtime(registry, state, experiment, run, status)
            set_plan_status(experiment, "已完成", str((run / "training_completion.json").relative_to(ROOT)))
            return 0
        except BaseException as error:
            status.update(state="failed", stage="needs_review", exit_code=1,
                          error=type(error).__name__ + ": " + str(error), finished_at_utc=now())
            save_runtime(registry, state, experiment, run, status)
            set_plan_status(experiment, "失败待核查", str((run / "status.json").relative_to(ROOT)))
            return 1


def advance(expected_experiment=None):
    with (BASE / "queue.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps({"state": "busy", "reason": "existing pipeline owns shared lock"}))
            return
        registry, state = read(BASE / "experiment_registry.json"), read(BASE / "orchestration_state.json")
        if any(next(x for x in registry["queue"] if x["id"] == key)["status"] != "completed" for key in ("M0", "M1")):
            print(json.dumps({"state": "not_prepared", "reason": "M0/M1 must finish; components not started"}))
            return
        remaining = [x for x in registry["queue"] if x["status"] != "completed"]
        if not remaining:
            print(json.dumps({"state": "all_completed"}))
            return
        entry = remaining[0]
        experiment = entry["id"]
        if experiment not in ALLOWED_EXPERIMENTS or experiment != expected_experiment:
            print(json.dumps({"state": "waiting_for_ordered_predecessor", "next_experiment": experiment,
                              "requested_experiment": expected_experiment}))
            return
        if experiment not in PROFILES:
            print(json.dumps({"state": "not_prepared", "next_experiment": experiment}))
            return
        if state.get("active_experiment") is not None:
            print(json.dumps({"state": "already_registered", "active": state["active_experiment"]}))
            return
        if entry["status"] != "planned":
            print(json.dumps({"state": "needs_review", "entry": entry}))
            return
        try:
            proof = preflight(experiment, require_gpu=True)
        except (FileNotFoundError, KeyError) as error:
            print(json.dumps({"state": "not_prepared", "next_experiment": experiment, "reason": str(error)}))
            return
        parent = BASE / experiment
        parent.mkdir(exist_ok=True)
        receipt_path = parent / "component_launch_receipt.json"
        if receipt_path.exists():
            print(json.dumps({"state": "already_launched", "receipt": read(receipt_path)}))
            return
        run = parent / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run.mkdir(exist_ok=False)
        launch = {"experiment": experiment, "created_at_utc": now(), "preflight": proof,
                  "fit_command": fit_command(experiment, run, proof), "max_epochs": 220,
                  "seed": 0, "scientific_change": PROFILES[experiment]["flags"],
                  "output_root": str(run)}
        write(run / "launch.json", launch)
        write(run / "status.json", {"state": "starting", "stage": "reserving_fit", "experiment": experiment})
        shutil.copy2(BASE / "optimization/readiness.json", run / "optimization_readiness_snapshot.json")
        shutil.copy2(BASE / "preparation/component_configs.json", run / "component_configuration_snapshot.json")
        write(receipt_path, {"experiment": experiment, "run_dir": str(run.relative_to(ROOT)), "created_at_utc": now()})
        command = [PYTHON, str(Path(__file__).resolve()), "--supervise", str(run)]
        supervisor_log = run / "supervisor.log"
        with supervisor_log.open("w") as log:
            worker = subprocess.Popen(command, cwd=ROOT, env=environment(), stdout=log, stderr=subprocess.STDOUT,
                                      stdin=subprocess.DEVNULL, start_new_session=True)
        owner = wait_identity(worker.pid, "--supervise " + str(run))
        entry.update(status="starting", output_dir=str(run.relative_to(ROOT)))
        state.update(phase="component_running", active_experiment={"id": experiment, "run_dir": str(run.relative_to(ROOT)),
                     "supervisor_pid": worker.pid, "supervisor_identity": owner}, last_update_utc=now())
        write(BASE / "experiment_registry.json", registry)
        write(BASE / "orchestration_state.json", state)
        (parent / "latest_path.txt").write_text(str(run.relative_to(ROOT)) + "\n")
        set_plan_status(experiment, "运行中", str((run / "status.json").relative_to(ROOT)))
        print(json.dumps({"state": "launched", "experiment": experiment,
                          "run_dir": str(run.relative_to(ROOT)), "supervisor_pid": worker.pid}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--next", action="store_true")
    parser.add_argument("--expected-experiment", choices=sorted(ALLOWED_EXPERIMENTS))
    parser.add_argument("--supervise", type=Path)
    args = parser.parse_args()
    if "preparation" in Path(__file__).parts:
        raise SystemExit("Draft is intentionally non-executable; stabilize and test under scripts first")
    if args.supervise:
        raise SystemExit(supervise(args.supervise))
    else:
        if args.expected_experiment is None:
            parser.error("--expected-experiment S2 or S3 is required for a scheduled launch")
        advance(expected_experiment=args.expected_experiment)
