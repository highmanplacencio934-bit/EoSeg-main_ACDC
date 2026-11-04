"""Fixed 80/20 ACDC patients, with validation intentionally sharing test cases."""
import hashlib
from collections import Counter
import re
from pathlib import Path

PATIENT_RANGES = {"train": (21, 100), "val": (1, 20), "test": (1, 20)}
PATTERNS = {
    "train": re.compile(r"patient(\d{3})_frame(\d+)_slice_(\d+)\.h5"),
    "val": re.compile(r"patient(\d{3})_frame(\d+)\.h5"),
    "test": re.compile(r"patient(\d{3})_frame(\d+)\.h5"),
}


def read_manifest(list_dir, split):
    if split not in PATIENT_RANGES:
        raise ValueError(f"Unsupported ACDC split: {split}")
    path = Path(list_dir) / f"{split}.txt"
    names = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not names or len(names) != len(set(names)):
        raise ValueError(f"Empty or duplicate ACDC manifest: {path}")
    low, high = PATIENT_RANGES[split]
    for name in names:
        match = PATTERNS[split].fullmatch(name)
        if match is None or not low <= int(match[1]) <= high:
            raise ValueError(f"Wrong patient or unsafe filename in {split}: {name}")
    return names


def validate_splits(base_dir, list_dir, stage=None):
    base_dir, list_dir = Path(base_dir), Path(list_dir)
    manifests = {split: read_manifest(list_dir, split) for split in PATIENT_RANGES}
    patients = {}
    for split, names in manifests.items():
        patients[split] = {int(PATTERNS[split].fullmatch(name)[1]) for name in names}
        low, high = PATIENT_RANGES[split]
        if patients[split] != set(range(low, high + 1)):
            raise ValueError(f"Missing patients in ACDC {split} manifest")
    expected_counts = {"train": 1512, "val": 40, "test": 40}
    if {name: len(items) for name, items in manifests.items()} != expected_counts:
        raise ValueError("ACDC 80/20 requires 1512 training slices and 40 evaluation volumes")
    for split in ("val", "test"):
        counts = Counter(int(PATTERNS[split].fullmatch(name)[1]) for name in manifests[split])
        if any(count != 2 for count in counts.values()):
            raise ValueError("Every ACDC evaluation patient must have exactly two frames")
    training_frames = {}
    for name in manifests["train"]:
        match = PATTERNS["train"].fullmatch(name)
        patient, frame, index = map(int, match.groups())
        training_frames.setdefault((patient, frame), []).append(index)
    frame_counts = Counter(patient for patient, frame in training_frames)
    if any(count != 2 for count in frame_counts.values()):
        raise ValueError("Every ACDC training patient must have exactly two frames")
    if any(sorted(indices) != list(range(max(indices) + 1)) for indices in training_frames.values()):
        raise ValueError("ACDC training slices must cover each frame without gaps")
    if patients["train"] & patients["test"]:
        raise ValueError("ACDC patient leakage between training and evaluation")
    if manifests["val"] != manifests["test"]:
        raise ValueError("ACDC val.txt must exactly mirror test.txt for shared validation/testing")
    required = ("test",) if stage in ("test", "predict") else ("train", "test")
    for split in required:
        folder = "ACDC_training_slices" if split == "train" else "ACDC_training_volumes"
        for name in manifests[split]:
            if not (base_dir / folder / name).is_file():
                raise FileNotFoundError(base_dir / folder / name)
    return {
        "policy": "user_confirmed_80_train_20_shared_validation_test",
        "counts": {s: len(names) for s, names in manifests.items()},
        "patient_counts": {s: len(ids) for s, ids in patients.items()},
        "patients": {s: sorted(ids) for s, ids in patients.items()},
        "sha256": {s: hashlib.sha256((list_dir / f"{s}.txt").read_bytes()).hexdigest() for s in manifests},
        "fit_evaluation_split": "test",
        "independent_val_split_used_for_selection": False,
        "validation_alias_of": "test",
        "test_used_for_selection": True,
    }
