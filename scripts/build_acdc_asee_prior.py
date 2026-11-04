"""Build ACDC foreground-only ASEE priors from raw TRAIN H5 labels, CPU/offline."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from datasets.acdc_splits import PATTERNS, read_manifest, validate_splits
from models.asee import validate_prior
from scripts.build_synapse_asee_prior import conditional_from_counts, encode_text, sha256

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENCODER = "pretrained/Bio_ClinicalBERT/d5892b39a4adaed74b92212a44081509db72f87b"


def resolve(path):
    path=Path(path).expanduser()
    path=(path if path.is_absolute() else ROOT/path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("ACDC prior inputs/outputs must belong to this project")
    return path


def load_label_mapping(path):
    value=json.loads(Path(path).read_text())
    if (value.get("dataset")!="ACDC" or value.get("verified") is not True
            or value.get("background_label")!=0 or not value.get("source","").strip()
            or [r.get("label_id") for r in value.get("classes",[])]!=[1,2,3]):
        raise ValueError("Expected verified ACDC foreground IDs 1,2,3; background excluded")
    if any(not row.get("name","").strip() or not row.get("description","").strip()
           for row in value["classes"]):
        raise ValueError("Every foreground label needs a name and description")
    return value


def count_training_slices(train_dir,names):
    train_dir=Path(train_dir).resolve()
    if not names or len(names)!=len(set(names)):
        raise ValueError("Training filenames must be nonempty and unique")
    counts=np.zeros(3,dtype=np.int64)
    pairs=np.zeros((3,3),dtype=np.int64)
    files={}
    for name in names:
        match=PATTERNS["train"].fullmatch(name)
        if match is None or not 21<=int(match[1])<=100:
            raise ValueError("Only fixed TRAIN patient021..100 slices are allowed")
        path=train_dir/name
        if path.is_symlink() or path.resolve().parent!=train_dir:
            raise ValueError("Training path escapes data directory")
        before=sha256(path)
        with h5py.File(path,"r") as sample:
            label=sample["label"][:]
        if (label.ndim!=2 or label.size==0 or not np.isfinite(label).all()
                or not np.isin(label,[0,1,2,3]).all()):
            raise ValueError(f"Expected raw ACDC labels 0..3: {path}")
        foreground=np.unique(label)
        foreground=foreground[foreground>0].astype(int)-1
        counts[foreground]+=1
        pairs[np.ix_(foreground,foreground)]+=1
        if sha256(path)!=before:raise RuntimeError("Training file changed: "+str(path))
        files[name]=before
    return counts,pairs,files


def build_prior(args):
    output=resolve(args.output)
    if output.exists():raise FileExistsError("Prior already exists: "+str(output))
    lists=resolve(args.list_dir)
    data=resolve(args.data_dir)
    label_map=resolve(args.label_map_json)
    mapping=load_label_mapping(label_map)
    mapping_hash=sha256(label_map)
    # This only checks held-out filenames and patient identities, not their labels.
    split=validate_splits(data,lists,stage="test")
    if split["counts"]!={"train":1512,"val":40,"test":40}:
        raise ValueError("Expected ACDC 80 training / 20 shared validation-test patients")
    names=read_manifest(lists,"train")
    counts,pairs,inputs=count_training_slices(data/"ACDC_training_slices",names)
    matrix=conditional_from_counts(counts,pairs,len(names),eps=1e-5)
    encoder=resolve(args.encoder_path)
    vendor_manifest=json.loads((encoder/"source_manifest.json").read_text())
    for name,info in vendor_manifest["files"].items():
        if sha256(encoder/name)!=info["sha256"]:
            raise RuntimeError("Pretrained encoder file mismatch: "+name)
    text,encoder_meta=encode_text([row["description"] for row in mapping["classes"]],encoder)
    if tuple(text.shape)!=(3,768):raise ValueError("Expected ACDC BioClinicalBERT CLS [3,768]")
    encoder_meta.update(identity_verification="Original official-HF file manifest SHA256 matched",
                        revision=vendor_manifest["revision"])
    if sha256(label_map)!=mapping_hash:raise RuntimeError("Label mapping changed")
    if any(sha256(lists/(s+".txt"))!=h for s,h in split["sha256"].items()):
        raise RuntimeError("Patient split changed")
    if any(sha256(data/"ACDC_training_slices"/n)!=h for n,h in inputs.items()):
        raise RuntimeError("Training H5 changed before publication")
    prior={"schema_version":1,"dataset":"acdc","background_label":0,"class_ids":[1,2,3],
           "class_names":[r["name"] for r in mapping["classes"]],
           "descriptions":[r["description"] for r in mapping["classes"]],
           "label_mapping_verified":True,"label_mapping_source":mapping["source"],
           "label_mapping_verification_scope":mapping["verification_scope"],
           "label_mapping_sha256":mapping_hash,"geometry_verified":False,
           "created_at_utc":datetime.now(timezone.utc).isoformat(),
           "text_features":text,"conditional_matrix":matrix,"text_encoder":encoder_meta,
           "statistics":{"unit":"train_h5_slice","eps":1e-5,"num_samples":len(names),
                         "train_list_sha256":split["sha256"]["train"],"val_list_sha256":split["sha256"]["val"],
                         "test_list_sha256":split["sha256"]["test"],
                         "train_cases":[f"patient{i:03d}" for i in split["patients"]["train"]],
                         "class_counts":counts.tolist(),"cooccurrence_counts":pairs.tolist(),
                         "train_h5_sha256":inputs,"validation_test_labels_used":False}}
    validate_prior(prior)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("xb") as handle:torch.save(prior,handle)
    summary={"prior":str(output),"sha256":sha256(output),"foreground_ids":[1,2,3],
             "text_shape":list(text.shape),"conditional_shape":list(matrix.shape),
             "training_slices":len(names),"training_patients":len(split["patients"]["train"]),"class_counts":counts.tolist(),
             "cooccurrence_counts":pairs.tolist(),"background_in_prior":False,
             "validation_test_labels_used":False,"device":"cpu"}
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir",default="ACDC")
    parser.add_argument("--list-dir",default="ACDC/lists_ACDC")
    parser.add_argument("--label-map-json",default="priors/acdc_label_map.verified.json")
    parser.add_argument("--encoder-path",default=DEFAULT_ENCODER)
    parser.add_argument("--output",default="priors/acdc_asee_80_20_v1.pt")
    args=parser.parse_args()
    torch.set_num_threads(2)
    build_prior(args)


if __name__=="__main__":
    main()
