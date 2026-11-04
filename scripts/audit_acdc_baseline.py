"""Read-only ACDC data audit and optional CPU-only real CLI initialization."""
import argparse
import collections
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def file_sha(path):
    value=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(4*1024*1024), b""):
            value.update(block)
    return value.hexdigest()


def audit_data():
    import h5py
    import numpy as np
    from datasets.acdc_splits import validate_splits
    result={"splits":validate_splits(ROOT/"ACDC",ROOT/"ACDC/lists_ACDC","fit"),"files":{},
            "slice_volume_matches":0,"spacing_available":False}
    volumes={}
    for folder,ndim in (("ACDC_training_volumes",3),("ACDC_training_slices",2)):
        summary={"count":0,"keys":collections.Counter(),"image_dtype":collections.Counter(),
                 "label_dtype":collections.Counter(),"labels":set(),"shape_counts":collections.Counter(),
                 "min":float("inf"),"max":float("-inf"),"nonempty_attributes":[]}
        for path in sorted((ROOT/"ACDC"/folder).glob("*.h5")):
            with h5py.File(path,"r") as handle:
                image,label=handle["image"][:],handle["label"][:]
                if image.ndim!=ndim or image.shape!=label.shape or not np.isfinite(image).all():
                    raise ValueError(f"Invalid data: {path}")
                if not np.isin(label,[0,1,2,3]).all():raise ValueError(f"Invalid labels: {path}")
                summary["count"]+=1
                summary["keys"][str(tuple(sorted(handle.keys())))]+=1
                summary["image_dtype"][str(image.dtype)]+=1;summary["label_dtype"][str(label.dtype)]+=1
                summary["labels"].update(map(int,np.unique(label)))
                summary["shape_counts"][str(image.shape)]+=1
                summary["min"]=min(summary["min"],float(image.min()))
                summary["max"]=max(summary["max"],float(image.max()))
                if handle.attrs or any(handle[key].attrs for key in handle):
                    summary["nonempty_attributes"].append(path.name)
                if ndim==3:
                    volumes[path.stem]=(image,label)
                else:
                    key,index=path.stem.rsplit("_slice_",1)
                    vi,vl=volumes[key]
                    if not np.array_equal(image,vi[int(index)]) or not np.array_equal(label,vl[int(index)]):
                        raise ValueError(f"Slice/volume mismatch: {path}")
                    result["slice_volume_matches"]+=1
        summary["labels"]=sorted(summary["labels"])
        result["files"][folder]=summary
    if (result["files"]["ACDC_training_volumes"]["count"],
        result["files"]["ACDC_training_slices"]["count"],result["slice_volume_matches"])!=(200,1902,1902):
        raise ValueError("Unexpected ACDC release counts")
    return result


def initialize_cli():
    # Hide CUDA before importing Torch: this audit never runs a GPU training step.
    os.environ["CUDA_VISIBLE_DEVICES"]=""
    import numpy as np
    import random
    import torch
    from main import cli_main
    captured={}
    def capture(trainer, model, **kwargs):
        data=kwargs["datamodule"]
        data.setup("fit")
        steps_per_epoch=len(data.train_dataloader())
        model._trainer=SimpleNamespace(estimated_stepping_batches=steps_per_epoch*trainer.max_epochs)
        cfg=model.configure_optimizers()
        optimizer=cfg["optimizer"]
        states={name:hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                for name,value in model.state_dict().items() if torch.is_tensor(value)}
        backbone=model.network.encoder.backbone
        captured.update(
            network_class=type(model.network).__module__+"."+type(model.network).__name__,
            model_class=type(model).__module__+"."+type(model).__name__,
            criterion_class=type(model.criterion).__module__+"."+type(model.criterion).__name__,
            backbone_name=model.network.encoder.backbone_name,patch_size=model.network.encoder.patch_size,
            embed_dim=backbone.embed_dim,depth=len(backbone.blocks),grid_size=list(backbone.patch_embed.grid_size),
            num_q=model.network.num_q,num_classes=model.num_classes,img_size=list(model.img_size),
            lr=model.lr,llrd=model.llrd,weight_decay=model.weight_decay,poly_power=model.poly_power,
            warmup_steps=list(model.warmup_steps),max_epochs=trainer.max_epochs,
            steps_per_epoch=steps_per_epoch,total_steps=steps_per_epoch*trainer.max_epochs,
            optimizer=type(optimizer).__name__,optimizer_groups=len(optimizer.param_groups),
            optimizer_initial_lr=[group["initial_lr"] for group in optimizer.param_groups],
            num_points=model.criterion.num_points,oversample_ratio=model.criterion.oversample_ratio,
            importance_sample_ratio=model.criterion.importance_sample_ratio,
            coefficients={"mask":model.criterion.mask_coefficient,"dice":model.criterion.dice_coefficient,
                          "class":model.criterion.class_coefficient,"no_object":model.criterion.eos_coef},
            state_tensor_count=len(states),state_sha256=states,
            rng_sha256={"python":hashlib.sha256(pickle.dumps(random.getstate())).hexdigest(),
                        "numpy":hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest(),
                        "torch":hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()},
            worker_seed_setting=os.getenv("PL_SEED_WORKERS"),seed=os.getenv("PL_GLOBAL_SEED"),
            cuda_initialized=torch.cuda.is_initialized(),split_summary=data.split_summary,
            cpu_probe_overrides={"accelerator":"cpu","precision":"32-true","logger":False,
                                 "enable_checkpointing":False,"callbacks":[],"compile_disabled":True},
            actual_fit_executed=False)
        assert type(model.network).__module__=="models.eoseg"
        assert type(model.criterion).__module__=="training.mask_classification_loss"
        assert model.network.num_q==6 and model.num_classes==4 and model.lr==5e-4
        assert model.img_size==(224,224) or list(model.img_size)==[224,224]
        assert not torch.cuda.is_initialized()
    sys.argv=["main.py","fit","--config","configs/acdc/vit_query_mul_scale_fusion.yaml",
              "--trainer.accelerator=cpu","--trainer.precision=32-true","--trainer.logger=false",
              "--trainer.enable_checkpointing=false","--trainer.callbacks=[]","--compile_disabled"]
    with patch("lightning.pytorch.Trainer.fit",capture):
        cli_main()
    weight=Path(os.environ["TORCH_HOME"])/"hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth"
    captured["cached_pretrained"]={"path":str(weight),"sha256":file_sha(weight)}
    return captured


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output_dir",type=Path,required=True)
    parser.add_argument("--initialize",action="store_true")
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    result={"data":audit_data(),"formal_training_started":False}
    if args.initialize:result["initialization"]=initialize_cli()
    (args.output_dir/"acdc_audit.json").write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print(json.dumps({key:value for key,value in result.items() if key!="initialization"},ensure_ascii=False,default=dict))
    if args.initialize:
        init=result["initialization"]
        print(json.dumps({key:value for key,value in init.items() if key not in ("state_sha256","optimizer_initial_lr","split_summary")},indent=2))

if __name__=="__main__":
    main()
