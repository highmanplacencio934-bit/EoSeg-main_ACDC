"""ACDC H5, fixed split, Dice and callback regression tests; CPU only."""
import csv
import hashlib
import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import h5py
import numpy as np
import torch

from datasets.acdc_dataset import ACDCBaselineTransform, ACDCSliceDataset, ACDCVolumeDataset, read_acdc_h5
from datasets.acdc_semantic import ACDCSemantic
from datasets.acdc_splits import read_manifest, validate_splits
from training.acdc_mask_classification_semantic import ACDCMaskClassificationSemantic
from training.acdc_test_logger import ACDCTestLogger

ROOT = Path(__file__).resolve().parents[1]


class ACDCDataTests(unittest.TestCase):
    def test_real_fixed_patient_split(self):
        info = validate_splits(ROOT / "ACDC", ROOT / "ACDC/lists_ACDC", "fit")
        self.assertEqual(info["counts"], {"train": 1512, "val": 40, "test": 40})
        self.assertEqual(info["patient_counts"], {"train": 80, "val": 20, "test": 20})

    def test_fit_and_independent_test_share_heldout_volumes(self):
        dm=ACDCSemantic(base_dir=str(ROOT/"ACDC"),list_dir=str(ROOT/"ACDC/lists_ACDC"),
                        batch_size=16,num_workers=0)
        dm.setup("fit")
        self.assertEqual(dm.split_summary["fit_evaluation_split"],"test")
        self.assertFalse(dm.split_summary["independent_val_split_used_for_selection"])
        self.assertTrue(dm.split_summary["test_used_for_selection"])
        self.assertEqual(dm.eval_dataset.split,"test")
        self.assertEqual(len(dm.eval_dataset),40)
        self.assertFalse(hasattr(dm,"test_dataset"))
        dm.setup("test")
        self.assertEqual(dm.test_dataset.split,"test")
        self.assertEqual(len(dm.test_dataset),40)
        self.assertEqual(dm.eval_dataset.cases,dm.test_dataset.cases)
        train_patients={name[7:10] for name in dm.train_dataset.sample_list}
        self.assertFalse(train_patients & {name[7:10] for name in dm.eval_dataset.cases})

    def test_checkpoint_config_uses_shared_test_score(self):
        import yaml
        config=yaml.safe_load((ROOT/"configs/acdc/vit_query_mul_scale_fusion.yaml").read_text())
        checkpoint=config["trainer"]["callbacks"][0]["init_args"]
        self.assertEqual(checkpoint["monitor"],"metrics/test_mean_dice")
        self.assertEqual(checkpoint["filename"],"best_test")
        self.assertEqual(config["trainer"]["max_epochs"],400)
        self.assertEqual(config["trainer"]["check_val_every_n_epoch"],1)

    def test_rgb_and_mask_contract(self):
        dm = ACDCSemantic(base_dir=str(ROOT/"ACDC"), list_dir=str(ROOT/"ACDC/lists_ACDC"),
                          batch_size=16, num_workers=0, img_size=(224,224))
        dm.setup("fit")
        image, target = dm.train_dataset[0]
        self.assertEqual(tuple(image.shape), (3,224,224))
        self.assertTrue(torch.equal(image[0], image[1]))
        self.assertEqual(target["masks"].dtype, torch.bool)
        self.assertTrue(set(target["labels"].tolist()) <= {0,1,2,3})
        self.assertEqual(len(dm.val_dataloader().dataset),40)
        self.assertEqual(len(dm.train_dataloader()),94)

    def test_h5_read_and_volume_normalization(self):
        ds=ACDCVolumeDataset(str(ROOT/"ACDC"), str(ROOT/"ACDC/lists_ACDC"),"test")
        item=ds[0]
        with h5py.File(ROOT/"ACDC/ACDC_training_volumes"/ds.cases[0],"r") as f:
            raw=f["image"][:]
            np.testing.assert_array_equal(item["image"].numpy(), (raw-raw.min())/(raw.max()-raw.min()))
            np.testing.assert_array_equal(item["label"].numpy(),f["label"][:])
        self.assertEqual(item["image"].ndim,3)

    def test_unsafe_duplicate_or_wrong_patient_rejected(self):
        for value in ("../patient031_frame01_slice_0.h5\n", "patient001_frame01_slice_0.h5\n",
                      "patient031_frame01_slice_0.h5\npatient031_frame01_slice_0.h5\n"):
            with tempfile.TemporaryDirectory() as tmp:
                Path(tmp,"train.txt").write_text(value)
                with self.assertRaises(ValueError):read_manifest(tmp,"train")

    def test_label_not_scribble_used_and_invalid_values_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp,"sample.h5")
            with h5py.File(p,"w") as f:
                f["image"]=np.ones((4,4),np.float32)
                f["label"]=np.ones((4,4),np.uint8)
                f["scribble"]=np.full((4,4),4,np.uint8)
            _,label=read_acdc_h5(p,2,255)
            self.assertTrue((label==1).all())
            with h5py.File(p,"r+") as f:f["label"][0,0]=7
            with self.assertRaises(ValueError):read_acdc_h5(p,2,255)

    def test_original_transform_seed_replay(self):
        image=np.arange(60,dtype=np.float32).reshape(6,10)
        label=(image%4).astype(np.uint8)
        transform=ACDCBaselineTransform((16,16))
        random.seed(0);np.random.seed(0)
        a,b=transform(image.copy(),label.copy())
        random.seed(0);np.random.seed(0)
        c,d=transform(image.copy(),label.copy())
        np.testing.assert_array_equal(a,c);np.testing.assert_array_equal(b,d)


class ACDCMetricTests(unittest.TestCase):
    def test_legacy_empty_mask_branches(self):
        metric=ACDCMaskClassificationSemantic.calculate_metric_percase_baseline
        a=np.ones((2,2),np.uint8);z=np.zeros_like(a)
        self.assertEqual(metric(a,z)[0],1)
        self.assertEqual(metric(z,z)[0],0)
        self.assertEqual(metric(z,a)[0],0)
        self.assertEqual(metric(a,a)[0],1)

    def test_case_dice_only_no_hd95_call(self):
        label=np.array([[[0,1],[2,3]]])
        module=ACDCMaskClassificationSemantic.__new__(ACDCMaskClassificationSemantic)
        torch.nn.Module.__init__(module);module.num_classes=4
        with patch("medpy.metric.binary.hd95",side_effect=AssertionError("HD95 forbidden")):
            result=module._compute_case_metrics(label,label)
        self.assertEqual(result,{"mean_dice":1.0,"dice_class_1":1.0,"dice_class_2":1.0,"dice_class_3":1.0})

    def test_logger_uses_fresh_post_module_metrics_and_null_hd95(self):
        with tempfile.TemporaryDirectory() as tmp:
            cb=ACDCTestLogger()
            t=SimpleNamespace(log_dir=tmp,default_root_dir=tmp,current_epoch=0,global_step=82,
                              max_epochs=400,sanity_checking=False,callback_metrics={},
                              checkpoint_callback=SimpleNamespace(best_model_path="best_test.ckpt",last_model_path="last.ckpt"))
            model=SimpleNamespace(eval_case_results=[])
            for epoch,score in enumerate((0.3,0.8,0.4)):
                t.current_epoch=epoch
                t.callback_metrics={"metrics/test_mean_dice":torch.tensor(score),
                                    "metrics/val_mean_dice":torch.tensor(1.0),
                                    **{f"metrics/test_dice_class_{c}":torch.tensor(score) for c in range(1,4)}}
                model.eval_case_results=[{"case":"patient001_frame01.h5","metrics":{"mean_dice":score,
                                          **{f"dice_class_{c}":score for c in range(1,4)}}}]
                cb.on_validation_end(t,model)
            cb.on_fit_end(t,model)
            with Path(tmp,"acdc_test_history.csv").open() as handle:
                rows=list(csv.DictReader(handle))
            self.assertEqual([r["epoch"] for r in rows],["1","2","3"])
            self.assertAlmostEqual(float(rows[1]["avg_dsc"]),0.8,places=6)
            self.assertTrue(all(r["hd95_mm"]==r["hd95_voxel"]=="" for r in rows))
            summary=json.loads(Path(tmp,"selection_summary.json").read_text())
            self.assertEqual(summary["best_epoch"],2)
            self.assertIsNone(summary["hd95_mm"])
            self.assertEqual(summary["evaluation_split"],"test")
            self.assertTrue(summary["test_used_for_selection"])

    def test_independent_test_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            cb=ACDCTestLogger()
            t=SimpleNamespace(log_dir=tmp,default_root_dir=tmp)
            m=SimpleNamespace(test_case_results=[{"case":"patient001_frame01.h5","metrics":
              {"mean_dice":0.5,"dice_class_1":0.4,"dice_class_2":0.5,"dice_class_3":0.6}}])
            cb.on_test_end(t,m)
            result=json.loads(Path(tmp,"acdc_independent_test_summary.json").read_text())
            self.assertAlmostEqual(result["mean_dice"],0.5)
            self.assertIsNone(result["hd95_voxel"])
            self.assertEqual(result["evaluation_split"],"test")
            self.assertTrue(result["shares_cases_with_model_selection"])

class ACDCLightningIntegrationTests(unittest.TestCase):
    def test_two_epoch_real_loss_logging_and_best_last(self):
        import yaml
        from main import LightningCLI, LightningModule, LightningDataModule
        from timm.models.vision_transformer import VisionTransformer

        def tiny_factory(name, **kwargs):
            return VisionTransformer(img_size=kwargs["img_size"],patch_size=kwargs["patch_size"],
                                     embed_dim=32,depth=3,num_heads=4,num_classes=0)

        with tempfile.TemporaryDirectory() as tmp:
            config=yaml.safe_load((ROOT/"configs/acdc/vit_query_mul_scale_fusion.yaml").read_text())
            config["model"]["init_args"]["img_size"]=[32,32]
            config["model"]["init_args"]["num_points"]=32
            config["model"]["init_args"]["network"]["init_args"]["encoder"]["init_args"]["img_size"]=[32,32]
            config["data"]["init_args"].update(img_size=[32,32],batch_size=2,num_workers=0)
            config["trainer"].update(accelerator="cpu",precision="32-true",devices=1,max_epochs=2,
                enable_model_summary=False,enable_progress_bar=False,log_every_n_steps=1,
                limit_train_batches=2,limit_val_batches=2,limit_test_batches=2)
            config["trainer"]["logger"]["init_args"].update(save_dir=tmp,name="cpu_integration")
            config_path=Path(tmp)/"cpu_config.yaml"
            config_path.write_text(yaml.safe_dump(config))
            with patch("models.vit.timm.create_model",side_effect=tiny_factory):
                cli=LightningCLI(LightningModule,LightningDataModule,run=False,
                    args=["--config",str(config_path)],subclass_mode_model=True,
                    subclass_mode_data=True,save_config_callback=None,seed_everything_default=0)
            model,data,trainer=cli.model,cli.datamodule,cli.trainer
            logger=trainer.logger
            checkpoint=trainer.checkpoint_callback
            with patch("medpy.metric.binary.hd95",side_effect=AssertionError("HD95 forbidden")):
                with patch.object(data,"test_dataloader",side_effect=AssertionError("Test during fit forbidden")):
                    trainer.fit(model,datamodule=data)
            path=Path(logger.log_dir)
            with (path/"acdc_test_history.csv").open() as handle:
                rows=list(csv.DictReader(handle))
            self.assertEqual([r["epoch"] for r in rows],["1","2"])
            with (path/"acdc_test_case_metrics.csv").open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))),4)
            self.assertTrue(Path(checkpoint.best_model_path).is_file())
            self.assertTrue(Path(checkpoint.last_model_path).is_file())
            summary=json.loads((path/"selection_summary.json").read_text())
            self.assertEqual(summary["last_evaluated_epoch"],2)
            self.assertEqual(summary["best_checkpoint"],checkpoint.best_model_path)
            self.assertEqual(summary["evaluation_split"],"test")
            self.assertIn("metrics/test_mean_dice",trainer.callback_metrics)
            self.assertNotIn("metrics/val_mean_dice",trainer.callback_metrics)
            manifest=json.loads((path/"run_manifest.json").read_text())
            self.assertEqual(manifest["split_summary"]["fit_evaluation_split"],"test")
            self.assertEqual(manifest["selection_policy"],"test set used for checkpoint selection; no gradient updates")
            with (path/"acdc_test_case_metrics.csv").open() as handle:
                cases={row["case"] for row in csv.DictReader(handle)}
            self.assertTrue(all(1<=int(name[7:10])<=20 for name in cases))
            selection_before=(path/"selection_summary.json").read_bytes()
            best_before=hashlib.sha256(Path(checkpoint.best_model_path).read_bytes()).hexdigest()
            with patch("medpy.metric.binary.hd95",side_effect=AssertionError("HD95 forbidden")):
                trainer.test(model,datamodule=data,ckpt_path=checkpoint.best_model_path)
            self.assertEqual((path/"selection_summary.json").read_bytes(),selection_before)
            self.assertEqual(hashlib.sha256(Path(checkpoint.best_model_path).read_bytes()).hexdigest(),best_before)
            with (path/"acdc_independent_test_cases.csv").open() as handle:
                test_cases={row["case"] for row in csv.DictReader(handle)}
            self.assertTrue(all(1<=int(name[7:10])<=20 for name in test_cases))
            self.assertEqual(cases,test_cases)


if __name__=="__main__":
    unittest.main()
