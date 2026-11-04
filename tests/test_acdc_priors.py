"""ACDC foreground prior and class-compatible loading; CPU only."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import torch
from torch import nn
from timm.models.vision_transformer import VisionTransformer

from models.asee import load_prior,validate_prior
from models.eoseg import EoSegMultiQSegFusion
from models.eoseg_asee import EoSegMultiQSegFusionASEE
from models.eoseg_asee_ccps import EoSegMultiQSegFusionASEECCPS
from scripts.build_acdc_asee_prior import count_training_slices,load_label_mapping
from scripts.build_synapse_asee_prior import conditional_from_counts

ROOT=Path(__file__).resolve().parents[1]


def encoder():
    out=nn.Module()
    out.backbone=VisionTransformer(img_size=32,patch_size=16,embed_dim=32,depth=4,
                                  num_heads=4,reg_tokens=4,num_classes=0)
    out.register_buffer("pixel_mean",torch.zeros(1,3,1,1))
    out.register_buffer("pixel_std",torch.ones(1,3,1,1))
    return out


class ACDCPriorTests(unittest.TestCase):
    def test_real_prior_shape_and_training_isolation(self):
        prior,digest=load_prior("priors/acdc_asee_80_20_v1.pt")
        self.assertEqual(prior["class_ids"],[1,2,3])
        self.assertEqual(tuple(prior["text_features"].shape),(3,768))
        self.assertEqual(tuple(prior["conditional_matrix"].shape),(3,3))
        self.assertFalse(prior["statistics"]["validation_test_labels_used"])
        self.assertEqual(prior["statistics"]["train_cases"],[f"patient{i:03d}" for i in range(21,101)])
        self.assertEqual(len(prior["statistics"]["train_h5_sha256"]),1512)
        expected=(ROOT/"ACDC/lists_ACDC/train.txt").read_text().splitlines()
        self.assertEqual(set(prior["statistics"]["train_h5_sha256"]),set(expected))
        self.assertEqual(len(digest),64)
        stats=prior["statistics"]
        expected_matrix=conditional_from_counts(stats["class_counts"],stats["cooccurrence_counts"],1512)
        torch.testing.assert_close(prior["conditional_matrix"],expected_matrix,rtol=0,atol=0)

    def test_background_is_not_a_prior_row(self):
        mapping=load_label_mapping(ROOT/"priors/acdc_label_map.verified.json")
        self.assertEqual(mapping["background_label"],0)
        self.assertEqual([r["label_id"] for r in mapping["classes"]],[1,2,3])
        prior,_=load_prior("priors/acdc_asee_80_20_v1.pt")
        self.assertNotIn(0,prior["class_ids"])

    def test_bad_rows_or_manifest_rejected(self):
        prior,_=load_prior("priors/acdc_asee_80_20_v1.pt")
        for key,value in (("class_ids",[0,1,2,3]),("conditional_matrix",torch.ones(4,4))):
            bad=copy.deepcopy(prior);bad[key]=value
            with self.assertRaises(ValueError):validate_prior(bad)
        bad=copy.deepcopy(prior);bad["statistics"]["val_list_sha256"]="wrong"
        with self.assertRaises(ValueError):validate_prior(bad)

    def test_counts_and_log_ratio_are_foreground_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            names=[]
            for i,labels in enumerate(([[0,1],[2,0]],[[2,3],[0,0]],[[0,0],[0,0]])):
                name=f"patient031_frame01_slice_{i}.h5";names.append(name)
                with h5py.File(Path(tmp)/name,"w") as f:f["label"]=np.array(labels,dtype=np.uint8)
            counts,pairs,files=count_training_slices(tmp,names)
            np.testing.assert_array_equal(counts,[1,2,1])
            np.testing.assert_array_equal(pairs,[[1,1,0],[1,2,1],[0,1,1]])
            expected=conditional_from_counts(counts,pairs,3)
            self.assertEqual(tuple(expected.shape),(3,3))
            self.assertTrue(torch.isfinite(expected).all())
            self.assertEqual(len(files),3)
            with self.assertRaises(ValueError):count_training_slices(tmp,["patient001_frame01_slice_0.h5"])

    def test_asee_three_rows_six_queries_forward_backward(self):
        model=EoSegMultiQSegFusionASEE(encoder(),4,"priors/acdc_asee_80_20_v1.pt")
        masks,classes=model(torch.rand(2,3,32,32))
        self.assertEqual(model.num_q,6)
        self.assertEqual(tuple(masks[0].shape),(2,6,32,32))
        self.assertEqual(tuple(classes[0].shape),(2,6,5))
        (masks[0].square().mean()+classes[0].square().mean()).backward()
        self.assertIsNotNone(model.asee.text_proj[0].weight.grad)
        self.assertIsNotNone(model.asee.prob_proj[0].weight.grad)

    def test_ccps_selector_has_three_outputs(self):
        model=EoSegMultiQSegFusionASEECCPS(encoder(),4,"priors/acdc_asee_80_20_v1.pt",selector_hidden_dim=16)
        masks,classes=model(torch.rand(2,3,32,32))
        self.assertEqual(model.num_q,6)
        self.assertEqual(tuple(model.asee.last_selector_weights.shape),(2,3))
        self.assertEqual(tuple(classes[0].shape),(2,6,5))
        (masks[0].square().mean()+classes[0].square().mean()).backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.asee.selector.parameters()))

    def test_disabled_asee_exact_baseline(self):
        baseline=EoSegMultiQSegFusion(encoder(),4,6)
        model=EoSegMultiQSegFusionASEE(encoder(),4,"priors/acdc_asee_80_20_v1.pt",enhance_patch=False,enhance_query=False)
        model.load_state_dict(baseline.state_dict(),strict=False)
        image=torch.rand(2,3,32,32)
        for training in (False,True):
            baseline.train(training);model.train(training)
            for actual,expected in zip(model(image),baseline(image)):
                torch.testing.assert_close(actual[0],expected[0],rtol=0,atol=0)

    def test_incompatible_semantic_or_query_count_rejected(self):
        for n,q in ((9,16),(4,8)):
            with self.assertRaises(ValueError):
                EoSegMultiQSegFusionASEE(encoder(),n,"priors/acdc_asee_80_20_v1.pt",num_q=q)

    def test_pretrained_uses_generic_weights_not_class_slicing(self):
        value=json.loads((ROOT/"pretrained/acdc_pretrained_manifest.json").read_text())
        self.assertEqual(value["segmentation"]["num_classes"],4)
        self.assertEqual(value["segmentation"]["foreground_label_ids"],[1,2,3])
        self.assertEqual(value["text_encoder"]["output_dim"],768)
        self.assertFalse(value["vision_encoder"]["class_specific"])
        self.assertFalse(value["text_encoder"]["class_specific"])


if __name__=="__main__":
    torch.set_num_threads(2)
    unittest.main()
