"""Regression coverage for public physical-wrapper constructor options."""
from pathlib import Path
import json,tempfile,unittest,hashlib
import numpy as np
import torch
from torch import nn
from training.mask_classification_semantic import MaskClassificationSemantic
from training.synapse_physical_semantic import SynapsePhysicalHD95Semantic
from training.deterministic_protocol import install_criterion

class TinyNetwork(nn.Module):
    num_blocks=3
    masked_attn_enabled=False
    def __init__(self):
        super().__init__()
        self.weight=nn.Parameter(torch.randn(2,3))

class ParameterPassthroughTests(unittest.TestCase):
    overrides=dict(ignore_idx=77,num_points=37,oversample_ratio=2.5,
        importance_sample_ratio=.5,no_object_coefficient=.2,
        mask_coefficient=2.,dice_coefficient=3.,class_coefficient=4.,
        mask_thresh=.7,overlap_thresh=.6)

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.manifest=Path(self.tmp.name)/"spacing.json"
        # Construction only: case validation is covered by existing geometry tests.
        self.manifest.write_text(json.dumps(dict(schema_version=1,units="mm",
            array_axis_order="zyx",verified=True,orthogonal_grid=True,cases={"case0001":dict(spacing_zyx_mm=[1.,1.,1.],shape_zyx=[1,8,8],label_sha256=hashlib.sha256(np.zeros((1,8,8),np.uint8).tobytes()).hexdigest(),source="SYNTHETIC_TEST_FIXTURE",mapping="identity")})))

    def build(self,wrapper=True,**options):
        kwargs=dict(network=TinyNetwork(),img_size=(8,8),num_classes=9,
                    attn_mask_annealing_enabled=False,**options)
        if wrapper:
            return SynapsePhysicalHD95Semantic(spacing_manifest=str(self.manifest),**kwargs)
        return MaskClassificationSemantic(**kwargs)

    def settings(self,m):
        c=m.criterion
        return dict(ignore_idx=m.ignore_idx,num_points=c.num_points,
            oversample_ratio=c.oversample_ratio,
            importance_sample_ratio=c.importance_sample_ratio,
            no_object_coefficient=c.eos_coef,mask_coefficient=c.mask_coefficient,
            dice_coefficient=c.dice_coefficient,class_coefficient=c.class_coefficient,
            mask_thresh=m.mask_thresh,overlap_thresh=m.overlap_thresh)

    def test_each_override_matches_parent(self):
        for key,value in self.overrides.items():
            with self.subTest(option=key):
                m=self.build(**{key:value})
                parent=self.build(wrapper=False,**{key:value})
                self.assertEqual(self.settings(m),self.settings(parent))

    def test_combined_overrides_survive_deterministic_criterion(self):
        m=self.build(**self.overrides)
        self.assertEqual(self.settings(m),self.overrides)
        self.assertTrue(install_criterion(m))
        self.assertEqual(self.settings(m),self.overrides)
        self.assertAlmostEqual(m.criterion.empty_weight[-1].item(),.2,places=7)

    def test_default_state_and_rng_equal_parent(self):
        torch.manual_seed(14);parent=self.build(wrapper=False);before=torch.get_rng_state().clone()
        torch.manual_seed(14);m=self.build();after=torch.get_rng_state()
        self.assertTrue(torch.equal(before,after))
        self.assertEqual(self.settings(m),self.settings(parent))
        self.assertEqual(set(m.state_dict()),set(parent.state_dict()))
        for k,v in parent.state_dict().items():
            self.assertTrue(torch.equal(v,m.state_dict()[k]),k)
        self.assertFalse(m.compute_hd95_voxel)

    def test_default_loss_and_input_gradients_equal_parent(self):
        for deterministic in (False,True):
            with self.subTest(deterministic=deterministic):
                torch.manual_seed(42);parent=self.build(wrapper=False)
                torch.manual_seed(42);m=self.build()
                if deterministic:install_criterion(parent);install_criterion(m)
                torch.manual_seed(81)
                masks=torch.randn(2,4,8,8)
                classes=torch.randn(2,4,10)
                gt=torch.zeros(2,8,8);gt[0,1:5,2:6]=1;gt[1,4:7,3:6]=1
                targets=[dict(labels=torch.tensor([1,3]),masks=gt.clone()) for _ in range(2)]
                outputs=[]
                for model in (parent,m):
                    pm=masks.clone().requires_grad_();pc=classes.clone().requires_grad_()
                    torch.manual_seed(102)
                    losses=model.criterion(masks_queries_logits=pm,class_queries_logits=pc,targets=targets)
                    total=model.criterion.loss_total(losses,lambda *a,**k:None)
                    total.backward()
                    outputs.append((losses,total.detach(),pm.grad,pc.grad))
                a,b=outputs
                self.assertEqual(set(a[0]),set(b[0]))
                for k in a[0]:self.assertTrue(torch.equal(a[0][k],b[0][k]),k)
                for x,y in zip(a[1:],b[1:]):self.assertTrue(torch.equal(x,y))

if __name__=="__main__":unittest.main()
