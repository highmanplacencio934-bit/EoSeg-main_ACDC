from pathlib import Path
import sys,json,tempfile,csv,unittest
from types import SimpleNamespace
from unittest.mock import patch,Mock
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'tests'),str(Path(__file__).resolve().parent)]
import numpy as np,torch,h5py
from lightning.pytorch import Trainer,LightningDataModule
from torch.utils.data import DataLoader
from medpy import metric
from metric_mode import dice_record,metric_scope,summary_dice,MetricModeSelection
from training.synapse_physical_semantic import SynapsePhysicalHD95Semantic
from training.synapse_roi_evaluation import roi_metrics_scope
from test_synapse_physical_hd95 import TinyNetwork,make_manifest

class MetricModeTests(unittest.TestCase):
 def fixture(self,root):
  label=np.zeros((2,8,8),np.uint8);label[:,2,2]=1
  spacing=root/'spacing.json';make_manifest(spacing,{'case0001':label})
  model=SynapsePhysicalHD95Semantic(network=TinyNetwork(),spacing_manifest=str(spacing),img_size=(8,8),num_classes=9,attn_mask_annealing_enabled=False).eval()
  batch={'case_name':'case0001','image':torch.zeros(label.shape),'label':torch.from_numpy(label)}
  return model,batch
 def test_legacy_empty_cases_and_no_distance_call(self):
  z=np.zeros((2,4,4),bool);a=z.copy();a[0,1,1]=True
  with patch('medpy.metric.binary.hd95',side_effect=AssertionError('HD95 called')):
   for pred,gt,value in [(a,a,1),(a,z,1),(z,a,0),(z,z,0)]:self.assertEqual(dice_record(pred,gt)['dice'],value)
 def test_single_view_preserves_dice_and_skips_hd95(self):
  with tempfile.TemporaryDirectory() as tmp:
   model,batch=self.fixture(Path(tmp))
   with roi_metrics_scope(model):
    expected=model.evaluate_synapse_case(batch)
    with metric_scope(model,False),patch('medpy.metric.binary.hd95',side_effect=AssertionError('HD95 called')):
     actual=model.evaluate_synapse_case(batch)
     self.assertEqual(actual['dice_per_class'],expected['dice_per_class'])
     self.assertEqual(actual['mean_dice'],expected['mean_dice'])
     self.assertIsNone(actual['mean_hd95'])
     self.assertEqual(actual['hd95_per_class'],[None]*8)
 def test_enabled_scope_retains_physical_distance(self):
  with tempfile.TemporaryDirectory() as tmp:
   model,batch=self.fixture(Path(tmp))
   with roi_metrics_scope(model),metric_scope(model,True),patch('medpy.metric.binary.hd95',wraps=metric.binary.hd95) as hd:
    result=model.evaluate_synapse_case(batch)
    self.assertGreater(hd.call_count,0);self.assertTrue(all('voxelspacing' in call.kwargs for call in hd.call_args_list));self.assertIsNotNone(result['mean_hd95_mm'])
 def test_scope_restore_on_error_and_conflicting_flags(self):
  with tempfile.TemporaryDirectory() as tmp:
   model,batch=self.fixture(Path(tmp));original=model.evaluate_synapse_case
   with self.assertRaises(RuntimeError):
    with metric_scope(model):raise RuntimeError('test')
   self.assertEqual(model.evaluate_synapse_case,original);self.assertNotIn('compute_hd95_mm',model.__dict__)
   model.compute_hd95_voxel=True
   with self.assertRaises(ValueError):
    with metric_scope(model):pass
   self.assertNotIn('compute_hd95_mm',model.__dict__)
 def test_dice_summary_matches_original_and_rejects_missing(self):
  from training.synapse_selection import summarize_cases
  results=[{'case':'a','dice_per_class':[.3,.7],'hd95_per_class':[None,None]},{'case':'b','dice_per_class':[.5,.9],'hd95_per_class':[None,None]}]
  original=summarize_cases([dict(x,hd95_per_class=[0,0]) for x in results],2,2)
  actual=summary_dice(results,2,2)
  for k in ('mean_dice','dice_class_1','dice_class_2'):self.assertEqual(actual[k],original[k])
  self.assertIsNone(actual['mean_hd95'])
  with self.assertRaises(ValueError):summary_dice(results[:1],2,2)
 def test_cpu_lightning_export_is_null_not_zero_and_test_logs_dice_only(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);model,batch=self.fixture(root)
   (root/'test_vol.txt').write_text('case0001\n')
   with h5py.File(root/'case0001.npy.h5','w') as f:f['image']=np.zeros((2,8,8),np.float32);f['label']=batch['label'].numpy()
   class Data(LightningDataModule):
    test_base_dir=str(root);list_dir=str(root);test_split='test_vol'
    def test_dataloader(self):return DataLoader([batch],batch_size=None)
   cb=MetricModeSelection(expected_test_cases=1,output_dir=str(root/'out'))
   trainer=Trainer(accelerator='cpu',devices=1,precision='32-true',logger=False,enable_checkpointing=False,enable_progress_bar=False,enable_model_summary=False,callbacks=[cb])
   with roi_metrics_scope(model),metric_scope(model),patch('medpy.metric.binary.hd95',side_effect=AssertionError('HD95 called')):
    trainer.test(model,datamodule=Data(),verbose=False)
   summary=json.loads((cb.run_dir/'physical_test_summary.json').read_text());self.assertIsNone(summary['mean_hd95_mm']);self.assertFalse(summary['hd95_mm_enabled'])
   with (cb.run_dir/'physical_case_organ_metrics.csv').open() as f:rows=list(csv.DictReader(f))
   self.assertEqual(len(rows),8);self.assertTrue(all(x['hd95_mm']==x['hd95_voxel']=='' for x in rows))
   self.assertFalse(any('hd95' in k for k in trainer.logged_metrics))
 def test_last50_schedule_best_last_and_null_training_csv(self):
  with tempfile.TemporaryDirectory() as tmp:
   cb=MetricModeSelection(output_dir=tmp,expected_test_cases=1);cb.run_dir=Path(tmp)
   result=dict(case='case1',mean_dice=.5,mean_hd95=None,dice_per_class=[.5]*8,hd95_per_class=[None]*8)
   summary=summary_dice([result],1,8);cb._evaluate=Mock(return_value=([result],summary))
   trainer=SimpleNamespace(current_epoch=169,max_epochs=220,global_step=1,logger=None,save_checkpoint=Mock())
   module=SimpleNamespace(compute_hd95_mm=False)
   cb.on_train_epoch_end(trainer,module);cb._evaluate.assert_not_called()
   trainer.current_epoch=170;cb.on_train_epoch_end(trainer,module)
   self.assertEqual(cb.best_epoch,171);self.assertEqual(cb.best_score,.5)
   trainer.current_epoch=171;cb.on_train_epoch_end(trainer,module)
   self.assertEqual(cb.best_epoch,171)
   paths=[str(x.args[0]) for x in trainer.save_checkpoint.call_args_list]
   self.assertEqual(sum(x.endswith('best_test.ckpt') for x in paths),1);self.assertEqual(sum(x.endswith('last.ckpt') for x in paths),3)
   with (cb.run_dir/'test_epoch_metrics.csv').open() as f:rows=list(csv.DictReader(f))
   self.assertEqual(len(rows),2);self.assertTrue(all(x['mean_hd95']=='' for x in rows))
if __name__=='__main__':unittest.main()
