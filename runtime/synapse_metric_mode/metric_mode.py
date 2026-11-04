"""Optional Dice-only evaluation. Original inference, losses and HD95 implementations stay intact."""
from contextlib import contextmanager
from types import MethodType,FunctionType
from pathlib import Path
import json,logging,math,sys,hashlib
from statistics import fmean
import numpy as np
from medpy import metric
from training.synapse_physical_selection import SynapsePhysicalHD95Selection
from training.synapse_test_selection import SynapseTestSelection
from training.synapse_selection import append_csv,should_evaluate

def dice_record(pred,gt,spacing=None,**kwargs):
 pred,gt=np.asarray(pred,dtype=bool),np.asarray(gt,dtype=bool)
 if pred.ndim!=3 or pred.shape!=gt.shape:raise ValueError('Equal-shape 3D masks required')
 a,b=bool(pred.any()),bool(gt.any())
 status='nonempty' if a and b else 'empty_gt_legacy_zero' if a else 'empty_pred_legacy_zero' if b else 'both_empty_legacy_zero'
 # Zeros satisfy the unchanged inference function's numeric return contract only.
 # Every public result/export below replaces all disabled distances with None.
 return dict(dice=float(metric.binary.dc(pred,gt)) if a and b else 1.0 if a and not b else 0.0,hd95_mm=0.0,hd95_voxel=None,status=status)

def sanitize(result,classes):
 result=dict(result)
 for key in ('mean_hd95','mean_hd95_mm','mean_hd95_voxel'):result[key]=None
 for key in ('hd95_per_class','hd95_mm_per_class','hd95_voxel_per_class'):result[key]=[None]*classes
 result.update(hd95_mm_enabled=False,hd95_voxel_enabled=False)
 return result

@contextmanager
def metric_scope(model,compute_hd95_mm=False):
 if type(compute_hd95_mm) is not bool:raise ValueError('compute_hd95_mm must be boolean')
 if not compute_hd95_mm and model.compute_hd95_voxel:raise ValueError('Dice-only mode requires compute_hd95_voxel=false')
 oldflag=model.__dict__.get('compute_hd95_mm','MISSING')
 model.compute_hd95_mm=compute_hd95_mm
 if compute_hd95_mm:
  try:yield model
  finally:
   if oldflag=='MISSING':model.__dict__.pop('compute_hd95_mm',None)
   else:model.compute_hd95_mm=oldflag
  return
 originals={k:model.__dict__.get(k,'MISSING') for k in ('calculate_metric_percase','evaluate_synapse_case','test_step')}
 evaluate=model.evaluate_synapse_case
 try:
  def calculate(self,pred,gt):
   if self._physical_case_context is None:raise RuntimeError('Full-case geometry context required')
   spacing,records=self._physical_case_context
   row=dice_record(pred,gt,spacing);records.append(row)
   return row['dice'],0.0
  model.calculate_metric_percase=MethodType(calculate,model)
  def case(self,batch):return sanitize(evaluate(batch),self.num_classes-1)
  model.evaluate_synapse_case=MethodType(case,model)
  def test_step(self,batch,batch_idx):
   result=self.evaluate_synapse_case(batch)
   for i,value in enumerate(result['dice_per_class'],1):self.log('test/dice_class_'+str(i),value,sync_dist=True,batch_size=1)
   self.log('test/mean_dice',result['mean_dice'],sync_dist=True,batch_size=1)
   return result
  model.test_step=MethodType(test_step,model)
  yield model
 finally:
  for key,value in originals.items():
   if isinstance(value,str) and value=='MISSING':model.__dict__.pop(key,None)
   else:setattr(model,key,value)
  if oldflag=='MISSING':model.__dict__.pop('compute_hd95_mm',None)
  else:model.compute_hd95_mm=oldflag

def summary_dice(results,expected,classes):
 if len(results)!=expected or len({x['case'] for x in results})!=expected:raise ValueError('Incomplete/duplicate cases')
 for x in results:
  if len(x['dice_per_class'])!=classes or not all(math.isfinite(v) and 0<=v<=1 for v in x['dice_per_class']):raise ValueError('Invalid Dice')
 result={'num_cases':expected}
 for i in range(classes):result['dice_class_'+str(i+1)]=fmean(x['dice_per_class'][i] for x in results);result['hd95_class_'+str(i+1)]=None
 result['mean_dice']=fmean(result['dice_class_'+str(i+1)] for i in range(classes));result['mean_hd95']=None
 return result

class MetricModeSelection(SynapsePhysicalHD95Selection):
 def _prepare_reports(self,module):
  super()._prepare_reports(module)
  p=self.run_dir/'metric_units.json';d=json.loads(p.read_text());d.update(hd95_mm_enabled=getattr(module,'compute_hd95_mm',True),metric_mode='dice_and_mm' if getattr(module,'compute_hd95_mm',True) else 'dice_only');p.write_text(json.dumps(d,ensure_ascii=False,indent=2))
  ready=Path(__file__).with_name('readiness.json')
  manifest_path=self.run_dir/'run_manifest.json'
  manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {'stage':'independent_test','argv':sys.argv,'precision':module.trainer.precision,'device':str(module.device)}
  manifest.update(compute_hd95_mm=getattr(module,'compute_hd95_mm',True),compute_hd95_voxel=module.compute_hd95_voxel,metric_interface_sources=json.loads(ready.read_text()).get('sources',{}),selection_metric='mean Dice; unchanged',geometry_preflight=module.physical_preflight)
  manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2,default=str))
 def _evaluate(self,trainer,module):
  if getattr(module,'compute_hd95_mm',True):return super()._evaluate(trainer,module)
  namespace=dict(SynapseTestSelection._evaluate.__globals__);namespace['summarize_cases']=summary_dice
  evaluate=FunctionType(SynapseTestSelection._evaluate.__code__,namespace)
  results,summary=evaluate(self,trainer,module)
  self._write_named(results,summary,trainer.current_epoch+1)
  return results,summary
 def on_train_epoch_end(self,trainer,module):
  if getattr(module,'compute_hd95_mm',True):return super().on_train_epoch_end(trainer,module)
  epoch=trainer.current_epoch+1
  trainer.save_checkpoint(str(self.run_dir/'last.ckpt'),weights_only=True)
  if not should_evaluate(epoch,trainer.max_epochs,self.last_n_epochs):return
  results,summary=self._evaluate(trainer,module)
  improved=summary['mean_dice']>self.best_score
  if improved:
   trainer.save_checkpoint(str(self.run_dir/'best_test.ckpt'),weights_only=True);self.best_score=summary['mean_dice'];self.best_epoch=epoch
  common={'epoch':epoch,'global_step':trainer.global_step}
  append_csv(self.run_dir/'test_epoch_metrics.csv',[dict(common,**summary,is_best=int(improved),best_epoch=self.best_epoch,best_mean_dice=self.best_score)])
  case_rows=[]
  for x in results:
   row=dict(common,case=x['case'],mean_dice=x['mean_dice'],mean_hd95=None)
   for name in ('dice','hd95'):row.update({name+'_class_'+str(i):v for i,v in enumerate(x[name+'_per_class'],1)})
   case_rows.append(row)
  append_csv(self.run_dir/'test_case_metrics.csv',case_rows)
  report=dict(selection_policy='test set used for selection; unchanged legacy Dice',epoch_numbering='one-based',max_epochs=trainer.max_epochs,first_evaluation_epoch=trainer.max_epochs-self.last_n_epochs+1,last_evaluated_epoch=epoch,expected_test_cases=self.expected_test_cases,best_epoch=self.best_epoch,best_mean_dice=self.best_score,best_checkpoint='best_test.ckpt',last_checkpoint='last.ckpt',hd95_mm_enabled=False,hd95_voxel_enabled=False)
  (self.run_dir/'selection_summary.json').write_text(json.dumps(report,indent=2))
  if trainer.logger is not None:trainer.logger.log_metrics({'selection/test_'+k:v for k,v in summary.items() if k=='mean_dice' or k.startswith('dice_class_')},step=trainer.global_step)
  logging.info('[DICE SELECTION] epoch=%d Dice=%.6f best_epoch=%d best=%.6f HD95=disabled',epoch,summary['mean_dice'],self.best_epoch,self.best_score)
 def on_test_epoch_end(self,trainer,module):
  if getattr(module,'compute_hd95_mm',True):
   super().on_test_epoch_end(trainer,module)
   p=self.run_dir/'physical_test_summary.json';d=json.loads(p.read_text());d['hd95_mm_enabled']=True;p.write_text(json.dumps(d,indent=2));return
  summary=summary_dice(self._test_results,self.expected_test_cases,module.num_classes-1)
  self._write_named(self._test_results,summary,self._test_checkpoint_epoch)
  report=dict(hd95_unit='mm',mean_dice=summary['mean_dice'],mean_hd95_mm=None,mean_hd95_voxel=None,hd95_mm_enabled=False,hd95_voxel_enabled=False,num_cases=len(self._test_results),**self._test_checkpoint_info)
  (self.run_dir/'physical_test_summary.json').write_text(json.dumps(report,indent=2))
