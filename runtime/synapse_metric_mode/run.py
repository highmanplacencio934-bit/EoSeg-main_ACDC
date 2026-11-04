"""V2 training with opt-in millimetre HD95; original active entry remains intact."""
from pathlib import Path
import json,hashlib,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(Path(__file__).resolve().parent)]
from scripts.run_synapse_deterministic_v2 import SynapseExecutionV2CLI
from training.deterministic_execution_v2 import execution_context,restore_backend
from training.synapse_roi_evaluation import roi_metrics_scope
from main import LightningCLI
from training.lightning_module import LightningModule
from datasets.lightning_data_module import LightningDataModule
from lightning.pytorch.callbacks import ModelSummary,LearningRateMonitor
from metric_mode import metric_scope,MetricModeSelection

class MetricModeCLI(SynapseExecutionV2CLI):
 def add_arguments_to_parser(self,parser):
  super().add_arguments_to_parser(parser);parser.add_argument('--compute_hd95_mm',type=bool,default=False)
 def before_instantiate_classes(self):
  super().before_instantiate_classes()
  if self.config.get('subcommand')=='fit' and self._protocol_options()['compute_hd95_mm']:raise ValueError('HD95 is reserved for explicitly requested independent test; fit is Dice-only')
  ready=Path(__file__).with_name('readiness.json')
  d=json.loads(ready.read_text())
  if not d.get('passed'):raise RuntimeError('Dice-only interface readiness not passed')
  for name,h in d['sources'].items():
   if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=h:raise RuntimeError('Metric interface changed: '+name)
 def after_instantiate_classes(self):
  super().after_instantiate_classes()
  if not any(isinstance(cb,MetricModeSelection) for cb in self.trainer.callbacks):raise ValueError('Use MetricModeSelection for nullable HD95 reporting')
 def _run(self,stage,model,kwargs):
  o=self._protocol_options()
  try:
   with execution_context(o['deterministic_profile'],enabled=o['deterministic_enabled']),roi_metrics_scope(model),metric_scope(model,compute_hd95_mm=o['compute_hd95_mm']):
    return getattr(LightningCLI,stage)(self,model,**kwargs)
  finally:
   if hasattr(self,'_previous_backend'):restore_backend(self._previous_backend)
 def fit(self,model,**kwargs):
  if self._protocol_options()['roi_capture_dir'] is not None:raise ValueError('Fit requires original 1-view and no capture')
  return self._run('fit',model,kwargs)
 def test(self,model,**kwargs):return self._run('test',model,kwargs)

if __name__=='__main__':
 MetricModeCLI(LightningModule,LightningDataModule,subclass_mode_model=True,subclass_mode_data=True,save_config_callback=None,seed_everything_default=0,trainer_defaults={'precision':'16-mixed','devices':1,'callbacks':[ModelSummary(max_depth=3),LearningRateMonitor(logging_interval='epoch')],'gradient_clip_val':.01,'gradient_clip_algorithm':'norm'})
