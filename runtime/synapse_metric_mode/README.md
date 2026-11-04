# Synapse：训练Dice-only，HD95仅显式独立测试开启

更新：2026-10-29。用户最新要求取消后50轮毫米HD95计算，不删除任何HD95实现。此约定替代未来训练默认mm-only或mm+voxel的历史要求。

## 当前行为与范围

后续新启动训练默认compute_hd95_mm=false，且model.init_args.compute_hd95_voxel=false。第171–220轮完整12病例只计算Dice、按mean Dice选择best，滚动保存last；模型、损失、优化器、增强、划分、精度和编译不变。不得在fit开启HD95；新入口对fit --compute_hd95_mm=true报错，确保距离只用于明确授权的独立test。

独立test默认也只计算Dice。只有用户明确要求HD95时显式--compute_hd95_mm=true；体素仍关闭，如需恢复另显式--model.init_args.compute_hd95_voxel=true。Dice-only不允许同时打开voxel。关闭的HD95在CSV留空、JSON为null，metric_units和run_manifest明确记录hd95_mm_enabled=false，不记为0。原预测函数内部仍需数字型二元组，适配层仅为保持原代码流程传递临时占位数，公开结果在聚合、日志、选模与落盘前全部置为None；没有HD95函数调用，且Dice聚合完全不使用距离。

## 后续入口与配置

为了不修改当前运行baseline-repair_2的冻结依赖，功能独立放在runtime/synapse_metric_mode。原main/ROI/V2、原损失/指标/模型和原YAML不改；历史结果、checkpoint和旧readiness保持。新入口继承已验收的V2标准流程，复用其确定性criterion、CCPS局部边界、execution_context和ROI推理路径，新增的只是指标开关和nullable报告回调。

- 入口：runtime/synapse_metric_mode/run.py。
- baseline：runtime/synapse_metric_mode/configs/baseline.yaml。
- ASEE双先验：runtime/synapse_metric_mode/configs/asee.yaml。
- ASEE+CCPS：runtime/synapse_metric_mode/configs/ccps.yaml。
- 回调：metric_mode.MetricModeSelection。必须由新入口导入；不要给旧入口直接使用带compute_hd95_mm的配置。
- 新接口门禁：runtime/synapse_metric_mode/readiness.json；同时保留原V2 G0–G5/profile/source门禁，不绕过。

三份配置的模型/seed/原增强/LR2e-4/220轮/LLRD/AdamW/原损失/FP16/默认compile/原18-12划分与对应deterministic_v2配置一致，仅指标回调、HD95开关及输出路径变化。已存在的低LR/100轮微调不热切换。未自动启动任何新训练。

## 命令

所有命令在服务器项目根目录执行，Python固定/Disk/HDD/Sk/miniconda3/envs/DeepLearning_xiakun/bin/python。每次保留HF_HUB_OFFLINE=1、TORCH_HOME=/Disk/HDD/Sk/.cache/torch、TIMM_USE_OLD_CACHE=1。下面以baseline示例；ASEE/CCPS替换对应yaml。

```bash
cd /Disk/HDD/Sk/EoSeg-main_Synapse_acdc/EoSeg-main_synapse
HF_HUB_OFFLINE=1 TORCH_HOME=/Disk/HDD/Sk/.cache/torch TIMM_USE_OLD_CACHE=1 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 OMP_NUM_THREADS=1 \
/Disk/HDD/Sk/miniconda3/envs/DeepLearning_xiakun/bin/python runtime/synapse_metric_mode/run.py fit \
  --config runtime/synapse_metric_mode/configs/baseline.yaml
```

下述是仅在用户要求独立毫米HD95测试时使用的接口，不表示本轮已执行：

```bash
HF_HUB_OFFLINE=1 TORCH_HOME=/Disk/HDD/Sk/.cache/torch TIMM_USE_OLD_CACHE=1 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 OMP_NUM_THREADS=1 \
/Disk/HDD/Sk/miniconda3/envs/DeepLearning_xiakun/bin/python runtime/synapse_metric_mode/run.py test \
  --config runtime/synapse_metric_mode/configs/baseline.yaml \
  --ckpt_path=/ABSOLUTE/PATH/TO/best_test.ckpt --compile_test \
  --compute_hd95_mm=true --trainer.logger=false
```


## 运行保护和启动器注意

当前baseline-repair_2/G6_2/20261029T022332Z仍按其启动时原mm-only评估口径运行；不停止、重启或中途改口径，也不改旧CSV。要对同一次完整G6严格对照，不能在中途更换指标接口。新功能只用于后续新run。源码新增在独立runtime目录，不进入旧V2训练代码扫描集合；旧readiness对baseline/ASEE/CCPS均仍通过。

旧G6、旧消融启动器的末端验收要求HD95数字，是历史mm模式的规则。不得将新Dice-only配置与这些旧启动器直接混用；后续独立启动适配必须使用新入口/回调，根据metric_units.hd95_mm_enabled验收空值，并以Dice-only聚合和50轮/600病例/4800器官行完整性验收。不得为了过旧验收伪造0。运行中的旧G6监督器不改。

## 验证证据

7项CPU回归覆盖：零HD95调用、legacy空掩码、1/2/4/6/8视角Dice不变、显式开启真实voxelspacing距离、异常恢复、完整CPU Lightning test日志/JSON/CSV空值、最后50轮/严格best更新/last。

使用既有缓存的完整12病例预测与已保存原指标对照，96项Dice和空掩码status逐值相同，geometry逐病例核验，HD95调用0。没有重新GPU推理或启动训练；不能将此声称为新的完整训练结果或整场训练测速。

三套新配置CLI解析通过；对应模型参数/seed/LR/预算保持。报告见outputs/audit/hd95_mm_disabled/20261029T071421Z/validation.json，cached_12_case_dice_regression.json及cpu_tests.log。真实GPU资源被当前训练占用，本轮没有新增GPU任务。
