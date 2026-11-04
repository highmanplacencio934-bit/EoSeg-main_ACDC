# ACDC 第二轮baseline与仅文本ASEE单次调度

基准与目标均为America/Denver（MST，UTC-7），2026-11-02。

|任务|基准时间|目标时间|任务ID|
|---|---|---|---|
|第二轮相同配置纯baseline|05:49:05|11:49:05（18:49:05 UTC）|acdc-baseline-6|
|仅文本ASEE（patch/query均开，CCPS关）|05:51:20|17:51:20（11月3日00:51:20 UTC）|acdc-12-asee|

两个任务均为ACTIVE、COUNT=1的当前对话单次任务；不是周期复训。保留两者。

严格顺序：当前baseline outputs/baseline/20261102T123802Z → baseline repeat_2 → ASEE文本单先验。前序未完保持运行，后继允许在独立目录建立一次性幂等等待启动监督器，待前序完整成功后顺延；失败不自动重训或越过，排队不称已训练。不得停止正常训练。

## 第二轮baseline

从原预训练重新初始化，不能加载第一轮best/last。以第一轮config.yaml/launch.json/startup_verification和冻结源码/配置/资产hash为准。纯EoSeg，ASEE/CCPS关，ROI/确定性开；seed0、LR5e-4、400轮/37600step、224/patch16/4类/6query、batch16/workers8、原ACDC增强、原AdamW/LLRD/warmup-poly/损失/裁剪/FP16/default compile与80/20名单保持。每轮完整40体积Dice选择，HD95关、无TTA。

输出outputs/baseline/repeat_2/<UTCstamp>，唯一回执repeat_2/launch_receipt.json及latest_path.txt，不覆盖第一轮指针。只读复用outputs/baseline/preparation/baseline_runtime.py；改变的仅为日志、输出、Startup输出及独立缓存路径。使用共享outputs/baseline/launch.lock防重复。已有健康等待/运行/完成不得再次启动。

## ASEE仅文本

配置outputs/ablations/asee_text_only_v2/preparation/experiment.yaml已准备并静态核对，尚未做本分支CPU/GPU验收，不等于训练已准备通过。

EoSegMultiQSegFusionASEE / ACDCMaskClassificationSemantic，use_text=true、use_probability=false、enhance_patch=true、enhance_query=true、patch_update=plan、freeze_text_projection=false、queries_per_class=2,num_q=6；prior为priors/acdc_asee_80_20_v1.pt，CCPS不实例化。V2 profile=asee，ROI/确定性开，其他科学变量与ACDC400轮协议保持。仅文本是独立分支，不可用现有双先验ASEE短测冒充。

输出outputs/ablations/asee_text_only_v2/<UTCstamp>及唯一launch_receipt.json。隔离启动适配放自己的preparation，不能套仅检查baseline的Startup，不能修改正在执行前序的依赖。先完成本变体CPU前后向（文本/patch/query参与，共现不调用无梯度）、默认GPU FP16/compile/batch16/workers8有界短测及完整40体积Dice-only/ROI真实记录，再从同一原预训练重新开始正式400轮。正式probe_batches=0，无短测Trace。不继承任何分割权重或低LR100轮参数。

## 通用核验与服务器

SSH anhuiuniversity.icu；项目 /Disk/HDD/Sk/EoSeg-main_Synapse_acdc/EoSeg-main_acdc；Python /Disk/HDD/Sk/miniconda3/envs/DeepLearning_xiakun/bin/python。
每次保留HF_HUB_OFFLINE=1、TORCH_HOME=/Disk/HDD/Sk/.cache/torch、TIMM_USE_OLD_CACHE=1；V2另保留CUBLAS_WORKSPACE_CONFIG=:4096:8、PYTHONHASHSEED=0、OMP_NUM_THREADS=1。正式GPU不得带空CUDA_VISIBLE_DEVICES。

第一轮/第二轮完整成功需训练退出0、last epoch399/global_step37600、400行history/16000体积行和best/last。重新核验实际PID/start_ticks/cwd/cmdline/GPU；readiness passed且源码/配置/清单/先验/预训练hash匹配。资源门槛空闲显存22000MiB、磁盘30GiB，允许其他用户并发，不足等待，不停他人、不降batch/精度/尺寸或关闭compile。先验仅训练集，标签1 RV、2 MYO、3 LV；报告同checkpoint逐器官及平均Dice，无背景。

只落实这两项唯一启动/等待，不添加其他训练、消融、TTA/HD95或长期监控；执行后删除/暂停对应单次任务，正常服务器训练和已授权一次等待继续。


## 2026-11-02 18:49 UTC scheduled check: blocked

First training reached400/37600 and exited0 with complete400 history and16000 volume rows. However last.ckpt is the same epoch251/23594 as best (mean Dice90.874135%); no final epoch399/37600 state was saved. Installed Lightning2.6.6 only updates save_last after a top-k save in this configuration. Original failed_completion_validation is preserved. repeat_2 was not launched or queued; do not bypass this predecessor gate for ASEE. Evidence: outputs/baseline/repeat_2/preparation/predecessor_completion_check.json and scheduled_check_report.md. No training/model/metric/source changes or retry. The one-time acdc-baseline-6 automation is retired after this report; the later ASEE automation remains scheduled and must inspect this blocker.


## 2026-11-03 00:51 UTC ASEE scheduled check: same blocker

Rechecked first baseline: training_exit=0 but status remains failed_completion_validation; the final400 checkpoint is absent as previously reported. repeat_2 has no run/waiting receipt and no actual ACDC training or scheduled-wait process was found. The 6-hour task did execute; its failure to launch repeat_2 is a completion-gate blocker, not a missed trigger. No repeat_2/ASEE training or GPU validation was launched, no waiter queued, and no scientific settings/weights were modified. Only-text ASEE stays static-config-only, CPU/GPU branch validation pending. Record: outputs/ablations/asee_text_only_v2/preparation/scheduled_check.json. Retire acdc-12-asee after this single check; no further monitoring. Both one-time tasks have now been processed; the blocker requires an explicit decision before resuming the queue.


## 2026-11-02 最新：同配置baseline再次训练授权

用户明确要求以相同科学配置再次执行纯ACDC baseline并核验seed。本轮独立输出使用outputs/baseline/repeat_2；从同一原DINO预训练重新初始化，ASEE/CCPS关闭，ROI和CUDA严格确定性开启，seed0、400/37600、224/patch16/4类/6query/LR5e-4/batch16/workers8及原ACDC增强和80/20名单保持。原first run虽缺最终last权重，已完整400轮及全部评估；这次是用户重新授权的新独立训练，不伪造原last、不改原失败验收、不自动重训旧run。

- seed由LightningCLI实际调用seed_everything(0, workers=True)，Python/NumPy/Torch CPU及CUDA播种；8个worker分别由pl_worker_init_function派生种子，不能把8个worker全部设成相同的0。
- 启动前环境PYTHONHASHSEED=0、CUBLAS_WORKSPACE_CONFIG=:4096:8、OMP_NUM_THREADS=1，严格deterministic=True/warn_only=False，cuDNN deterministic=True/benchmark=False。固定seed不等于完整400轮已实证逐位一致；复训目的仍包括核验这一点。
- 后续ACDC统一使用training.acdc_checkpoint.ACDCModelCheckpoint，只修复非best轮的last保存，best最高Dice规则不变。每轮另输出[EPOCH-BEST] epoch/best_epoch/best_dice百分数；原训练模型、损失、增强、精度及Dice公式未改。旧日志/权重不改，历史第400轮权重不能补回。
- 前次保存/日志修复37项CPU测试、真实八batch/六次更新/40体积GPU功能通过；附加旧回调对照曾退出-9，随后服务器重启。当前授权下正在补齐同环境旧/新回调对照，未通过不能仅刷新hash放行。
- 本轮只启动baseline一次；此前两个单次任务已处理删除，不恢复ASEE队列、不另建定时任务。实际PID/启动/门禁以outputs/baseline/repeat_2的独立记录为准。


### 新用户授权已落实为正式repeat_2

outputs/baseline/repeat_2/20261103T025120Z 已实际启动，seed0/初始化/RNG/优化器一致，保存回调修复已实测。旧单次任务不重建，首轮历史last缺失不改写。以.codex/acdc_baseline_repeat_2_training_run.md及本run状态为准；没有排队ASEE。


### 当前唯一后继任务：acdc-2-45-asee

新用户授权按2026-11-02 19:59:01 MST起2小时45分钟，22:44:01 MST检查并落实仅文本ASEE。当前唯一前序为正在运行的outputs/baseline/repeat_2/20261103T025120Z；前序未完不停止、不抢跑，允许本项唯一等待。原始首轮last缺失仅历史，不在repeat_2完整成功后重复误用旧门槛。旧单次任务不恢复；最新合同见acdc_asee_text_only_after_baseline_schedule.md。
