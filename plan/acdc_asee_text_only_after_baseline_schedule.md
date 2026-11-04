# ACDC仅文本ASEE：baseline repeat_2后的单次任务

任务ID：acdc-2-45-asee。基准2026-11-02 19:59:01 MST，目标2026-11-02 22:44:01 MST（2026-11-03 05:44:01 UTC），延迟2小时45分钟，单次执行。新授权取代此前本轮不自动接续ASEE的阶段性限制，但不恢复已删除的旧任务。

目前只设置任务，未启动ASEE训练、未创建等待进程。当前baseline保持运行。配置已静态准备，本分支CPU/GPU及隔离启动适配由定时任务按以下合同完成，不冒称已通过。

## 执行合同

用户授权的一次性ACDC仅文本ASEE正式训练。基准2026-11-02 19:59:01 America/Denver（2026-11-03 02:59:01 UTC），目标2小时45分钟后：2026-11-02 22:44:01 MST（2026-11-03 05:44:01 UTC）。

先读ACDC项目AGENTS.md、.codex/acdc_asee_text_only_after_baseline_schedule.md和.codex/acdc_baseline_repeat_2_training_run.md。SSH anhuiuniversity.icu；项目/Disk/HDD/Sk/EoSeg-main_Synapse_acdc/EoSeg-main_acdc；Python=/Disk/HDD/Sk/miniconda3/envs/DeepLearning_xiakun/bin/python。不得误用同级Synapse，不把本地PowerShell/SSHFS当服务器执行。所有Python保留HF_HUB_OFFLINE=1、TORCH_HOME=/Disk/HDD/Sk/.cache/torch、TIMM_USE_OLD_CACHE=1；V2训练还保留CUBLAS_WORKSPACE_CONFIG=:4096:8、PYTHONHASHSEED=0、OMP_NUM_THREADS=1，GPU不得继承CPU验证的空CUDA_VISIBLE_DEVICES。

当前唯一前序为baseline repeat_2：outputs/baseline/repeat_2/20261103T025120Z（启动时training PID46680仅参考，现场核验PID/start_ticks/cwd/cmdline）。先检查status、training_exit、run_manifest、best/last、selection_summary、400轮history/16000体积行、last内部epoch399/global_step37600、完整40体积/轮及退出0。只在该run完整成功后做本项GPU验证及正式训练；前序正常运行不得停止或并发抢跑。前序未完可建立唯一、一次性幂等服务器等待监督器，待自然成功且资源足够后顺延；排队不称已训练。不因任务触发时间到点停止训练。首轮outputs/baseline/20261102T123802Z的历史last251缺失问题已记录，不伪造/重训，也不在当前repeat_2完整成功后继续误用它阻挡本项。前序失败/证据不齐保留并报告，不跳过、不自动失败重训。

本项独立根outputs/ablations/asee_text_only_v2，配置preparation/experiment.yaml已静态准备。先核对launch_receipt.json、waiting_receipt.json、waiting_status/status、实际进程、独立launch.lock及共享outputs/baseline/launch.lock；已有健康等待/验证/运行/完成只核验，不重复登记或启动。此前旧acdc-12-asee任务只记录前序阻碍、未正式训练；不能把旧scheduled_check.json误认正式run或重试失败训练的授权。本次是用户重新授权的一次启动，不恢复其他旧队列。

使用scripts/run_acdc_deterministic_v2.py fit，protocol=acdc_ce2d_stableseg_execution_v2、profile=asee、ROI=true、deterministic_enabled=true。实际模型EoSegMultiQSegFusionASEE，训练类ACDCMaskClassificationSemantic；use_text=true、use_probability=false、enhance_patch=true、enhance_query=true、patch_update=plan、freeze_text_projection=false、queries_per_class=2、num_q=6、num_classes=4，prior=priors/acdc_asee_80_20_v1.pt；CCPS关闭且不实例化，不套CCPS专用局部BN/backward_pass_autocast控制。只关闭共现分支，不改先验资产，保留原可学习query。

固定seed0、400轮/37600step、LR5e-4、224/patch16、4类/6query、batch16/workers8，固定80训练患者1512切片/20人40体积、原ACDC随机旋转/翻转/resize/归一化、AdamW/WD.05/LLRD.8/原warmup-poly/原损失采样及系数/clip.01/FP16/default outer compile。每次从同一原DINO预训练重新初始化，不加载baseline或其他分割best/last，不继承低LR100轮或短测Trace配置。正式probe_batches=0，无停止短测回调。每轮完整40体积1-view Dice选best，使用已修复training.acdc_checkpoint.ACDCModelCheckpoint每轮维护last，日志每轮有[EPOCH-BEST]；两HD95关闭、距离CSV空/JSON null，不自动TTA/HD95。按verified ACDC标签1=RV右心室腔、2=MYO心肌、3=LV左心室腔展示Dice及平均，无背景。

核验ACDC readiness passed/profile及当前源码/配置/80-20名单/先验/预训练哈希。现有AS EE完整双先验短测不能冒充本项文本单分支通过。需要本分支实际CPU入口前后向验证：文本、patch/query先验参与计算/梯度，共现projection无调用无梯度，改变禁用共现输入不改输出，CCPS不存在。然后在前序完整成功后做真实默认compile/FP16/batch16/workers8有界GPU验证（至少2次真实AdamW更新，不能把GradScaler跳步计入），完整40体积/120器官Dice-only，ROI记录true、HD95零调用，退出0后更新本项readiness并从原预训练正式400轮。

如果隔离启动器/Startup/分支探针/等待适配尚未准备，应在本项preparation内补齐必要实现与测试，复用ACDC已验证实现，不因缺启动器直接放弃；不得改前序冻结源码/config、repeat_runtime.py、原模型/损失/增强或科学变量。不能套只断言纯baseline的Startup；ASEE启动必须真实断言本项开关和criterion/profile/seed/ROI/HD95。新增适配/输出不得覆盖历史；已有失败GPU冒烟或正式run不擅自重训。

启动GPU验证/正式训练时现场剩余显存>=22000MiB、磁盘>=30GiB。允许其他用户并发，资源不足等待/报告，不停止他人、不减batch/输入/精度或关compile。保存命令、解析配置、环境、源码/资产hash、所有初始state含_extra_state、Python/NumPy/Torch/CUDA RNG、实际优化器分组，与自身冒烟初始化比较。实际确认PL_GLOBAL_SEED=0、PL_SEED_WORKERS=1、8worker派生播种、严格deterministic=True/warn_only=False、cuDNN deterministic=True/benchmark=False。启动后核验真实PID/GPU/startup及epoch/step推进，不能把回执当训练。

完成仅指训练退出0、last epoch399/global_step37600、400轮history/16000病例行及best/last齐全。汇报真实启动/等待/失败及文件变化；不承诺同seed完整400轮已逐位可复现。仅本项ASEE文本一次，不启动CCPS、多seed、其他消融、TTA/HD95或长期监控。落实唯一启动或等待后删除/暂停本单次任务，不停止正常服务器训练或已授权一次性等待进程。


## 本次单次任务已落实：服务器一次性等待

核验时间2026-11-03T06:07:06Z。baseline repeat_2仍健康训练，已保存第354轮/33276step，本次没有停止/重启或修改它。ASEE尚未开始GPU验证或正式训练。

新增preparation/cpu_probe.py和asee_runtime.py。实际ACDC V2 CPU入口（只对CPU接口用tiny backbone）前后向已通过：文本、patch、query分支调用且有非零有限梯度，共现投影0调用/无梯度，修改禁用共现矩阵输出逐值不变，CCPS不存在。配置、前序PID/缺轮拒绝、启动前显存/磁盘门禁均检查通过。CPU首次mock签名失败已在隔离测试中修正并保留日志，不是GPU/正式训练失败。

唯一等待监督器PID=365888（start_ticks=1305043），回执outputs/ablations/asee_text_only_v2/waiting_receipt.json。只等待当前baseline完整成功后，串行做本变体默认compile/FP16/batch16/workers8八batch及完整40体积GPU验收、HD95零调用检查，通过才从原预训练正式400轮；formal probe=0。独立readiness目前仅CPU通过、GPU待执行，不冒充已启动。前序、GPU或正式run失败不自动重训。服务器等待进程不依赖本机和本聊天继续运行。

准备期间归档替换了一次尚未触发GPU的闲置等待器，以补齐冒烟之后正式训练前的第二次资源检查；日志/回执留在preparation/scheduling_audit中，没有重复正式训练、没有触碰baseline或其他用户。

本应用单次任务acdc-2-45-asee处理后删除，服务器唯一等待器继续；不另建应用定时或长期监控。后续真实阶段以waiting_status.json、launch_receipt.json及run status为准。新增/修改文件清单见preparation/file_changes.json。


## 后继已推进为正式训练

baseline完整结束后文本分支GPU通过；第一次正式进程在任何训练步之前被tuple/list元数据比较断言误拦截，原失败保留。本轮用户明确授权继续，隔离比较修复及3项回归后，新run outputs/ablations/asee_text_only_v2/20261103T064255Z 的Startup通过，初始化/RNG/优化器与冒烟逐值一致，PID=433734。真实阶段以根launch_receipt及本run status/progress为准；不再描述为尚未启动。详见acdc_asee_text_only_training_run.md。
