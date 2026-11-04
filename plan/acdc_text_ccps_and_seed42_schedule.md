# ACDC仅文本完整CCPS与baseline seed42单次任务

|任务|基准Denver时间|目标Denver时间|任务ID|
|---|---|---|---|
|ASEE仅文本+完整CCPS，seed0|2026-11-03 05:31:51 MST|2026-11-03 06:31:51 MST|acdc-1-asee-ccps|
|纯baseline，seed42|2026-11-03 05:34:42 MST|2026-11-03 13:34:42 MST|acdc-8-baseline-seed42|

两条均ACTIVE、单次，不替换或取消对方。严格当前ASEE文本run→本完整CCPS→seed42 baseline串行；不停止前序，未完允许唯一一次性服务器等待。这里只创建任务/静态配置，尚未做对应CPU/GPU验收，未启动训练或服务器等待器。

## 用户已确认的语义

- 用户将8小时任务的seed从最初写的0明确纠正为42，用来与seed0比较；Python/NumPy/Torch及worker派生随机序列随42变化，解释器PYTHONHASHSEED仍按现有V2协议固定0，不修改协议代码。
- CCPS完整/全部开启按既有完整主配置：visual_prior、patch门控开、query门控关、patch_gate_effect=both，上下文重加权和残差强度调节都开。ASEE patch/query均开，文本开、共现关；不额外新增query门控或presence辅助损失消融。
- 两项400轮/LR5e-4/224/patch16/4类6query/batch16workers8/原ACDC增强/80-20及原损失/优化器保持，ROI/严格确定性开，两HD95关、无TTA。均从原DINO预训练重新初始化，不加载分割checkpoint。
- 已存在的ASEE Startup JSON tuple/list比较问题已修复。新适配必须统一元数据序列化表示，保留真实state/RNG/优化器值的严格比较；seed42不能要求与seed0初始化相同，也不能套用hard-coded seed0断言。

## 第一项执行合同

用户授权的单次ACDC完整模型训练（ASEE仅文本+CCPS）。基准2026-11-03 05:31:51 America/Denver（12:31:51 UTC），目标1小时后06:31:51 MST（13:31:51 UTC）。先读ACDC AGENTS.md及.codex/acdc_text_ccps_and_seed42_schedule.md。SSH anhuiuniversity.icu，项目/Disk/HDD/Sk/EoSeg-main_Synapse_acdc/EoSeg-main_acdc，Python /Disk/HDD/Sk/miniconda3/envs/DeepLearning_xiakun/bin/python。每次Python带HF_HUB_OFFLINE=1、TORCH_HOME=/Disk/HDD/Sk/.cache/torch、TIMM_USE_OLD_CACHE=1；V2训练带CUBLAS_WORKSPACE_CONFIG=:4096:8、PYTHONHASHSEED=0、OMP_NUM_THREADS=1；GPU不继承空CUDA_VISIBLE_DEVICES。不得使用同级Synapse或把本地PowerShell/SSHFS当服务器。

严格当前ASEE文本run outputs/ablations/asee_text_only_v2/20261103T064255Z→本完整模型→已授权seed42 baseline串行。本条只落实本完整模型一次。核对前序status/exit/实际PID-start_ticks-cwd-cmdline、400轮history/16000体积行、best/last、last内部epoch399/global_step37600、每轮40体积完整且退出0；前序仍正常运行保持，不停止、不并发抢跑。可建立唯一服务器一次性幂等等待器，待自然成功/资源足够后顺延；等待不写成训练。前序失败或身份/证据不齐不跳过/重训。

独立根outputs/experiments/asee_text_ccps_full_v2，preparation/experiment.yaml已静态准备；launch_receipt/waiting_receipt、launch.lock和共享outputs/baseline/launch.lock防重复。已有健康等待/验证/运行/完成只核验，不重复。缺隔离适配时在本项preparation补齐必要CPU探针/Startup/监督器/等待器，不修改任何当前冻结依赖，不直接套ASEE或纯baseline的错误断言；保留所有历史失败，不自动覆盖正式失败训练。

模型EoSegMultiQSegFusionASEECCPS，训练类ACDCMaskClassificationSemantic，scripts/run_acdc_deterministic_v2.py fit、profile=ccps及acdc_ce2d_stableseg_execution_v2；必须保留CCPS V2局部BN边界和backward_pass_autocast=off。ASEE use_text=true/use_probability=false/enhance_patch=true/enhance_query=true、patch_update=plan、freeze_text_projection=false、queries_per_class=2、num_q=6、num_classes=4、固定priors/acdc_asee_80_20_v1.pt。CCPS采用当前完整主配置：selector_mode=visual_prior、selector_on_patch=true、selector_on_query=false、selector_hidden_dim=256、init=.95、detach_prior_for_selector=false、patch_gate_effect=both，上下文重加权和残差强度调节都开启；“完整/全部开启”依照既有完整主配置，不擅自新增query门控或辅助presence损失变体。必须实际核验selector参与梯度及both机制，不要把ASEE query增强与CCPS query门控混淆。

从同一原DINO预训练重新初始化，绝不从前序ASEE或任一分割best/last续训。seed0、400轮/37600step、LR5e-4、224/patch16/4类/6query、batch16/workers8、固定80训练患者1512切片/20人40体积、原ACDC增强和归一化、AdamW/WD.05/LLRD.8/warmup-poly/损失/clip.01/FP16/default compile保持。ROI/严格确定性开，两HD95关，逐轮完整40体积1-view Dice选best并滚动last，使用ACDCModelCheckpoint及[EPOCH-BEST]日志，无TTA。正式probe_batches=0，无Trace停止回调。按verified ACDC标签1RV/2MYO/3LV展示Dice，无背景。

核验ACDC ccps readiness passed及源码/数据/预训练/先验哈希，本项配置和预备代码冻结。仅文本CCPS须对应实际CPU入口前后向：概率投影无调用/无梯度，文本/patch/query及selector参与，禁用共现矩阵变化不改变输出，both机制和CCPS边界正确；CPU小模型覆盖仅作接口证明。前序成功后真实默认compile/FP16/batch16/workers8有界GPU（至少2次真实AdamW更新）、完整40体积120类Dice-only、ROI记录true、HD95零调用，退出0后才能正式400轮。不要拿仅ASEE或双先验CCPS的冒烟冒充本项。

每次GPU冒烟和正式启动前分别核验空闲显存>=22000MiB/磁盘>=30GiB，允许其他用户并发，资源不足等待，不停止他人、不降batch/尺寸/精度或关闭compile。保存全部初始state含_extra_state、Python/NumPy/Torch/CUDA RNG、实际优化器组、命令、配置、环境和哈希；JSON记录与内存tuple/list比较先统一序列化表示，不能松开真实值比较。优化器组按实际ACDC仅文本CCPS核验，不能照抄Synapse415组或ASEE401组。真实seed0、8worker派生播种、strict=true/warn_only=false、cuDNN deterministic=true/benchmark=false须在Startup验证。启动后核验实际PID/GPU及epoch/step推进。

仅训练退出0、400轮history/16000体积行、last399/37600及best完整才算完成。报告真实阶段和文件更改。本条只做本项，不提前启动8小时的baseline seed42、不新增其他任务或长期监控。落实唯一启动或等待后删除/暂停本单次任务，不停止正常训练。

## 第二项执行合同

用户授权的单次ACDC baseline seed42对照。基准2026-11-03 05:34:42 America/Denver（12:34:42 UTC），目标8小时后13:34:42 MST（20:34:42 UTC）。用户已明确纠正：此次seed=42，用于比较seed0，不再是seed0重复。先读ACDC AGENTS.md、.codex/acdc_text_ccps_and_seed42_schedule.md及.codex/acdc_baseline_repeat_2_training_run.md。SSH anhuiuniversity.icu；项目/Disk/HDD/Sk/EoSeg-main_Synapse_acdc/EoSeg-main_acdc；Python /Disk/HDD/Sk/miniconda3/envs/DeepLearning_xiakun/bin/python。每次Python带HF_HUB_OFFLINE=1、TORCH_HOME=/Disk/HDD/Sk/.cache/torch、TIMM_USE_OLD_CACHE=1；V2训练带CUBLAS_WORKSPACE_CONFIG=:4096:8、PYTHONHASHSEED=0、OMP_NUM_THREADS=1，不继承CPU验证的空CUDA_VISIBLE_DEVICES。不要改同级Synapse或把本地PowerShell当服务器。

严格当前ASEE仅文本→完整仅文本ASEE+CCPS（outputs/experiments/asee_text_ccps_full_v2）→本baseline seed42串行。本条只落实seed42一次。核对CCPS唯一回执、最新run、status/exit/真实PID-start_ticks-cmdline-cwd、last epoch399/global_step37600、400轮history/16000体积行及best/last，完整40体积/轮且退出0才算成功。前序正常未完保持，不停止或并发抢跑；允许本项唯一一次性幂等服务器等待器，待前序自然成功/资源足够后顺延，排队不称训练。若1小时任务因额度等未执行且CCPS尚无正式run/等待，按已授权顺序先唯一补齐/排队完整仅文本CCPS，再顺延本项，不跳过；已有正式失败/GPU失败不自动重训覆盖。

独立根outputs/baseline_seed42，preparation/experiment.yaml为静态配置；与seed0参照outputs/baseline/repeat_2/20261103T025120Z逐项核对。纯EoSegMultiQSegFusion，ASEE/CCPS均关闭且不实例化；scripts/run_acdc_deterministic_v2.py fit、profile=baseline，ROI/严格确定性开启。唯一科学变量改变：seed_everything=42，因此Python random/NumPy/Torch CPU及CUDA初始种子42，PL_GLOBAL_SEED=42、PL_SEED_WORKERS=1，8worker由42的固定随机序列派生。PYTHONHASHSEED仍固定0，作为与seed0相同的解释器哈希协议；现有configure_strict要求其为0，不改为42或改协议代码。CUBLAS/OMP/cuDNN等数值设置保持。

400轮/37600step、LR5e-4、224/patch16/4类/6query、batch16/workers8、相同原DINO预训练来源、固定80/20患者名单/1512切片/40体积、原ACDC增强/归一化、AdamW/WD.05/LLRD.8/原warmup-poly/损失/clip.01/16-mixed/default compile全部保持。从原DINO预训练重新初始化，不能加载seed0/ASEE/CCPS的分割best/last。两HD95关闭，逐轮完整40体积1-view Dice选best/last，ACDCModelCheckpoint滚动last及[EPOCH-BEST]日志，无TTA；正式probe_batches=0，无停止短测回调。不更换数据划分来模拟seed影响。

核验ACDC baseline readiness passed/profile/source及预训练/名单hash，独立launch/waiting回执、launch.lock和共享outputs/baseline/launch.lock防重复。准备不全应在本项preparation补齐隔离Startup/等待/监督与必要CPU/GPU核验，不能直接套repeat_runtime.Startup中硬编码seed0的断言，不能为了通过初始化检查将seed42对齐seed0。记录并实际核验42、worker_init函数、strict=true/warn_only=false、cuDNN deterministic=true/benchmark=false、纯baseline和HD95关闭。初始化Query/分割头等随机权重、RNG应随42变化，不能要求与seed0完全相同；预训练来源、科学超参及373优化器组应匹配。与本项自身seed42冒烟对照时保留完整state/RNG/优化器的严格值核验，并统一JSON tuple/list表示。

前序完整成功后做必要的真实默认compile/FP16/batch16/workers8有界seed42 GPU冒烟（至少2次实际AdamW更新、完整40体积/ROI true/HD95零调用），通过后从原预训练正式400轮；不能把seed0冒烟冒充42的Startup实测。每次GPU启动分别核验空闲显存>=22000MiB、磁盘>=30GiB；其他用户可并发但资源不足等待，不中止他人、不降batch/精度/尺寸或关compile。不修改当前前序冻结依赖/模型/损失/增强/原YAML，只新增独立适配及输出。

启动后核验真实PID/GPU及epoch/step，保存完整命令/环境/解析配置/sourcehash/初始化含_extra_state/RNG/373组优化器。退出0、last内部399/37600、400轮history/16000体积行和best/last才算完整；按verified label_id展示RV/MYO/LV及平均Dice，与seed0同口径比较百分点，不跨epoch按器官挑最好，不因两种seed一高一低就认定创新有效或失败。此任务只训练seed42纯baseline一次，不自动增加其他seed或TTA/HD95。落实唯一启动或等待后删除/暂停自身单次任务，不影响正常训练、不新增长期监控；报告真实阶段和新增/修改/删除文件。


## acdc-1-asee-ccps本次处理：独立适配已准备、GPU验证中

当前唯一前序ASEE文本run已完整400轮/37600step、退出0，400轮/16000体积及best/last齐全，last399/37600，已实际复核。未更改前序数据或权重。

完整CCPS的隔离preparation/cpu_probe.py及ccps_runtime.py已创建。真实ACDC ccps CLI的CPU前后向验证通过：text/patch/query和selector均调用且有非零有限梯度，prob_proj零调用/无梯度，改变禁用共现矩阵输出不变，both上下文重加权及残差强度公式逐值核对通过，局部BN256边界及AOT backward_pass_autocast=off生效。CPU初版tiny32不满足真实BN256门禁已保留失败日志，改为仅CPU测试的embed1024/depth3骨干保留该门禁后通过，没有改生产模型或科学设置。

唯一服务器监督器当前执行本变体GPU验收，run=outputs/experiments/asee_text_ccps_full_v2/preparation/gpu_validation/20261103T133846Z。Startup实际核验text=true/prob=false/patch=query=true、完整CCPS visual_prior/patchgate/both/init.95/querygate=false、seed0/ROI/strict及CCPS BN/AOT边界通过。默认FP16/compile/batch16/workers8八batch、完整40体积及HD95零调用尚在执行，不能提前当正式训练。通过后该唯一监督器才从原DINO预训练正式400轮，失败不自动重训。

应用本条单次任务处理后删除，不影响8小时baseline seed42任务，不停止当前GPU验证或后续已授权正式训练。真实阶段以根gpu_validation_receipt、launch_receipt及run status为准。


## 第一项已正式启动

outputs/experiments/asee_text_ccps_full_v2/20261103T134235Z，PID=1134380，核验已5轮/470step。本变体GPU已通过且初始化与冒烟一致。仅删除本条一次任务，seed42任务保留；实际状态以本根launch_receipt/run status为准。详见acdc_asee_text_ccps_full_training_run.md。


## 2026-11-03 baseline seed42 已正式启动

outputs/baseline_seed42/20261103T205219Z，本次核验训练 PID=1835139，第10轮训练快照/global_step=940。前序完整仅文本 CCPS 已400轮成功。seed42独立CPU、GPU八batch/6次真实更新/40体积及HD95零调用通过，正式初始state/RNG/373优化器组与自身冒烟严格一致；对照seed0的Query/分割头/RNG改变、预训练编码器与优化器分组不变。纯baseline，ASEE/CCPS不实例化；ROI和严格确定性开，两HD95关，无TTA。仅seed从0改42，400轮/LR5e-4/原增强/80-20等保持；PYTHONHASHSEED仍0，8worker按42派生。新隔离preparation及生产依赖冻结，详见.codex/acdc_baseline_seed42_training_run.md与正式启动证据。应用acdc-8-baseline-seed42已删除，后台训练继续；不自动启动其他实验或长期监控，400轮最终结果尚未得到。
