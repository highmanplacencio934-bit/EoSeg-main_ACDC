# ACDC完整仅文本ASEE+CCPS正式训练

输出：outputs/experiments/asee_text_ccps_full_v2/20261103T134235Z。启动核验training PID=1134380，supervisor PID=1127217，现场已完成5轮/470step，loss有限；不是排队状态。PID后续须重新核验start_ticks/cwd/cmdline。

前序ASEE仅文本完整400轮、退出0、last399/37600及400轮/16000体积行已核验通过，best126平均Dice89.349079%，RV右心室腔88.594341%、MYO心肌83.569521%、LV左心室腔95.883369%，两HD95未计算。权重/数据/结果保留。

## 配置及实际Startup

EoSegMultiQSegFusionASEECCPS/ACDCMaskClassificationSemantic，ASEE text=true、probability=false、patch=query=true、plan、文本投影可训练、4类/6query。CCPS采用完整visual_prior主配置：patch门控开、query门控关、hidden256、init.95、detach_prior=false、patch_gate_effect=both（上下文重加权及残差调节均开），不新增辅助损失或其他变体。

入口scripts/run_acdc_deterministic_v2.py/profile=ccps；局部BN256编译边界已实际生效，backward_pass_autocast=off，网络整体默认compile/16-mixed保持。seed0、8worker派生播种、strict=true/warn_only=false、cuDNN deterministic=true/benchmark=false、ROI开，两HD95关、无TTA。400轮/37600step、LR5e-4、224/patch16、batch16/workers8、原ACDC增强/归一化/80-20/AdamW/LLRD/warmup-poly/损失/clip.01不变。

从原DINO预训练重新初始化，无分割checkpoint加载；实际415优化器组。正式probe_batches=0、无Trace停止回调。完整初始化state含_extra_state、RNG和优化器与本变体自己的已通过GPU冒烟严格一致（JSON统一表示，未放宽数值）。每轮完整40体积1-view Dice选best并滚动last，输出[EPOCH-BEST]。

## 验收与边界

CPU真实ccps CLI前后向接口和both公式核验通过：text/patch/query/selector参与且梯度非零有限，probability投影零调用/无梯度，禁用共现矩阵变化输出不变。CPU使用embed1024/depth3的接口骨干保留真实BN256边界；最初tiny32未满足该保护的失败日志归档，未修改生产门禁或模型。

本项GPU默认compile/FP16/batch16/workers8，8个原生batch、6次真实AdamW更新、40体积/120类Dice、ROI=true、HD95零调用及best/last/日志通过。CPU小模型或其他分支未冒充本项GPU通过。

只有完整400轮/37600step、400history/16000体积行及best/last齐全且退出0才记完成。本run仍在训练，尚无最终成绩。不自动启动seed42 baseline；由其另一个已授权单次任务按顺序核验。没有修改前序冻结依赖、生产模型/损失/数据增强或清理旧输出。
