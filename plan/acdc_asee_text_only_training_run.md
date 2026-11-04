# ACDC仅文本ASEE正式训练

输出：outputs/ablations/asee_text_only_v2/20261103T064255Z；核验训练PID=433734、监督器PID=433414。进程身份须随状态重新核验。

baseline repeat_2已完整400轮/37600step、退出0，400轮/16000体积评估齐全、last内部epoch399及best齐全，前序完成验收通过。ASEE文本分支实际CPU前后向及默认GPU FP16/compile/batch16/workers8八batch/至少2次真实更新、40体积评估、ROI=true/HD95零调用均已通过。

## 正式配置

EoSegMultiQSegFusionASEE，use_text=true/use_probability=false，enhance_patch=true/enhance_query=true，patch_update=plan，文本投影可训练，4类/6query、固定训练集先验。CCPS不存在。使用ACDC V2/asee profile，ROI和严格确定性开启，warn_only=false、cuDNN deterministic=true/benchmark=false。真实seed0，8worker派生播种。

400轮/37600step、LR5e-4、224/patch16、batch16/workers8、原80/20名单和ACDC随机增强/归一化、AdamW/WD.05/LLRD.8/warmup-poly/原损失/clip.01/16-mixed/default compile保持。从原DINO预训练重新初始化，不加载分割权重；正式probe_batches=0，无短测Trace停止回调。每轮完整40体积1-view Dice选best，滚动last已修复，两HD95关、无TTA。实际401优化器组。Startup与自身已通过冒烟的初始state含_extra_state、RNG及优化器组严格一致。

## 启动适配修复与证据

此前20261103T063528Z在on_fit_start断言失败，0训练step，无history或checkpoint，原目录及退出1保留。原因是原比较用内存tuple(AdamW betas)对JSON list，误判初始化及分组不同。修复仅在比较边界使用与记录文件同一JSON表示，未放宽权重/RNG/优化器真实数值比较，未改模型、loss或训练参数。3项真实冒烟元数据回归：旧比较必失败，新比较通过，实际state/RNG及LR变更仍被拒绝。本轮新用户已明确授权继续；失效回执归档到startup_fix_audit，不覆盖失败目录，不从失败或冒烟权重续训。

原始场景已复验：本run的initialization_comparison.json两项true，startup_verification.json通过；见preparation/startup_fix_audit及test_startup_comparison.py。运行依赖及配置现冻结，不自动启动其他实验/多seed/TTA/HD95。完整400轮成绩尚未产生。
