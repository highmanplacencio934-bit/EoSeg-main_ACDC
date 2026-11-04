# ACDC 工作文档索引

> 2026-11-02 最新默认：用户已授权纯 ACDC baseline 正式训练。默认入口为 scripts/run_acdc_deterministic_v2.py fit，配置 configs/acdc/deterministic_v2/baseline.yaml，roi_metrics=true、deterministic_enabled=true、profile=baseline。ASEE/CCPS关闭，两HD95关闭，无TTA；原随机增强、80/20、400轮/37600step、224/patch16/4类/6query/LR5e-4/batch16/workers8保持。main.py及原YAML保留为显式原CUDA兼容路径，不是当前默认启动方案。开关仍可显式覆盖，不热切换正在训练的进程。


本目录只提供当前 ACDC 项目的工作约定、代码参数和真实验收状态。已逐份审查原65份Markdown，移除57份不适用的历史材料；清理时保留以下8份，当前另增acdc_baseline_training_run.md，不把其他数据集的成绩、定时任务或GPU通过状态改名为ACDC记录。

| 文档 | 用途 |
|---|---|
| [acdc_reproduction_baseline.md](acdc_reproduction_baseline.md) | 唯一baseline的模型、主干、LR、LLRD、优化器、数据、增强和评估设置 |
| [acdc_baseline_run_audit.md](acdc_baseline_run_audit.md) | 完整名单、严格加载、创新/V2/单视角ROI修复与真实CPU/GPU验收 |
| [random_result.md](random_result.md) | ACDC当前seed、8-worker与CUDA开关的实际情况及复现核查顺序 |
| [cuda_solution_plan.md](cuda_solution_plan.md) | ACDC独立V2实施、G0–G5短测通过与G6未执行的范围 |
| [hd95_mm_disabled.md](hd95_mm_disabled.md) | Dice-only规则及真实spacing缺失时的HD95边界 |
| [terminal_conventions.md](terminal_conventions.md) | 本项目服务器路径、Python及必要环境前缀 |
| [server_gpu_execution_rules.md](server_gpu_execution_rules.md) | 按剩余显存判断并发，不停止其他任务 |
| README.md | 本索引 |

当前baseline入口为main.py，配置configs/acdc/vit_query_mul_scale_fusion.yaml：400轮、224输入、patch16、4类、6query、基础LR5e-4、batch16/workers8。原CUDA，无V2/ROI/TTA/创新模块，两HD95关闭；每轮完整40测试体积按Dice选best/last。没有ACDC正式成绩，尚未授权长训。

用户2026-11-02最新决定采用80人训练、20人共用验证测试。训练patient021–100共1512切片，评估patient001–020共40体积；val.txt与test.txt逐行相同，实际fit/test均读test.txt，每轮按test Dice选best。batch16/drop_last下94步/轮，400轮37600步。三前景先验已按80位训练患者重建，77项CPU回归通过。

论文截图人数为70/10/20；当前80/20是用户选择，替代此前的预留10人及验证集选模安排。完整体积Dice、标签名与几何核验范围见baseline说明。

证据：
- ACDC适配：outputs/audit/acdc_baseline_adaptation/latest_path.txt。
- 划分与评估专项复核：outputs/audit/acdc_split_evaluation/latest_path.txt。
- 80/20当前划分与选模验收：outputs/audit/acdc_split_80_20/latest_path.txt。
- 较早用户评估确认/公开代码依据：outputs/audit/acdc_evaluation_confirmation/latest_path.txt，作为历史记录。
- 三前景先验/预训练适配：outputs/audit/acdc_prior_adaptation/latest_path.txt；[先验资产](../priors/README.md)与[通用预训练](../pretrained/README.md)。
- 本轮文档清理：outputs/audit/acdc_codex_cleanup/latest_path.txt；其中review和清理清单按文件给出理由，markdown_before_cleanup.tar.gz保存清理前原文。备份不属于当前执行依据。
- acdc_baseline_snapshot.json：当前参数、来源hash及分级验收状态。
- baseline_audit_snapshot.json、baseline_postfix_snapshot.json、baseline_test_interface_snapshot.json为复制的非Markdown旧快照，本轮未改动，不能作为ACDC的模型/数据/训练验收或启动门禁。

删除文档不等于删除同名源码或改变运行协议。本轮没有启动训练、创建定时任务或迁移其他项目的实验授权。

最新全部修复验收：outputs/audit/acdc_all_fixes/20261102T091215Z。128项当前CPU回归、原baseline默认GPU短测、V2三模型各两次冷/暖八batch及完整40体积通过。TTA实现与全部调用接口已删除，作者训练增强保留。新增配置和独立入口不自动改变main.py baseline；没有400轮训练成绩。

最新独立评估精度对齐：outputs/audit/acdc_fit_test_precision/20261102T111902Z。fit训练与选模不变；--compile_test的test及可视化复用相同指标编译上下文，40×3类、病例均值和总体汇总原值零差，ROI同样零差。128项CPU回归通过。详见acdc_baseline_run_audit.md末节。

当前正式训练见[acdc_baseline_training_run.md](acdc_baseline_training_run.md)。

后继两项单次训练安排见[acdc_baseline_repeat_schedule.md](acdc_baseline_repeat_schedule.md)。

- 当前第二次baseline正式训练及seed/worker/确定性核验：acdc_baseline_repeat_2_training_run.md。

- 当前仅文本ASEE单次后继任务：acdc_asee_text_only_after_baseline_schedule.md。

- 当前仅文本ASEE正式训练与Startup修复证据：acdc_asee_text_only_training_run.md。

- 11月3日后继完整仅文本CCPS和seed42 baseline：acdc_text_ccps_and_seed42_schedule.md。

- 当前完整仅文本ASEE+CCPS训练：acdc_asee_text_ccps_full_training_run.md。

- 当前 seed42 纯 baseline 正式对照、独立验收与证据：[acdc_baseline_seed42_training_run.md](acdc_baseline_seed42_training_run.md)。只改训练种子，其他科学设置与 seed0 repeat_2 一致；最终训练尚未完成。
