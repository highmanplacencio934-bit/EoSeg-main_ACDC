# ACDC baseline与适配审计

> 2026-11-02 最新默认：用户已授权纯 ACDC baseline 正式训练。默认入口为 scripts/run_acdc_deterministic_v2.py fit，配置 configs/acdc/deterministic_v2/baseline.yaml，roi_metrics=true、deterministic_enabled=true、profile=baseline。ASEE/CCPS关闭，两HD95关闭，无TTA；原随机增强、80/20、400轮/37600step、224/patch16/4类/6query/LR5e-4/batch16/workers8保持。main.py及原YAML保留为显式原CUDA兼容路径，不是当前默认启动方案。开关仍可显式覆盖，不热切换正在训练的进程。


更新：2026-11-02 America/Denver。当前80训练患者1512切片、20评估患者40体积，val/test共用；原baseline为400轮/224/patch16/4类/6query/LR5e-4/batch16/workers8。每轮完整40体积Dice选best/last，两HD95关闭，无TTA。没有正式训练成绩。

## 1. 五项审计修复

| 问题 | 修复 | 验证 |
|---|---|---|
| 少一帧但患者数正确仍可通过 | datasets/acdc_splits.py要求1512/40/40及每患者两帧，训练切片索引连续 | 缺帧/缺片CPU回归拒绝 |
| 可视化strict=False、原FP32/eager路径不同 | 全state严格加载；共用原体积推理，默认FP16/compile | 缺query拒绝；完整状态CPU一致；同权重GPU单病例三类Dice精确一致 |
| 缺ACDC创新配置、旧CCPS依赖Synapse spacing | 独立ASEE/CCPS YAML，使用ACDCMaskClassificationSemantic，presence=0 | CPU前后向与三模型实际GPU入口通过 |
| 原V2门禁仍面向Synapse | ACDC独立entry/profile/source/readiness | G0–G5短测通过三模型；完整400轮重复未做 |
| ROI/TTA绑定旧物理包装 | ACDC单视角Dice-only ROI；按用户最新要求删除TTA实现与全部调用参数 | 同权重40×3 ROI/原Dice逐值相同；活动源码/YAML无TTA引用 |

root build_prior.py也已改为真实ACDC先验构建入口，旧dummy示例存于本轮审计；未覆盖现有先验、预训练或原始参考文件归档。

## 2. 当前回归与GPU证据

- CPU 128项通过：ACDC数据/完整名单、增强回放、CLI小模型、严格加载、三前景先验、创新机制、单视角ROI、训练安全和指标开关。
- 原CUDA真实ViT-L：原科学400轮预算下八batch有界短测，batch16/workers8、224、FP16/default compile；每轮94步、总37600步保持。完整40体积评估、best/last和独立test均退出0。
- 同一短测best：ROI与非ROI的120项Dice、严格可视化一个测试体积三类Dice原值精确相同。fit与独立test原值有最大1.8612e-9微差，120项分器官值经过同样float32表示转换后逐值相同；保存预测CPU最小复现已重现compiled指标边界，checkpoint全部状态一致，未改生产评估或放宽容差。短测权重不是论文成绩。
- V2：baseline、ASEE、ASEE+CCPS各两个独立进程，八个原生batch；初始化state/RNG/优化器分组、输入、loss、scaled/unscaled梯度、每步state和AdamW moments精确一致，覆盖冷/暖编译缓存；每次6次真实AdamW更新和完整40体积评估。
- ACDC CUDA算子：CE/点采样各30次重复及真实4类criterion三次重放均一致。
- 上述V2短测最高reserved约10370MiB；不是完整400轮峰值保证。启动仍预留临时开销，不干扰其他用户。

证据根：outputs/audit/acdc_all_fixes/20261102T091215Z。当前readiness：outputs/determinism/acdc_ce2d_stableseg_execution_v2/readiness.json，精确来源hash门禁；CPU、算子和短轨迹范围明确，不代表G6通过。

## 3. 保留的失败与范围

首次算子诊断把CPU标签直接交给GPU loss，测试脚手架设备断言失败；只修复探针的device transfer后重跑，生产损失未变。第一次CPU扩展检查对YAML LR字符串直接比较float失败，修正测试解析后通过。
TTA移除后扩大运行的历史Synapse集成测试有两项因本副本没有Synapse标签表/外部教师utils.py失败，原测试和日志保留；它们不作为ACDC验收。已运行其中不依赖外部资产的ROI开关测试。另有一处对已删除发布函数的mock已随TTA接口一并移除。原入口GPU各阶段均退出0，但最初末尾“fit/test原值逐位一致”的审计断言失败；已用保存预测确认fit编译指标与独立test之间float32表示差异，原失败日志保留，生产评分未改。

## 4. 保护与未完成项

原main.py、baseline YAML、datasets/acdc_dataset.py原训练增强、数据清单、模型、原损失、预训练和先验均按修改前hash保护。TTA专属8文件和对应字节码删除；混合文件只移除TTA阶段/参数，单视角训练、ROI与HD95算法保留。原副本及删除清单存本轮审计；同级Synapse未修改。

未启动正式400轮训练、定时任务或G6完整重复；真实NIfTI几何和HD95_mm未核验。正式训练完成标准：退出0，last epoch399/global_step37600，400轮history、16000体积行（48000项前景Dice）、best/last及来源清单完整。训练另待用户指令。

旧80/20数据与先验证据：outputs/audit/acdc_split_80_20/20261102T082608Z；旧70/10/20材料只作历史，不覆盖当前80/20或将其成绩改名为本项目成绩。

## 2026-11-02 fit/test精度对齐已修复

- 原因：fit编译validation_step，独立test原来仅编译forward；NumPy指标在不同执行上下文中出现float32/float64差异。独立汇总也曾保留float64，而fit日志使用Lightning标量类型。
- 当前main.py test --compile_test同时编译ACDC forward及同一个_compute_case_metrics。只在NumPy指标调用内部使用正常CPU dispatch，退出即恢复inference_mode；GPU前向继续FP16/原compile。可视化共用同一指标函数，删除重复公式。
- 独立test汇总采用与fit相同的逐病例均值和Lightning默认浮点标量表示。原fit、validation_step、on_validation_epoch_end、on_validation_end及best选择代码未改；不是修改训练分数来迎合test。
- 真实RTX5090D同一只读短测checkpoint，完整40体积：120项分器官Dice、40项病例均值、总体均值及三类汇总原值精确一致；ROI路径同样一致，可视化1体积三类及均值一致；容差0。源checkpoint SHA不变。
- 128项CPU回归通过；旧test调用位置在同一新回归中失败，新实现通过。覆盖inference_mode恢复、模型方法恢复和fit/test汇总类型。
- 新评估复现默认compile训练时须显式--compile_test；eager训练权重仍用默认eager test。该结果限定当前相同权重、精度和编译协议，不声称跨模式或跨环境逐位相同。
- 两HD95继续关闭，无TTA；数据划分、作者随机增强、模型、损失、seed、优化器和预算不变。没有正式训练或重训，原权重与历史CSV保留。
- 本次变更只涉及独立评估。已逐AST核验fit相关代码不变、逐hash核验其余训练依赖/配置/清单/先验/预训练不变；原G0–G5训练证据保留，新增CPU与GPU独立评估证据纳入readiness，未冒称本轮重新执行三模型训练轨迹。
- 证据：outputs/audit/acdc_fit_test_precision/20261102T111902Z/gpu_retry2/comparison.json、training_preservation.json、cpu_final_exit.json及file_changes.json。


## ACDC baseline正式运行记录

输出：outputs/baseline/20261102T123802Z。训练PID以status.json实时身份核验；启动时2740388，监督器2740196。已核验cuda:0、纯EoSeg/ASEE关/CCPS关、ROI开/严格确定性开、400轮/37600step、LR5e-4、batch16/workers8、每轮完整40体积Dice、两HD95关。33项ACDC CPU回归及8batch实际GPU/40体积验证通过，同best的fit/test逐值一致，ROI状态正确。正式配置不含短测Trace、不加载分割checkpoint；原增强/模型/损失/80-20划分保持。后台监督器不依赖本机终端，失败不自动重启。尚未完成400轮，不是最终成绩。


## 日志/保存修复与新复训核验

修复证据：outputs/audit/acdc_checkpoint_logging_fix/20261103T021059Z。37项CPU回归及单次8batch/40体积GPU功能通过；历史跨运行梯度比较有差异，同环境旧回调控制退出-9，不能伪称逐位一致。新授权下的播种证据和旧/新回调对照位于outputs/audit/acdc_baseline_repeat_launch/20261103T024118Z，通过后才能更新readiness的源差分。本次不改原400轮结果、权重及失败验收。
