# ACDC 随机性、seed与重复性核查

> 2026-11-02 最新默认：用户已授权纯 ACDC baseline 正式训练。默认入口为 scripts/run_acdc_deterministic_v2.py fit，配置 configs/acdc/deterministic_v2/baseline.yaml，roi_metrics=true、deterministic_enabled=true、profile=baseline。ASEE/CCPS关闭，两HD95关闭，无TTA；原随机增强、80/20、400轮/37600step、224/patch16/4类/6query/LR5e-4/batch16/workers8保持。main.py及原YAML保留为显式原CUDA兼容路径，不是当前默认启动方案。开关仍可显式覆盖，不热切换正在训练的进程。


适用：当前main.py + configs/acdc/vit_query_mul_scale_fusion.yaml。当前使用原CUDA路径；以下依据本项目代码、CPU入口探针及当前GPU短测，不沿用其他数据集的确定性验收结果。

## 1. 当前种子和开关

| 项目 | 当前情况与证据边界 |
|---|---|
| 全局seed | main.cli_main的seed_everything_default=0；真实CLI探针PL_GLOBAL_SEED=0 |
| Python random/NumPy/Torch | 使用Lightning播种流程；真实ViT CPU初始化已保存这三者的RNG哈希 |
| DataLoader worker | PL_SEED_WORKERS=1，num_workers=8；Lightning为各worker派生种子，不应把8个worker强行设置为同一seed |
| CUDA RNG | 原CUDA有Torch播种；独立V2的三模型双进程短轨迹已通过，完整400轮未验证 |
| PYTHONHASHSEED | 当前标准命令没有显式在Python启动前固定；CPU探针CLI前后均未设置，不能宣称已固定 |
| cuDNN benchmark | 当前CPU及原GPU冒烟均False；正式运行继续记录实际值 |
| cuDNN deterministic | 当前CPU及原GPU冒烟均False |
| 严格确定性算法 | 当前CPU及原GPU冒烟torch.are_deterministic_algorithms_enabled()=False |
| CUBLAS_WORKSPACE_CONFIG | 当前标准入口未设置；CPU探针为空 |
| matmul precision | main.LightningCLI设置medium，保持现有数值路径 |

探针报告见outputs/audit/acdc_codex_cleanup/latest_path.txt指向目录的acdc_cli_backend_probe.json。该次仅用微型替代主干和CPU/FP32/compile-disabled检查入口开关，没有GPU训练。此前真实ViT-L初始化证据在baseline适配审计目录，二者不要混淆。

同一全局seed时，各worker的派生seed不同是设计行为：让并行worker使用不同的随机流。重复性还依赖同一名单、样本顺序、worker数、预取/持久化方式、环境与算子路径；workers被播种并不自动证明全训练可重复。

## 2. 数据和输入先核对

- 固定train/val/test名单hash；80训练/20评估患者无交叉，val.txt与test.txt是同一名单。
- 训练1512张切片，shuffle=True/drop_last=True，batch16/workers8，每轮94批次。
- 读取H5 image/label，不读取scribble。
- 训练切片min-max、评估体积min-max、模型内部ImageNet标准化保持；不为复现改成统一归一化。
- 原随机旋转/翻转和resize保留；诊断可固定输入，正式训练不能默默取消增强。
- 保存前若干batch的样本名、增强图像、标签和损失随机点，而不是仅比较文件名。

## 3. 初始化、优化器与首步

当前真实CPU初始化记录：纯EoSeg、224/patch16/4类/6query、388状态张量、373优化器组，LR5e-4、AdamW/WD.05/LLRD.8、warmup500/1000、400轮/37600step。
从同一缓存预训练重新初始化；比较模型状态的内容、RNG和参数组，不仅比较seed文本。

若有重复结果差异，先固定同batch，比较前向、匹配、最终采样点、分项loss、scaled/unscaled/裁剪后梯度、GradScaler状态、真实AdamW更新与moments。
预热起点LR为0时参数暂不移动，不代表梯度和AdamW状态相同；GradScaler跳步也不能当成一次成功更新。

## 4. 同checkpoint先排除评分变化

使用同一权重、1-view、精度与compile模式，比较40体积×3类的120项Dice；恢复原体积后再评估，保留legacy空mask。
当前每轮测试集选择best，未用独立val选模；HD95均关闭、未计算字段为空。不能拿TTA与1-view或不同选模集直接比较。

## 5. 后续行动边界

默认baseline不额外启用严格确定性、设置新的CUBLAS协议或替换criterion。独立ACDC修复协议已按[cuda_solution_plan.md](cuda_solution_plan.md)完成必要短测，与原入口的差异分别记录；正式训练仍需授权。
不以更换硬件、减batch、关compile、改FP32、换增强或挑选较低baseline来解释或改善结果；不要猜测固定的精度差幅度。
独立V2已有三模型双进程八batch冷/暖短轨迹证据；原CUDA长训和V2完整400轮重复、多seed均未执行，须另行授权。

## 6. CPU探针对照条件

先验适配后的真实baseline CPU初始化在与原探针相同的OMP_NUM_THREADS=1条件下，388状态张量/RNG/373优化器组逐项一致。最初OMP2探针与旧条件不同且哈希不一致，日志保留，不将其当作已证明baseline改变或CUDA根因；这也不授权给正式baseline命令追加OMP覆盖。原CUDA全训练重复性仍未验证。

80/20后同条件真实CLI初始化对照在outputs/audit/acdc_split_80_20/20261102T082608Z/baseline_comparison.json，388状态与RNG逐项一致。训练名单扩充导致每轮/总步数变为94/37600，记录中单独列出这一变化。

## 7. 本轮独立V2状态

scripts/run_acdc_deterministic_v2.py显式配置V2时才启用严格模式及CUBLAS/PYTHONHASHSEED/OMP环境；原main.py和主YAML不变。baseline/ASEE/CCPS各两独立进程八batch的初始state/RNG/参数组、数据/增强、loss、scaled/unscaled梯度和每次AdamW状态完全相同，均覆盖6次真实更新与完整40体积评估；该结论不延伸为400轮逐位保证。
TTA已删除；训练原随机增强及各worker独立派生seed保持，不把8个worker设成同一seed。证据：outputs/audit/acdc_all_fixes/20261102T091215Z/gpu_validation_retry1。具体门禁见cuda_solution_plan.md。

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


## 本次ACDC实际播种核验

入口scripts/run_acdc_deterministic_v2.py的默认seed=0，LightningCLI._set_seed实际使用workers=True；同版本Fabric seed_everything播种Python random、NumPy全局和torch.manual_seed（含CUDA），PL_GLOBAL_SEED=0/PL_SEED_WORKERS=1。8个worker使用pl_worker_init_function分别派生Python/NumPy/Torch种子，保持worker之间不同、重复运行同配置序列可复现。ACDC增强采用已播种的random.random/np.random.randint，损失/初始化采用Torch随机源，未发现新建未播种default_rng/RandomState路径。正式Startup将再次核验实际worker_init_fn及CPU/CUDA初始seed为0；本文件不是完整400轮逐位可重复的证明。证据：outputs/audit/acdc_baseline_repeat_launch/20261103T024118Z/seed_source_audit.json.
