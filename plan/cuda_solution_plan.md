# ACDC CUDA确定性实施与验收Plan

> 2026-11-02 最新默认：用户已授权纯 ACDC baseline 正式训练。默认入口为 scripts/run_acdc_deterministic_v2.py fit，配置 configs/acdc/deterministic_v2/baseline.yaml，roi_metrics=true、deterministic_enabled=true、profile=baseline。ASEE/CCPS关闭，两HD95关闭，无TTA；原随机增强、80/20、400轮/37600step、224/patch16/4类/6query/LR5e-4/batch16/workers8保持。main.py及原YAML保留为显式原CUDA兼容路径，不是当前默认启动方案。开关仍可显式覆盖，不热切换正在训练的进程。


状态：用户“全部修复”已授权本轮移植与必要测试；独立ACDC V2及G0–G5有界验证完成。默认main.py原CUDA路径保留，没有400轮/G6训练授权或成绩。

## 1. 当前事实

固定80/20、1512训练切片/40评估体积、400轮/37600step、224/patch16/4类/6query/LR5e-4、batch16/workers8及每轮1-view Dice选择。原训练增强、原归一化、AdamW/WD.05/LLRD.8/warmup500-1000/poly.9、loss12544/3/.75和5/5/2/.1、clip.01、FP16/default compile不变，两HD95关闭，TTA已删除。

独立入口scripts/run_acdc_deterministic_v2.py；configs/acdc/deterministic_v2/baseline.yaml、asee.yaml、ccps.yaml；协议acdc_ce2d_stableseg_execution_v2。门禁outputs/determinism/acdc_ce2d_stableseg_execution_v2/readiness.json要求本数据集、profile、G0–G5和全部当前源/配置/清单/先验/预训练hash。旧数据集readiness不可替代。

## 2. 数值路径边界

Python启动前CUBLAS_WORKSPACE_CONFIG=:4096:8、PYTHONHASHSEED=0、OMP_NUM_THREADS=1；HF_HUB_OFFLINE=1、TORCH_HOME=/Disk/HDD/Sk/.cache/torch、TIMM_USE_OLD_CACHE=1保留。
严格模式cuDNN benchmark=False/deterministic=True、torch.use_deterministic_algorithms(True,warn_only=False)，保持medium matmul。禁止用warn-only、减batch、FP32或关compile绕过失败。

复用已验证的确定性CE/双线性点采样目标和坐标规则，4类之外no-object索引4、权重.1保持。仅CCPS profile采用局部BN编译边界及backward_pass_autocast=off；baseline/ASEE不套用此专用控制。
数学目标相同不表示原CUDA与V2逐位一致；跨协议成绩须说明路径，不能将数值协议带来的变化归因于创新。

## 3. 验收状态

| 阶段 | 证据 | 状态 |
|---|---|---|
| G0 数据/配置/CPU | 128项当前回归、实际网络profile及完整80/20 | 通过 |
| G1 原GPU入口 | 原baseline八batch及完整40体积、best/last/重载 | 通过 |
| G2 GPU算子 | CE和点采样各30次，4类/6query/5logits | 通过重复性 |
| G3 criterion | 真实batch16、224标签，loss与梯度三次重放 | 通过 |
| G4 独立进程更新 | 三模型各两次，scaled/unscaled梯度、6次实际AdamW更新、state/moments | 通过短测 |
| G5 冷/暖缓存轨迹 | 三模型原生8-worker八batch，输入/RNG/损失/更新精确一致，完整40体积 | 通过短测 |
| G6 完整训练重复 | 同模型至少两次独立400轮，每轮40体积 | 未授权、未执行 |

G0–G5通过不证明完整400轮逐位一致、跨硬件或环境一致。原CUDA并未被强行对齐到V2。诊断探针首次CPU/GPU target设备不一致的失败保留，只修改探针设备传输。

## 4. 后续执行与保护

尚无长期等待监督器、自动G6或定时任务。后续用户授权后才正式训练；同模型对照必须同科学设置、先验、预训练、数值profile和环境，独立目录防覆盖。
每次记录实际PID/命令、源码/资产hash、全初始state含_extra_state、Python/NumPy/Torch/CUDA RNG、优化器分组和GradScaler实际跳步。完整重复比较状态内容和逐轮/逐体积Dice，不能仅比较ckpt文件SHA或最终近似分数。

原训练增强保留，诊断固定输入不写回正式配置。其他用户存在时按现场余量判断，不停止他人。当前短测最高reserved约10370MiB，正式运行须额外留开销并监测。
证据：outputs/audit/acdc_all_fixes/20261102T091215Z/gpu_validation_retry1；独立readiness与源hash为当前状态。全原始几何/真实HD95仍未验证。

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
