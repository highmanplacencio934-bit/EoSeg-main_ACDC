# ACDC 当前代码的 baseline

> 2026-11-02 最新默认：用户已授权纯 ACDC baseline 正式训练。默认入口为 scripts/run_acdc_deterministic_v2.py fit，配置 configs/acdc/deterministic_v2/baseline.yaml，roi_metrics=true、deterministic_enabled=true、profile=baseline。ASEE/CCPS关闭，两HD95关闭，无TTA；原随机增强、80/20、400轮/37600step、224/patch16/4类/6query/LR5e-4/batch16/workers8保持。main.py及原YAML保留为显式原CUDA兼容路径，不是当前默认启动方案。开关仍可显式覆盖，不热切换正在训练的进程。


辅助约定：[随机性与seed现状](random_result.md)、[独立V2实施与验收](cuda_solution_plan.md)、[HD95边界](hd95_mm_disabled.md)。

更新：2026-11-02 America/Denver。依据当前 configs/acdc/vit_query_mul_scale_fusion.yaml 及代码，不将其他数据集或未核实的论文参数写成本项目事实。

## 1. 配置选择与适配边界

默认baseline仍选vit_query_mul_scale_fusion.yaml；另有独立ASEE/ASEE+CCPS配置及deterministic_v2目录，不自动启用。采用其中的EoSegMultiQSegFusion / ACDCMaskClassificationSemantic，关闭所有创新模块。training/acdc_baseline_semantic.py虽然名为baseline，却不是此YAML调用的训练类，不能据它的普通AdamW或0.3NLL+0.7Dice推断本次损失。

当前按用户2026-11-02指令使用80/20患者划分和每轮测试集Dice选模；训练数据由70人扩大为80人，派生步数和先验统计已同步。原配置与源码备份见当前审计目录originals。本次另核对上游公开ACDC配置和加载器，确认每轮测试集Dice选模的调用；没有作者外部名单或论文发布时版本证明，不能声称当前副本与发表版本逐字相同。

## 2. 固定参数

| 项目 | 当前实际设置 |
|---|---|
| 网络 | models.eoseg.EoSegMultiQSegFusion |
| 训练模块 | training.acdc_mask_classification_semantic.ACDCMaskClassificationSemantic |
| 主干包装/名称 | models.vit.ViT / vit_large_patch14_reg4_dinov2 |
| 主干默认结构 | ViT-L，1024维、24 blocks、16 heads；以CPU真实初始化记录进一步核验 |
| 输入/patch/网格 | 224×224 / 16 / 14×14 |
| 类别/query | 4（含背景0）/6；class head含额外no-object通道 |
| query注入/融合 | 最后三block前注入；三层预测用末层query确定权重；共享预测头，仅监督融合结果 |
| masked attention/annealing | False/False |
| batch/workers | 16/8；shuffle=True，drop_last=True，pin_memory=True，训练persistent_workers=True |
| epochs/LR | 400 / 5e-4 |
| 优化器 | 公共LightningModule.configure_optimizers的AdamW；WD=.05 |
| LLRD | .8，llrd_l2_enabled=True，lr_mult=1，保留代码中norm及block分组行为 |
| 调度 | 非主干warmup500；主干先冻结LR500step再warmup1000；step级poly .9 |
| 损失 | 原MaskClassificationLoss/Mask2FormerHungarianMatcher |
| 采样 | num_points=12544，oversample_ratio=3，importance_sample_ratio=.75 |
| 损失权重 | mask/dice/class=5/5/2，no-object=.1 |
| ignore/background | ignore255；drop_background=False |
| 精度/裁剪/seed | 16-mixed / norm .01 / 0 |
| 编译/matmul | fit默认torch.compile；matmul precision=medium |
| 设备 | 作者入口默认devices=1、accelerator=auto；正式运行必须确认服务器GPU |
| 当前协议 | 原CUDA，无确定性CE/点采样修复，无ROI，无TTA，无ASEE/CCPS |

不能套用Synapse的220轮、384输入、16query、9类、LR2e-4，也不能套补充TransUNet代码的SGD、224默认以外的模型和CE+Dice损失。

训练1512张切片，drop_last=True，每轮94批次/1504个切片位置；每轮剩8张由shuffle决定不参与该轮。单卡无梯度累积时400轮预计37600step。不能用Synapse30360step做完成标准。

## 3. 预训练与模型初始化

models/vit.py通过timm.create_model(pretrained=True,img_size=224,patch_size=16,num_classes=0)加载默认模型；不从任何分割best/last续训。保留timm的patch kernel/位置编码适配与register token处理，不手动将patch16改14。

服务器TORCH_HOME缓存的dinov2_vitl14_reg4_pretrain.pth是本次预训练来源，实际路径/hash与真实初始化核验在acdc_audit.json。pretrained目录的BioClinicalBERT用于三前景文本特征；DINO主干仍读取共享TORCH_HOME缓存，两者都不按4类裁切权重。真实路径与SHA见pretrained/acdc_pretrained_manifest.json；纯baseline不加载BERT或任何先验。

独立ACDC V2/readiness及短测已接入，完整400轮复现成绩仍无。默认seed和worker播种不排除原CUDA训练的数值分叉；当前入口的实际后端开关与验证边界见random_result.md，不能沿用其他数据集的通过结论。

## 4. 数据与划分

用户2026-11-02确定80人训练、20人共用于每轮评估和测试。现有100位患者全部分配，按患者固定名单，同一患者两帧及全部切片归同一组。

| 用途 | 患者 | 数据 |
|---|---|---|
| 梯度训练 | patient021–100，80人 | 1512张二维H5切片 |
| 每轮评估及独立测试 | patient001–020，20人 | 40个三维H5体积 |
| val.txt兼容别名 | 与test.txt逐行相同 | 同一40个体积 |

ACDC/lists_ACDC/train.txt按患者/帧/切片数字顺序固定。fit的eval_dataset和独立test的test_dataset都显式split="test"，读取test.txt；val.txt仅是同一名单的兼容别名，setup要求其逐行一致。训练患者与评估患者无交叉，val/test按本项目约定共用。名单不随seed随机重划。

论文截图给出70训练/10验证/20测试。公开数据模块在fit和test阶段均读取test.txt，配置每轮按metrics/test_mean_dice选best。本次80/20比例来自用户最新实验决定，具体patient021–100/001–020名单是本项目固定实现，不能据截图推断作者实际患者编号。

全量数据仍为1902个二维H5及200个三维H5，两者是同100位患者两帧的不同表示。此前已全量核验1902切片与对应体积image/label逐值一致。使用image(float32)与完整label(uint8)，标签0..3；scribble不参与训练或统计。当前H5无可靠spacing/affine，原生成脚本缺失。

400轮中的每一轮评估完整40体积、1-view、三个前景Dice，测试集参与checkpoint选择；独立test同一best时仍使用相同20人。400轮共16000条逐体积行，三个类别共48000项Dice。两种HD95关闭。

## 5. 输入与在线增强契约

保持复制代码中的ACDCBaselineTransform：
1. 训练先对每张切片min-max，常量切片置零。
2. 50%概率随机k∈{0,1,2,3}旋转90k度并随机沿0/1轴翻转；否则再投一次50%硬币，25%总概率小角度整数[-20,20)旋转，25%总概率不做随机几何。
3. 调整至224×224，图像scipy zoom order3，标签order0；小角度原实现图像/标签rotate都order0。没有加新增强或修改插值。
4. 图像转float32、灰度复制三通道；标签每个有效类别构成一个bool mask，含背景，labels为int64。
5. 模型内部再做ImageNet mean/std标准化，不额外/255。

评估按整个体积min-max，逐片cubic调整至224、取argmax后nearest恢复原切片尺寸，再在完整3D预测上算Dice。训练切片min-max与评估体积min-max存在粒度差异，是当前baseline行为，未统一改动。

ACDCSemantic的color_jitter_enabled、scale_range目前未参与transform，不是Synapse的颜色/ScaleJitter.5–2/补零384裁剪。训练增强与TTA不同；当前没有自动TTA。

## 6. 评估、checkpoint及报告

每轮（1–400）通过Lightning验证阶段完整推理40测试体积，三个前景类别分别算3D Dice；每类别先平均体积，再对三个类别平均。不含背景；每患者两帧均等参与，当前不拆ED/ES指标。

三个前景类别标签1/2/3的Dice，在每个体积恢复原空间网格后计算：
~~~text
D(v,c) = 2 * |Pred(v,c) ∩ GT(v,c)| / (|Pred(v,c)| + |GT(v,c)|)
Dice(c) = sum_v D(v,c) / 40
Mean Dice = (Dice(1) + Dice(2) + Dice(3)) / 3
~~~
每个测试患者恰好两帧，因此先对患者两帧平均再对20患者平均，与上述40体积等权平均一致。不能替换成二维切片Dice平均、把所有体素池化后算一个Dice，或按器官大小加权。
全部40×3=120个测试体积-类别GT均非空，本次核验没有触发pred非空/GT空的错误满分分支；实现仍保留原legacy规则，预测为空且GT非空时Dice为0。

完整输出包括每个体积的三类Dice、各类平均和三类总平均；CSV原值0..1，展示百分数乘100。逐类及平均值必须来自同一best epoch/checkpoint，不跨epoch逐类挑最佳。


legacy空mask保持：pred和GT都有前景计算2交集/(两者体素数和)；pred非空/GT空得1；否则0。这是保留的历史代码口径，不宣称数学上合理。

默认不计算毫米或体素HD95。原ACDC辅助函数中其他指标的数值只是占位，本轮从公开日志/汇总移除其伪0，CSV空/JSON null。缺真实spacing，独立测试也不能把常数5或Synapse间距表当ACDC真实mm。

监控metrics/test_mean_dice，mode=max，save_top_k=1，save_last=True，save_weights_only=True；固定best_test.ckpt和last.ckpt，不存50份。400轮而非最后50轮选模，不加早停。相等时按Lightning top-k规则保留已有best。

输出在outputs/baseline/acdc_vit_query_mul_scale_fusion/version_*/；CSVLogger自动分配版本。新run不能复用已存在的history目录。
- checkpoints/best_test.ckpt、last.ckpt
- run_manifest.json：实际命令、解析CLI、版本、源文件/配置/名单hash、参数组及split_summary
- acdc_test_history.csv：每轮平均及RV/MYO/LV代码名称对应Dice原值0..1
- acdc_test_case_metrics.csv：逐轮逐体积三类Dice
- acdc_best_result.txt、selection_summary.json：同一best的均值/逐类，one-based epoch
- 独立test由同一回调输出acdc_independent_test_cases.csv和summary，不自动训练或开启TTA/HD95

ACDCTestLogger改用on_validation_end，读取模型on_validation_epoch_end汇总完成后的当前轮指标，避免旧回调读到前一轮；选模监控和值计算不变。

标签名称：1=RV右心室腔、2=MYO心肌、3=LV左心室腔，0背景；见priors/acdc_label_map.verified.json。已按[ACDC官方定义](https://www.creatis.insa-lyon.fr/Challenge/acdc/evaluation.html)和4位训练患者图像/label抽样核验语义，不代表原始NIfTI全量对应或真实几何已核验。

## 7. 命令（本轮未启动正式训练）

所有命令在服务器ACDC根目录，固定环境Python；以下使用当前原CUDA路径，不附带Synapse V2环境/入口。

~~~bash
cd /Disk/HDD/Sk/EoSeg-main_Synapse_acdc/EoSeg-main_acdc
HF_HUB_OFFLINE=1 TORCH_HOME=/Disk/HDD/Sk/.cache/torch TIMM_USE_OLD_CACHE=1 \
/Disk/HDD/Sk/miniconda3/envs/DeepLearning_xiakun/bin/python main.py fit \
  --config configs/acdc/vit_query_mul_scale_fusion.yaml
~~~

独立test（仍Dice-only；复现默认compile训练时显式compile_test）：

~~~bash
HF_HUB_OFFLINE=1 TORCH_HOME=/Disk/HDD/Sk/.cache/torch TIMM_USE_OLD_CACHE=1 \
/Disk/HDD/Sk/miniconda3/envs/DeepLearning_xiakun/bin/python main.py test \
  --config configs/acdc/vit_query_mul_scale_fusion.yaml \
  --ckpt_path /ABSOLUTE/PATH/TO/best_test.ckpt --compile_test
~~~

正式400轮前必须在实际GPU上做独立FP16/default compile/batch16/workers8冒烟，确认完整40体积评估、当前轮日志、HD95零调用与best/last；CPU通过不能替代。训练是否启动须用户授权。

## 8. 完成标准与未完成项

完整训练应退出0、last内部epoch399/global_step37600、400轮history、16000条体积行（48000项类别Dice）、best/last与来源记录齐全。不能把CPU小模型2轮或初始化当ACDC成绩。

GPU冒烟、独立V2与创新训练短链路已完成；400轮长训、全量原始NIfTI身份/几何核验和真实mm HD95仍未执行。三前景先验已离线生成并做CPU接口验收，未用于当前纯baseline。

## 9. 三前景先验资产

priors/acdc_asee_80_20_v1.pt：text_features[3,768]、conditional_matrix[3,3]，严格按label1/2/3排序，背景不编码。仅80训练患者1512张原始H5标签统计；未打开验证或测试H5。共现矩阵沿用原log条件概率比公式，背景全空切片只计入总N，不产生背景行列。

通用DINOv2/BioClinicalBERT原权重不改；ACDC适配发生在三条结构描述、统计矩阵及按前景数检查的加载器。queries_per_class=2得到6query；4语义类之外仍有no-object通道。未来显式配置创新模型时才加载prior；当前数据/先验采用80/20，每轮test Dice选模；其余配置参数见第2节。

先验/权重说明见[priors说明](../priors/README.md)、[pretrained说明](../pretrained/README.md)，CPU验收和归档见当前审计。ACDC的ASEE/CCPS使用现有ACDCMaskClassificationSemantic，不调用旧Synapse物理训练包装；独立V2三模型GPU短测已通过，400轮未执行。

80/20实测验收与旧先验归档见outputs/audit/acdc_split_80_20/20261102T082608Z。77项CPU回归通过，实际baseline初始化388状态/RNG/373参数组与原探针一致；新步数94/37600已按真实DataLoader核验。当前完整GPU短测状态见最新审计，400轮训练仍需授权。

## 10. 本轮修复和独立入口

五项全代码审计问题已处理；逐项证据见[运行审计](acdc_baseline_run_audit.md)。TTA相关实现和接口已按用户要求删除，当前只做单视角推理；原训练随机增强文件保持原hash。root build_prior.py委托scripts/build_acdc_asee_prior.py真实训练集构建器，旧示例只归档保存。

严格可视化入口scripts/visualize_acdc_results.py默认16-mixed、compiled forward与ACDC指标，与独立test复用原逐片预测/恢复；不得用缺失权重的strict=False结果作比较。独立单视角ROI入口scripts/evaluate_acdc.py test --roi_metrics=true --compile_test；ROI只用于指标计数，不裁切模型输入。

已准备configs/acdc/vit_query_mul_scale_fusion_asee.yaml、vit_query_mul_scale_fusion_asee_ccps.yaml。创新配置保留第2节未研究变量；ASEE双先验/patch/query开启，prior固定三前景80人版本。CCPS visual_prior、patch gate开/query gate关、presence=0、init=.95、both；不增加新辅助损失。

独立数值协议仅通过scripts/run_acdc_deterministic_v2.py与configs/acdc/deterministic_v2/{baseline,asee,ccps}.yaml显式选择。profile与网络匹配，启动前使用ACDC专用readiness及源哈希门禁。数学损失目标与原始路径相同，确定性CE/点采样和CCPS执行边界会改变浮点路径，不承诺与原CUDA逐位相同；不能把协议差异当作创新收益。默认main.py与主YAML没有启用V2或ROI。

本轮128项CPU测试、原CUDA GPU短测及V2三模型各两次八batch冷/暖缓存验证通过；没有ACDC正式分数。证据：outputs/audit/acdc_all_fixes/20261102T091215Z。

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
