# ACDC Dice-only与HD95启用边界

> 2026-11-02 最新默认：用户已授权纯 ACDC baseline 正式训练。默认入口为 scripts/run_acdc_deterministic_v2.py fit，配置 configs/acdc/deterministic_v2/baseline.yaml，roi_metrics=true、deterministic_enabled=true、profile=baseline。ASEE/CCPS关闭，两HD95关闭，无TTA；原随机增强、80/20、400轮/37600step、224/patch16/4类/6query/LR5e-4/batch16/workers8保持。main.py及原YAML保留为显式原CUDA兼容路径，不是当前默认启动方案。开关仍可显式覆盖，不热切换正在训练的进程。


当前main.py + configs/acdc/vit_query_mul_scale_fusion.yaml的训练与独立test都只计算Dice，每轮完整40测试体积，默认1-view，不自动TTA。
毫米与体素HD95均不调用；CSV距离字段为空、JSON为null。原辅助函数中的0占位不是距离成绩，不对外报告。

ACDC H5没有spacing/affine。补充代码中的固定z_spacing=5不能作为逐病例真实几何，也不能使用另一数据集的间距表。当前尚不能计算经核验的真实mm HD95。

后续只有明确要求独立距离测试时，才准备原始NIfTI或可靠逐体积几何，对齐当前H5的轴序、shape、标签、裁剪/重采样记录，再适配并验证独立测试接口。
原HD95相关源码保留，不删除；本文件不是已实现毫米开关或已获测试授权的声明。不得为得到数值而把单位网格伪称mm。

2026-11-02最新约定为80位患者训练、20位患者共用验证测试。400轮每轮完整40体积test Dice选best，两种HD95继续关闭。独立test评估相同test.txt，并按RV/MYO/LV及平均Dice导出。

当前新增ACDC创新配置、单视角ROI和独立V2同样只使用Dice，不导入Synapse物理包装。TTA代码与参数已删除，原训练随机增强保留。GPU短测与CPU调用守卫均核验Dice-only；独立test默认也不计算任何HD95。
