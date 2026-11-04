# ACDC 服务器GPU规则

> 2026-11-02 最新默认：用户已授权纯 ACDC baseline 正式训练。默认入口为 scripts/run_acdc_deterministic_v2.py fit，配置 configs/acdc/deterministic_v2/baseline.yaml，roi_metrics=true、deterministic_enabled=true、profile=baseline。ASEE/CCPS关闭，两HD95关闭，无TTA；原随机增强、80/20、400轮/37600step、224/patch16/4类/6query/LR5e-4/batch16/workers8保持。main.py及原YAML保留为显式原CUDA兼容路径，不是当前默认启动方案。开关仍可显式覆盖，不热切换正在训练的进程。


适用：/Disk/HDD/Sk/EoSeg-main_Synapse_acdc/EoSeg-main_acdc，RTX5090D32GB；通过SSH执行，不用本机GPU替代。

用户已允许与其他用户GPU任务并发；启动前按本次实际配置峰值及临时工作区核验剩余显存。其他用户存在或SM利用率高本身不是禁止条件，余量不足则等待/报告，不中止、暂停、重启或清理他人的任务。不能自动降batch/尺寸/精度或关compile来塞入显存。
不能争用或停止同级Synapse已授权训练；本轮已授权并完成ACDC GPU短测，正式400轮训练仍需用户指令。
本轮224/batch16/workers8/FP16/default compile的V2三模型短测最高reserved约10370MiB，原入口峰值见当前审计。短测不是完整训练峰值保证，正式运行仍加临时开销；不把其他分辨率的峰值当ACDC结论。
任务授权、唯一输出、防覆盖、源文件hash和数据完整性继续适用；允许并发不等于自动授权长训或新实验。
