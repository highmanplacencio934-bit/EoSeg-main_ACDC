# ACDC 服务器终端约定

> 2026-11-02 最新默认：用户已授权纯 ACDC baseline 正式训练。默认入口为 scripts/run_acdc_deterministic_v2.py fit，配置 configs/acdc/deterministic_v2/baseline.yaml，roi_metrics=true、deterministic_enabled=true、profile=baseline。ASEE/CCPS关闭，两HD95关闭，无TTA；原随机增强、80/20、400轮/37600step、224/patch16/4类/6query/LR5e-4/batch16/workers8保持。main.py及原YAML保留为显式原CUDA兼容路径，不是当前默认启动方案。开关仍可显式覆盖，不热切换正在训练的进程。


当前项目：Y:\EoSeg-main_acdc；服务器 /Disk/HDD/Sk/EoSeg-main_Synapse_acdc/EoSeg-main_acdc，SSH anhuiuniversity.icu。

~~~bash
cd /Disk/HDD/Sk/EoSeg-main_Synapse_acdc/EoSeg-main_acdc
HF_HUB_OFFLINE=1 TORCH_HOME=/Disk/HDD/Sk/.cache/torch TIMM_USE_OLD_CACHE=1 \
/Disk/HDD/Sk/miniconda3/envs/DeepLearning_xiakun/bin/python ...
~~~

代理已获直接终端/SSH权限；正式训练仍需任务授权。SSHFS不等于服务器运行。当前ACDC baseline使用main.py/原CUDA，不直接调用Synapse ROI/V2/readiness，不继承其长训、低LR或定时安排。
默认seed0，由Lightning负责worker派生播种；真实GPU前必须核验环境和资源。CPU检查可在自己的子环境隐藏CUDA，不能让空CUDA_VISIBLE_DEVICES进入正式GPU命令。
每轮完整40测试体积Dice选best，共400轮，不自动TTA/HD95。
所有新产物放本项目outputs；不要写同级Synapse。每次交付列出文件变化和未验证项。

独立ACDC V2已适配并完成短测。显式使用scripts/run_acdc_deterministic_v2.py及configs/acdc/deterministic_v2对应profile时，另外在Python启动前设置CUBLAS_WORKSPACE_CONFIG=:4096:8、PYTHONHASHSEED=0、OMP_NUM_THREADS=1，并通过ACDC专用readiness；不把这些设置静默追加到原CUDA baseline。正式400轮仍需用户指令。
