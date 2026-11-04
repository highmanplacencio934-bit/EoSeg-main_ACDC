# ACDC baseline seed42 对照训练

核验时间：2026-11-03T21:00:27.159137+00:00。本轮按用户授权启动一次，非 seed0 重复；唯一科学变量为 seed_everything 从 0 改为 42。

## 实际状态

- 正式 run：outputs/baseline_seed42/20261103T205219Z。
- 训练 PID：1835139；监督器 PID：1826544。PID 仅为本次快照，后续按 status 的 start_ticks/cwd/cmdline 重新核验。
- 本次进度快照：第 10 轮训练部分，global_step=940；不将训练部分完成当作该轮评估完成。
- 正式 Startup 与本次 seed42 冒烟的全部初始 state/RNG、373 组优化器逐值一致。
- 对照 seed0：Query、分割头和 RNG 已改变；预训练编码器状态与优化器分组相同。没有将 seed42 初始化强行对齐 seed0。
- 尚未完成 400 轮，不提供最终 seed 差异结论。

## 固定配置与数值设置

纯 models.eoseg.EoSegMultiQSegFusion，ACDCMaskClassificationSemantic，profile=baseline，ASEE/CCPS 不实例化。入口 scripts/run_acdc_deterministic_v2.py fit，协议 acdc_ce2d_stableseg_execution_v2。ROI=true，deterministic_enabled=true。

seed=42，LR=5e-4，400 轮/37600 step，224/patch16，4 类/6 query，batch16/workers8；原 AdamW/WD .05/LLRD .8/warmup-poly/损失/clip .01、16-mixed/default compile、80/20 患者名单（1512 切片/40 体积）、原 ACDC 随机训练增强和归一化不变。从同一原 DINO 预训练重新初始化，不加载分割 checkpoint；正式 probe_batches=0，无短测停止回调。

Python/NumPy/Torch CPU/CUDA 随 42 播种，PL_GLOBAL_SEED=42、PL_SEED_WORKERS=1；8 worker 通过 lightning.fabric.utilities.seed.pl_worker_init_function 派生固定种子。解释器 PYTHONHASHSEED 仍为 0，CUBLAS_WORKSPACE_CONFIG=:4096:8、OMP_NUM_THREADS=1 与 seed0 相同。strict=true/warn_only=false，cuDNN deterministic=true/benchmark=false；不能仅据短测声称完整 400 轮已逐位可复现。

每轮完整 40 体积、1-view Dice 选 best_test.ckpt，ACDCModelCheckpoint 每轮滚动 last.ckpt，保留 [EPOCH-BEST] 日志。毫米/体素 HD95 均关闭，距离 CSV 空/JSON null，无 TTA。类别 1=RV 右心室腔、2=MYO 心肌、3=LV 左心室腔，无背景汇总。最终比较同一 best 的各类及平均 Dice，不跨 epoch 挑类。

## 前序与启动验收

前序 outputs/experiments/asee_text_ccps_full_v2/20261103T134235Z 已实际通过完整 400 轮/37600 step、400 条 history/16000 体积行、last399/37600、best/last 和退出 0 验收。最佳第 339 轮，平均 Dice 89.896280%。未修改前序权重或产物。

seed42 实际 CPU CLI 前后向通过；本种子真实 GPU 默认 compile/FP16/batch16/workers8 八 batch、6 次真实 AdamW 更新、完整 40 体积/120 器官评估通过，HD95 断言守卫零调用，退出 0。预训练、数据列表及源码哈希经原 ACDC readiness 与本项 frozen_sources 核验，不仅刷新哈希放行。

与 seed0 参照 outputs/baseline/repeat_2/20261103T025120Z 的科学配置完整归一化对照通过：除 seed 和独立输出/记录路径外完全一致。新的隔离 Startup 没有复用硬编码 seed0 的断言；序列化 tuple/list 统一表示后比较真实值，实际不同值仍会被拒绝。

## 入口与证据

- outputs/baseline_seed42/preparation/baseline42_runtime.py：唯一启动、前序验收、资源检查、GPU admission、后台监督和完成验收。
- preparation/cpu_probe.py、cpu_probe_result.json、admission_cpu_checks.json：CPU 入口、配置、元数据比较回归。
- preparation/readiness.json、launch_verification.json：CPU/GPU 和正式启动证据；smoke_startup 指向本种子真实短测。
- 根 launch_receipt.json、waiting_receipt.json、latest_path.txt；正式 run 的 config.yaml、launch.json、status.json、startup_verification.json、initialization_comparison.json、seed0_initialization_comparison.json、progress.json。
- 根共享 outputs/baseline/launch.lock 与本项独立锁/回执防重复。仅资源满足（>=22000MiB、>=30GiB）时启动；不停止其他用户、不改 batch/精度/compile。

应用一次任务 acdc-8-baseline-seed42 已删除。唯一服务器监督器和训练继续，不依赖本机开关机；失败保留证据且不自动重训，不新增长期监控或其他实验。完整完成仍须退出 0、last399/37600、400 轮/16000 体积行及 best/last 全部通过，之后再与 seed0 同口径比较。
