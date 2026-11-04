# ACDC baseline第二次同配置正式训练

输出：outputs/baseline/repeat_2/20261103T025120Z
启动核验training PID=46680，supervisor PID=46552。后续必须重新核验PID/start_ticks/cwd/cmdline。

## 实际启动与播种

seed=0，真实PL_GLOBAL_SEED=0、Torch CPU/CUDA初始seed=0。LightningCLI调用seed_everything(0,workers=True)，实际train DataLoader使用lightning.fabric.utilities.seed.pl_worker_init_function，8个worker各自派生Python/NumPy/Torch种子，PL_SEED_WORKERS=1；不是8个worker都设成相同0。启动前PYTHONHASHSEED=0、CUBLAS_WORKSPACE_CONFIG=:4096:8、OMP_NUM_THREADS=1。strict deterministic=True、warn_only=False、cuDNN deterministic=True、benchmark=False、matmul precision=medium。

实际EoSegMultiQSegFusion，ASEE/CCPS均关闭；ROI开、两HD95关闭、无TTA。400轮/37600step、LR5e-4、224/patch16/4类/6query、batch16/workers8，原ACDC增强/归一化、80训练患者1512切片及20人40体积、AdamW/WD.05/LLRD.8/warmup-poly/损失/clip.01/FP16/default compile全部保持。从同一原DINO预训练重新初始化，不加载分割checkpoint；正式probe_batches=0，无Trace停止回调。所有初始state/RNG及373组优化器与首轮一致，见startup_verification.json、initialization_comparison.json。

## 改动与验收

只增加每轮[EPOCH-BEST]日志、ACDCModelCheckpoint非best轮last保存、Startup播种记录，输出/logger/protocol/cache独立；显式seed0与首轮默认0相同。模型、数据、损失、科学参数和CUDA实现未修改。

保存/日志修复37项CPU测试通过。本次重启后同环境旧/新回调各8个原生batch、6次实际AdamW更新、完整40体积评估均退出0；初始化、输入/RNG、loss、scaled/unscaled梯度、权重、AdamW状态逐值相同。ROI=true，两HD95空。readiness附当前源差分及该证据后通过，未仅刷新hash。证据：outputs/audit/acdc_baseline_repeat_launch/20261103T024118Z/same_environment_comparison.json。

之前跨历史运行比较有梯度差异，旧控制退出-9并随后服务器重启，原记录保留；当前短测不证明全部历史环境可对齐，也不证明400轮最终必然逐位一致。完整复现待本轮完成后比较逐轮/逐病例及best/last状态内容，不能只比文件SHA。

首轮已完整400轮/37600step且所有评估齐全，但历史best/last均在251轮。原权重/CSV/failed_completion_validation保留，未伪造最终400轮权重。本轮是用户最新直接授权的新独立训练，不自动恢复旧ASEE队列。

后台监督器不依赖本机连接。只启动baseline一次，无额外定时、监控、创新/TTA/HD95任务；运行依赖冻结。
