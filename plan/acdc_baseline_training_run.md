# ACDC baseline正式训练


## ACDC baseline正式运行记录

输出：outputs/baseline/20261102T123802Z。训练PID以status.json实时身份核验；启动时2740388，监督器2740196。已核验cuda:0、纯EoSeg/ASEE关/CCPS关、ROI开/严格确定性开、400轮/37600step、LR5e-4、batch16/workers8、每轮完整40体积Dice、两HD95关。33项ACDC CPU回归及8batch实际GPU/40体积验证通过，同best的fit/test逐值一致，ROI状态正确。正式配置不含短测Trace、不加载分割checkpoint；原增强/模型/损失/80-20划分保持。后台监督器不依赖本机终端，失败不自动重启。尚未完成400轮，不是最终成绩。

原始化及验证：outputs/audit/acdc_baseline_launch/20261102T122507Z
