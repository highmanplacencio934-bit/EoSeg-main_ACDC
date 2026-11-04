# ACDC 三前景先验

本目录只放当前ACDC使用的标签/先验；其他数据集的旧文件已移到本轮outputs/audit/acdc_prior_adaptation的legacy_priors，不覆盖同级项目。

- acdc_label_map.verified.json：0背景，1右心室腔RV、2心肌MYO、3左心室腔LV。依据[ACDC官方标签定义](https://www.creatis.insa-lyon.fr/Challenge/acdc/evaluation.html)与patient031/050/070/090训练图像/标签抽样叠加核验；不代表原始NIfTI全量身份或几何核验。
- acdc_asee_80_20_v1.pt：ASEE使用的foreground-only bundle，行顺序严格[1,2,3]；text_features=[3,768]，conditional_matrix=[3,3]。背景没有文本或共现先验行。
- acdc_prior_manifest.json：可读的路径、SHA、形状、标签顺序、训练样本数、统计矩阵及构建来源。

先验仅使用固定train.txt的1512张未增强H5 label，来自patient021–100共80患者；不读patient001–020验证/测试病例的图像或标签构建统计，不使用scribble。统计单位是一张切片；类别在切片上存在时计数一次。背景全空切片保留在总样本数N中，但没有背景矩阵行/列。

沿用原先验公式与eps=1e-5：
~~~text
P(j) = (count(j)+eps)/(N+eps)
P(j|i) = (pair(i,j)+eps)/(count(i)+eps)
M[i,j] = log((P(j|i)+eps)/(P(j)+eps))
~~~
conditional_matrix是上述log-ratio，不是应当每行和为1的概率表；不能擅自归一化或截掉负值。

文本使用本地BioClinicalBERT、eval/inference、last_hidden_state的CLS特征；没有微调文本模型。投影至ViT的1024维发生在后续ASEE模型中，不在此文件里预训练投影。
ACDC num_classes=4包含背景；queries_per_class=2作用于3前景，num_q=6。掩码分类头额外no-object通道的索引为4，不是第四个前景，也不是背景0。

构建接口（拒绝覆盖已有prior）：
~~~bash
HF_HUB_OFFLINE=1 TORCH_HOME=/Disk/HDD/Sk/.cache/torch TIMM_USE_OLD_CACHE=1 \
/Disk/HDD/Sk/miniconda3/envs/DeepLearning_xiakun/bin/python -m scripts.build_acdc_asee_prior
~~~
重建必须另指定新output并记录原因，不覆盖已有训练使用的文件。当前bundle经过CPU加载/形状/梯度验证不等于ACDC创新长训或V2/GPU已就绪；尤其不要套用依赖其他数据集spacing的CCPS训练包装类。

当前采用用户2026-11-02确认的80/20划分。val.txt与test.txt逐行相同，fit与独立test均读取test.txt；训练名单与这20位患者无交叉。旧70人统计的prior及其manifest保存在outputs/audit/acdc_split_80_20对应目录的legacy_priors。
