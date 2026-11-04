# ACDC 预训练模型

pretrained/acdc_pretrained_manifest.json记录当前用途、真实路径、SHA256与3前景/1背景的关系。

## 视觉主干

当前纯baseline仍由timm默认离线流程加载：
/Disk/HDD/Sk/.cache/torch/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth

该DINOv2是通用主干，不含本项目4类分割头；不移动共享cache、不重复复制、不裁切权重。模型保持224输入/patch16；后续分割头由EoSeg初始化，4语义类之外另有no-object输出。manifest记录这份共享缓存的SHA，未修改原模型加载入口。

## 文本编码器

Bio_ClinicalBERT/d5892b39a4adaed74b92212a44081509db72f87b保留官方原文件、LICENSE和source_manifest.json；已有5个官方文件SHA全部核对。vendor目录的README/config/vocab/model/source_manifest均不改写成ACDC特定权重。

BioClinicalBERT是通用医学文本编码器，不按3类或4类裁剪词表/权重，不需要为ACDC重新训练。ACDC适配发生在三条解剖结构描述及输出的3×768文本特征，背景不编码。读取时local_files_only=True，在CPU上提取CLS；原模型MLM/next-sentence头未用于AutoModel特征提取，加载日志中的这些unexpected head keys不代表随机缺失的主干权重。

纯baseline不使用BERT或任何解剖先验。只有后续显式配置ASEE/CCPS才读取priors/acdc_asee_80_20_v1.pt；先验统计对应80位训练患者1512张切片。该路径由后续显式配置创新模型使用。
