# 机器学习课程设计

本仓库为机器学习课程设计四个图像任务的代码与运行结果。四个任务分别对应图像分类、图像情感分类、中心凹检测定位和医学影像分割，均使用 PyTorch 完成训练与提交。

## 目录结构

```
.
├── 01_plant_classification/     植物分类（5 类）
│   ├── plant_classification.ipynb
│   └── results/
├── 02_fer2013/                  图像情感分类（6 类）
│   ├── train_fer.py
│   └── results/
├── 03_fovea_localization/       中心凹检测定位
│   ├── train_fovea.py
│   └── results/
├── 04_vessel_segmentation/      医学影像分割
│   ├── train_vessel.py
│   └── results/
├── requirements.txt
└── README.md
```

## 实验概览

| 任务 | 模型与方法 | 数据规模 | 评价指标 | 线上成绩 |
| --- | --- | --- | --- | --- |
| 植物分类 | ConvNeXt-Tiny 迁移学习 + 水平翻转 TTA | 500 张训练 / 378 张测试 | Accuracy | 0.9232 |
| 图像情感分类 | ResNet18 + DenseNet121 异构集成，TenCrop 推理 | 28,275 张训练 / 7,065 张测试 | Accuracy / Macro F1 | 0.719 |
| 中心凹检测定位 | ResNet34 热图回归 + 局部 FPN 两级精炼，5 折交叉验证 | 80 张训练 / 20 张测试 | RMSE（像素） | 21.38883 |
| 医学影像分割 | SA-ResUNet + 五通道级联精炼网络，5 折交叉验证 | 20 张训练 / 20 张测试 | Dice | 0.85817 |

各任务的详细方案、训练过程和结果分析见课程设计报告。

## 运行环境

- 平台：竞赛提供的 Notebook 环境
- GPU：NVIDIA Tesla T4（16 GB）
- Python 3.10，PyTorch 2.x，torchvision 0.15+
- 其他依赖见 `requirements.txt`

## 运行说明

四个脚本中的数据集路径均按竞赛环境的挂载目录编写，例如：

- `/kaggle/input/competitions/2026task1/...`
- `/kaggle/input/competitions/2026task2/fer_data/fer_data`
- `/kaggle/input/competitions/2026task-4/segmentation`

在本地或其他环境运行时，修改各文件 `Config` 中的路径即可。脚本会自行完成训练、验证、测试集推理与提交文件生成，结果默认写入当前工作目录。

- `01_plant_classification/plant_classification.ipynb`：按单元格顺序运行。
- `02_fer2013/train_fer.py`：`python train_fer.py`。
- `03_fovea_localization/train_fovea.py`：`python train_fovea.py`。
- `04_vessel_segmentation/train_vessel.py`：`python train_vessel.py`。

任务 3 和任务 4 采用 5 折交叉验证，完整训练耗时较长；任务 4 中 Stage-1 与 Stage-2 的权重会分别保存在 `checkpoints/` 下，重复运行时若检测到权重则会直接加载。

## 结果文件

`results/` 目录保存了各任务的提交文件、训练日志、指标表格和可视化图片：

- 任务 1：训练/验证损失与准确率曲线、混淆矩阵、分类报告、提交文件。
- 任务 2：35 轮训练日志、消融实验汇总、逐类指标、混淆矩阵、提交文件。
- 任务 3：5 折误差对比、消融趋势、坐标散点、误差分布、提交文件。
- 任务 4：阶段消融汇总、逐折结果、逐图指标、面积占比、预测掩码、提交文件。

模型权重文件体积较大，未上传至仓库。
