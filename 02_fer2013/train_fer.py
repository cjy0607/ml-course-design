# -*- coding: utf-8 -*-
"""FER2013 六类人脸表情识别。

训练 ResNet18 与 DenseNet121 两个模型，使用 TenCrop 多视角推理与加权融合，
并保存训练日志、指标表格与可视化结果。
"""

import os
import random
import warnings
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

warnings.filterwarnings("ignore")


# ==============================================================================
# 模块 1：全局超参数与实验环境配置
# ==============================================================================
class Config:
    SEED = 42
    # 若为 False，当检测到 /kaggle/working 下已有训练好的权重与日志时将直接加载并出图；
    # 若想从头完整重跑一遍 35 轮训练，请保持 True
    FORCE_RETRAIN = True

    DATA_DIR = "/kaggle/input/competitions/2026task2/fer_data/fer_data"
    TRAIN_DIR = os.path.join(DATA_DIR, "train")
    TEST_DIR = os.path.join(DATA_DIR, "test")
    SUB_TEMPLATE_PATH = "/kaggle/input/competitions/2026task2/submission.csv"

    WORK_DIR = "/kaggle/working"
    ANALYSIS_DIR = "/kaggle/working/analysis_outputs"

    # 训练超参数
    IMG_RESIZE = 128
    IMG_CROP = 112
    BATCH_SIZE = 128
    TENCROP_BATCH_SIZE = 32
    EPOCHS = 35
    WARMUP_EPOCHS = 3
    MIXUP_PROB = 0.5
    MIXUP_ALPHA = 0.2
    LABEL_SMOOTHING = 0.05

    # 最佳集成权重 (ResNet18 : DenseNet121 = 0.4 : 0.6)
    W_R18 = 0.4
    W_DN121 = 0.6

    # 权重与中间结果保存路径
    SAVE_R18_PATH = os.path.join(WORK_DIR, "best_resnet18_fer_v4_pro.pth")
    SAVE_DN121_PATH = os.path.join(WORK_DIR, "best_densenet121_fer_v4.pth")
    R18_PROBS_PATH = os.path.join(WORK_DIR, "val_probs_resnet18_v4.npy")
    DN121_PROBS_PATH = os.path.join(WORK_DIR, "val_probs_densenet121_v4.npy")
    VAL_TARGETS_PATH = os.path.join(WORK_DIR, "val_targets.npy")
    HISTORY_CSV_PATH = os.path.join(ANALYSIS_DIR, "table1_training_epoch_history.csv")
    OUTPUT_SUB_PATH = os.path.join(WORK_DIR, "submission_ensemble_r18_dn121_tencrop.csv")

    # ImageFolder 字典序编号 -> 竞赛官方编号映射
    # ImageFolder: {'Angry': 0, 'Fear': 1, 'Happy': 2, 'Neutral': 3, 'Sad': 4, 'Surprise': 5}
    # Official:    {0: Angry, 1: Fear, 2: Happy, 3: Sad, 4: Surprise, 5: Neutral}
    LABEL_MAPPING = {0: 0, 1: 1, 2: 2, 3: 5, 4: 3, 5: 4}
    OFFICIAL_CLASS_NAMES = {
        0: "Angry (0)",
        1: "Fear (1)",
        2: "Happy (2)",
        3: "Sad (3)",
        4: "Surprise (4)",
        5: "Neutral (5)"
    }


def seed_everything(seed=42):
    """固定所有随机种子，保证数据集划分与训练结果可复现"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def setup_plot_style():
    """配置 Matplotlib 学术绘图风格（自动检测中文字体，避免 Linux 环境乱码）"""
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["axes.unicode_minus"] = False
    sns.set_theme(style="whitegrid")

    # 检测系统中是否安装可用中文字体
    cjk_fonts = ["Noto Sans CJK SC", "WenQuanYi Micro Hei", "SimHei", "Microsoft YaHei"]
    available_fonts = set(f.name for f in fm.fontManager.ttflist)
    for font_name in cjk_fonts:
        if font_name in available_fonts:
            plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            return True
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    return False


# ==============================================================================
# 模块 2：数据集分层划分、多策略数据增强与 TenCrop 变换
# ==============================================================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((Config.IMG_RESIZE, Config.IMG_RESIZE), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.RandomCrop((Config.IMG_CROP, Config.IMG_CROP)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([
        transforms.RandomAffine(degrees=12, translate=(0.06, 0.06), scale=(0.92, 1.08))
    ], p=0.6),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.18), ratio=(0.3, 3.3), value=0)
])

val_single_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((Config.IMG_RESIZE, Config.IMG_RESIZE), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop((Config.IMG_CROP, Config.IMG_CROP)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

tencrop_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((Config.IMG_RESIZE, Config.IMG_RESIZE), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.TenCrop((Config.IMG_CROP, Config.IMG_CROP)),
    transforms.Lambda(lambda crops: torch.stack([
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)(transforms.ToTensor()(crop))
        for crop in crops
    ]))
])


class SubsetWithTransform(Dataset):
    """支持对训练集与验证集子集分别绑定不同数据增强的封装类"""
    def __init__(self, folder_dataset, subset_indices, transform=None):
        self.folder_dataset = folder_dataset
        self.indices = subset_indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img_path, label = self.folder_dataset.samples[real_idx]
        img = Image.open(img_path).convert("L")
        if self.transform:
            img = self.transform(img)
        return img, label


class TestTenCropDataset(Dataset):
    """测试集 TenCrop 数据加载器"""
    def __init__(self, img_paths, transform):
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("L")
        return self.transform(img)


# ==============================================================================
# 模块 3：保留浅层空间分辨率的异构网络架构定义 (ResNet18_v4 & DenseNet121_v4)
# ==============================================================================
def build_fer_resnet18(num_classes=6):
    """
    改进版 ResNet18：
    将首层步长为 2 的 MaxPool 替换为 Identity，使 112x112 输入在进入 layer1~layer4 时
    保持 56x56 -> 28x28 -> 14x14 -> 7x7 分辨率，既完整复用 ImageNet conv1 预训练权重，
    又避免 48x48 小尺寸人脸在深层坍塌为 2x2。
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.maxpool = nn.Identity()
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, num_classes)
    )
    return model


def build_fer_densenet121(num_classes=6):
    """
    改进版 DenseNet121：
    同样将 features.pool0 替换为 Identity，通过跨层特征拼接（Concatenation）复用浅层眉眼与嘴角边缘纹理，
    与 ResNet18 的残差相加机制形成强互补的异构特征表达。
    """
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    model.features.pool0 = nn.Identity()
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, num_classes)
    )
    return model


# ==============================================================================
# 模块 4：Mixup + 随机擦除 + Warmup 余弦退火训练与日志记录引擎
# ==============================================================================
def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1.0 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def lr_lambda(current_epoch):
    if current_epoch < Config.WARMUP_EPOCHS:
        return float(current_epoch + 1) / float(Config.WARMUP_EPOCHS)
    progress = float(current_epoch - Config.WARMUP_EPOCHS) / float(max(1, Config.EPOCHS - Config.WARMUP_EPOCHS))
    return max(0.01, 0.5 * (1.0 + np.cos(np.pi * progress)))


def train_model_with_history(model, model_name, save_path, criterion, backbone_lr, head_lr, head_key,
                             train_loader, val_loader, val_dataset_len, device):
    """训练单个模型并记录逐轮训练/验证指标"""
    backbone_params = [p for n, p in model.named_parameters() if head_key not in n]
    head_params = [p for n, p in model.named_parameters() if head_key in n]
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": backbone_lr},
        {"params": head_params, "lr": head_lr}
    ], weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = torch.amp.GradScaler("cuda")

    best_acc = 0.0
    history_rows = []

    print(f"\n========== 开始训练 {model_name} (共 {Config.EPOCHS} 轮) ==========")
    for epoch in range(1, Config.EPOCHS + 1):
        model.train()
        running_loss, total = 0.0, 0
        current_lr = optimizer.param_groups[0]["lr"]

        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad()
            use_mixup = (np.random.rand() < Config.MIXUP_PROB)
            with torch.amp.autocast("cuda"):
                if use_mixup:
                    mixed_imgs, y_a, y_b, lam = mixup_data(imgs, targets, alpha=Config.MIXUP_ALPHA)
                    outputs = model(mixed_imgs)
                    loss = lam * criterion(outputs, y_a) + (1.0 - lam) * criterion(outputs, y_b)
                else:
                    outputs = model(imgs)
                    loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * imgs.size(0)
            total += targets.size(0)

        scheduler.step()
        train_loss = running_loss / total

        # 单视角验证集评估
        model.eval()
        val_preds, val_targets_list, val_loss_sum = [], [], 0.0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                with torch.amp.autocast("cuda"):
                    outputs = model(imgs)
                    loss = criterion(outputs, targets)
                val_loss_sum += loss.item() * imgs.size(0)
                val_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                val_targets_list.extend(targets.cpu().numpy())

        val_loss = val_loss_sum / val_dataset_len
        val_acc = accuracy_score(val_targets_list, val_preds)
        val_f1 = f1_score(val_targets_list, val_preds, average="macro")

        is_best = False
        mark = ""
        if val_acc > best_acc:
            best_acc = val_acc
            is_best = True
            torch.save(model.state_dict(), save_path)
            mark = f"  <-- 已保存最佳 {model_name}!"

        history_rows.append({
            "Model": model_name,
            "Epoch": epoch,
            "LR": current_lr,
            "Train_Loss": round(train_loss, 4),
            "Val_Loss": round(val_loss, 4),
            "Val_Acc": round(val_acc * 100, 2),
            "Val_Macro_F1": round(val_f1, 4),
            "Is_Best": is_best
        })

        print(f"[{model_name}] Epoch [{epoch:02d}/{Config.EPOCHS:02d}] "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc*100:.2f}% | Macro F1: {val_f1:.4f}{mark}")

    return history_rows


# ==============================================================================
# 模块 5：TenCrop 十裁剪推理与异构模型网格搜索加权融合
# ==============================================================================
def evaluate_tencrop(model, loader, device, desc="TenCrop 评估"):
    """对每张图片进行 10 视角裁剪推理并平均 Softmax 概率分布"""
    model.eval()
    all_probs, all_targets = [], []
    with torch.no_grad():
        for crops, targets in tqdm(loader, desc=desc):
            bs, ncrops, c, h, w = crops.size()
            crops = crops.view(-1, c, h, w).to(device, non_blocking=True)
            with torch.amp.autocast("cuda"):
                logits = model(crops)
                probs = F.softmax(logits, dim=1)
            avg_probs = probs.view(bs, ncrops, -1).mean(dim=1)
            all_probs.append(avg_probs.cpu().numpy())
            all_targets.extend(targets.numpy())
    return np.concatenate(all_probs, axis=0), np.array(all_targets)


# ==============================================================================
# 模块 6：指标汇总与结果可视化
# ==============================================================================
def export_analysis(history_df, r18_probs, dn121_probs, val_targets,
                                       class_names, test_sub_df, train_targets):
    """导出指标表格与对比图"""
    os.makedirs(Config.ANALYSIS_DIR, exist_ok=True)
    setup_plot_style()

    # 1. 保存表 1：35 轮训练历史记录表
    if history_df is not None and len(history_df) > 0:
        history_df.to_csv(Config.HISTORY_CSV_PATH, index=False, encoding="utf-8-sig")
        print(f"[表格导出 1/3] 完整 Epoch 训练日志已保存至: {Config.HISTORY_CSV_PATH}")

    # 计算单模型与异构集成预测结果
    r18_preds = r18_probs.argmax(axis=1)
    dn121_preds = dn121_probs.argmax(axis=1)
    ens_probs = Config.W_R18 * r18_probs + Config.W_DN121 * dn121_probs
    ens_preds = ens_probs.argmax(axis=1)

    r18_acc = accuracy_score(val_targets, r18_preds) * 100
    r18_f1 = f1_score(val_targets, r18_preds, average="macro")
    dn121_acc = accuracy_score(val_targets, dn121_preds) * 100
    dn121_f1 = f1_score(val_targets, dn121_preds, average="macro")
    ens_acc = accuracy_score(val_targets, ens_preds) * 100
    ens_f1 = f1_score(val_targets, ens_preds, average="macro")

    # 2. 保存表 2：各阶段消融实验结果
    ablation_data = [
        {"Stage": "Stage 1", "Model_Variant": "ResNet18 v1 (Baseline, Unmapped)",
         "Resolution": "48x48", "Key_Techniques": "ImageNet Pretrained + Basic Finetune",
         "Val_Acc_Pct": 65.77, "Val_Macro_F1": 0.6383, "Online_Score": 0.405},
        {"Stage": "Stage 2", "Model_Variant": "ResNet18 v2 (Label Mapped + Tuned)",
         "Resolution": "48x48", "Key_Techniques": "Official Label Mapping + Label Smoothing (0.1)",
         "Val_Acc_Pct": 67.15, "Val_Macro_F1": 0.6533, "Online_Score": 0.651},
        {"Stage": "Stage 3", "Model_Variant": "ResNet18 + Flip TTA",
         "Resolution": "48x48", "Key_Techniques": "2-View Horizontal Flip TTA",
         "Val_Acc_Pct": 68.00, "Val_Macro_F1": 0.6619, "Online_Score": None},
        {"Stage": "Stage 4", "Model_Variant": "ResNet18 + CBAM Attention",
         "Resolution": "48x48", "Key_Techniques": "Channel & Spatial Attention in Layer3/Layer4",
         "Val_Acc_Pct": 66.97, "Val_Macro_F1": 0.6532, "Online_Score": None},
        {"Stage": "Stage 5", "Model_Variant": "ResNet18 v4 (Single View)",
         "Resolution": "112x112", "Key_Techniques": "Remove MaxPool + Mixup + RandomErasing + CosineLR",
         "Val_Acc_Pct": 69.55, "Val_Macro_F1": 0.6773, "Online_Score": None},
        {"Stage": "Stage 6", "Model_Variant": "ResNet18 v4 + TenCrop (10-View)",
         "Resolution": "112x112", "Key_Techniques": "High-Res Stem + 10-Crop Softmax Averaging",
         "Val_Acc_Pct": round(r18_acc, 2), "Val_Macro_F1": round(r18_f1, 4), "Online_Score": None},
        {"Stage": "Stage 7", "Model_Variant": "DenseNet121 v4 + TenCrop (10-View)",
         "Resolution": "112x112", "Key_Techniques": "Remove Pool0 + Class-Balanced Weights + TenCrop",
         "Val_Acc_Pct": round(dn121_acc, 2), "Val_Macro_F1": round(dn121_f1, 4), "Online_Score": None},
        {"Stage": "Stage 8", "Model_Variant": "Heterogeneous Ensemble (0.4 R18 + 0.6 DN121)",
         "Resolution": "112x112", "Key_Techniques": "Cross-Architecture Feature Fusion + TenCrop",
         "Val_Acc_Pct": round(ens_acc, 2), "Val_Macro_F1": round(ens_f1, 4), "Online_Score": 0.719}
    ]
    ablation_df = pd.DataFrame(ablation_data)
    ablation_csv_path = os.path.join(Config.ANALYSIS_DIR, "table2_ablation_study_summary.csv")
    ablation_df.to_csv(ablation_csv_path, index=False, encoding="utf-8-sig")
    print(f"[表格导出 2/3] 消融实验演进总表已保存至: {ablation_csv_path}")

    # 3. 保存表 3：各类别详细评估指标对比表 (ResNet18 v2 基准 vs ResNet18 v4 vs 异构双模型集成)
    # 基准 v2 各类别历史指标
    v2_f1_ref = {"Angry": 0.6005, "Fear": 0.5174, "Happy": 0.8589, "Neutral": 0.6254, "Sad": 0.5480, "Surprise": 0.7697}
    p_r18, r_r18, f1_r18, sup = precision_recall_fscore_support(val_targets, r18_preds, zero_division=0)
    p_dn, r_dn, f1_dn, _ = precision_recall_fscore_support(val_targets, dn121_preds, zero_division=0)
    p_ens, r_ens, f1_ens, _ = precision_recall_fscore_support(val_targets, ens_preds, zero_division=0)

    per_class_rows = []
    for i, cname in enumerate(class_names):
        per_class_rows.append({
            "Class_Name": cname,
            "Support": int(sup[i]),
            "Baseline_v2_F1": v2_f1_ref.get(cname, 0.0),
            "ResNet18_v4_Precision": round(p_r18[i], 4),
            "ResNet18_v4_Recall": round(r_r18[i], 4),
            "ResNet18_v4_F1": round(f1_r18[i], 4),
            "DenseNet121_v4_F1": round(f1_dn[i], 4),
            "Ensemble_Precision": round(p_ens[i], 4),
            "Ensemble_Recall": round(r_ens[i], 4),
            "Ensemble_F1": round(f1_ens[i], 4),
            "F1_Gain_vs_v2": round(f1_ens[i] - v2_f1_ref.get(cname, 0.0), 4)
        })
    per_class_df = pd.DataFrame(per_class_rows)
    per_class_csv_path = os.path.join(Config.ANALYSIS_DIR, "table3_per_class_metrics_comparison.csv")
    per_class_df.to_csv(per_class_csv_path, index=False, encoding="utf-8-sig")
    print(f"[表格导出 3/3] 6 类表情详细指标对比表已保存至: {per_class_csv_path}")

    # -------------------- 图 1：35 轮训练收敛曲线对比图 (Loss / Val Acc / Macro F1) --------------------
    if history_df is not None and len(history_df) > 0:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
        r18_hist = history_df[history_df["Model"] == "ResNet18_v4"]
        dn_hist = history_df[history_df["Model"] == "DenseNet121_v4"]

        # 子图 1: Loss 曲线
        axes[0].plot(r18_hist["Epoch"], r18_hist["Train_Loss"], "--", color="#1f77b4", alpha=0.8, label="ResNet18_v4 Train Loss")
        axes[0].plot(r18_hist["Epoch"], r18_hist["Val_Loss"], "-", color="#1f77b4", linewidth=2, label="ResNet18_v4 Val Loss")
        axes[0].plot(dn_hist["Epoch"], dn_hist["Train_Loss"], "--", color="#d62728", alpha=0.8, label="DenseNet121_v4 Train Loss")
        axes[0].plot(dn_hist["Epoch"], dn_hist["Val_Loss"], "-", color="#d62728", linewidth=2, label="DenseNet121_v4 Val Loss")
        axes[0].set_title("(a) Training & Validation Loss Curve", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Cross-Entropy Loss")
        axes[0].legend(fontsize=9)

        # 子图 2: Validation Accuracy 曲线
        axes[1].plot(r18_hist["Epoch"], r18_hist["Val_Acc"], "-o", markersize=3.5, color="#1f77b4", linewidth=2, label="ResNet18_v4 (Best: 69.55%)")
        axes[1].plot(dn_hist["Epoch"], dn_hist["Val_Acc"], "-s", markersize=3.5, color="#d62728", linewidth=2, label="DenseNet121_v4 (Best: 70.40%)")
        axes[1].axhline(67.15, color="gray", linestyle=":", linewidth=1.8, label="Baseline ResNet18_v2 (67.15%)")
        axes[1].set_title("(b) Single-Crop Validation Accuracy (%)", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Validation Accuracy (%)")
        axes[1].legend(fontsize=9, loc="lower right")

        # 子图 3: Validation Macro F1 曲线
        axes[2].plot(r18_hist["Epoch"], r18_hist["Val_Macro_F1"], "-o", markersize=3.5, color="#2ca02c", linewidth=2, label="ResNet18_v4 Macro F1")
        axes[2].plot(dn_hist["Epoch"], dn_hist["Val_Macro_F1"], "-s", markersize=3.5, color="#ff7f0e", linewidth=2, label="DenseNet121_v4 Macro F1")
        axes[2].axhline(0.6533, color="gray", linestyle=":", linewidth=1.8, label="Baseline v2 Macro F1 (0.6533)")
        axes[2].set_title("(c) Validation Macro F1 Curve", fontsize=12, fontweight="bold")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Macro F1-Score")
        axes[2].legend(fontsize=9, loc="lower right")

        plt.tight_layout()
        fig1_path = os.path.join(Config.ANALYSIS_DIR, "fig1_training_convergence_curves.png")
        plt.savefig(fig1_path, bbox_inches="tight")
        plt.close()
        print(f"[图表导出 1/5] 训练收敛曲线图已保存至: {fig1_path}")

    # -------------------- 图 2：消融实验演进与双模型融合权重搜索图 --------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    # 左图：不同阶段模型性能演进柱状图
    stages_labels = [
        "ResNet18 v1\n(Baseline)",
        "ResNet18 + CBAM\n(Overfitted)",
        "ResNet18 v2\n(Tuned)",
        "ResNet18 v2\n(+Flip TTA)",
        "ResNet18 v4\n(+TenCrop)",
        "DenseNet121 v4\n(+TenCrop)",
        "Ensemble\n(R18 + DN121)"
    ]
    acc_vals = [65.77, 66.97, 67.15, 68.00, r18_acc, dn121_acc, ens_acc]
    f1_vals = [63.83, 65.32, 65.33, 66.19, r18_f1 * 100, dn121_f1 * 100, ens_f1 * 100]

    x = np.arange(len(stages_labels))
    width = 0.36
    bars1 = axes[0].bar(x - width / 2, acc_vals, width, label="Val Accuracy (%)", color="#3b82f6")
    bars2 = axes[0].bar(x + width / 2, f1_vals, width, label="Val Macro F1 (x100)", color="#10b981")
    axes[0].set_ylim(58, 76)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(stages_labels, fontsize=8.5)
    axes[0].set_ylabel("Score (%)")
    axes[0].set_title("(a) Ablation Study Across Model Iterations", fontsize=12, fontweight="bold")
    axes[0].legend(loc="upper left")
    for bar in bars1:
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.25,
                     f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    # 右图：异构模型融合权重 w 搜索曲线 (w * R18 + (1-w) * DN121)
    weights_grid = np.linspace(0.0, 1.0, 11)
    grid_accs, grid_f1s = [], []
    for w in weights_grid:
        p_blend = (w * r18_probs + (1.0 - w) * dn121_probs).argmax(axis=1)
        grid_accs.append(accuracy_score(val_targets, p_blend) * 100)
        grid_f1s.append(f1_score(val_targets, p_blend, average="macro") * 100)

    axes[1].plot(weights_grid, grid_accs, "-o", color="#2563eb", linewidth=2.2, label="Ensemble Val Accuracy (%)")
    axes[1].plot(weights_grid, grid_f1s, "-s", color="#d97706", linewidth=2.2, label="Ensemble Macro F1 (x100)")
    axes[1].axvline(0.4, color="#dc2626", linestyle="--", linewidth=1.8, label=f"Optimal Weight w=0.4 ({ens_acc:.2f}%)")
    axes[1].set_xlabel("Weight of ResNet18_v4 (w)  [1-w for DenseNet121_v4]")
    axes[1].set_ylabel("Validation Metric (%)")
    axes[1].set_title("(b) Ensemble Weight Sensitivity Analysis", fontsize=12, fontweight="bold")
    axes[1].legend(loc="lower center")

    plt.tight_layout()
    fig2_path = os.path.join(Config.ANALYSIS_DIR, "fig2_ablation_and_ensemble_analysis.png")
    plt.savefig(fig2_path, bbox_inches="tight")
    plt.close()
    print(f"[图表导出 2/5] 消融实验与融合权重分析图已保存至: {fig2_path}")

    # -------------------- 图 3：6 类表情 F1-Score 提升对比图 (v2 基准 vs R18_v4 vs 最终异构集成) --------------------
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(class_names))
    w_bar = 0.26

    b1 = ax.bar(x - w_bar, per_class_df["Baseline_v2_F1"], w_bar, label="Baseline ResNet18 v2 (Online 0.651)", color="#94a3b8")
    b2 = ax.bar(x, per_class_df["ResNet18_v4_F1"], w_bar, label="ResNet18 v4 + TenCrop", color="#3b82f6")
    b3 = ax.bar(x + w_bar, per_class_df["Ensemble_F1"], w_bar, label="Ensemble R18_v4 + DN121_v4 (Online 0.719)", color="#ef4444")

    ax.set_ylim(0.40, 0.98)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, fontsize=11, fontweight="bold")
    ax.set_ylabel("F1-Score", fontsize=11)
    ax.set_title("Per-Class F1-Score Comparison: Baseline v2 vs. Improved v4 & Heterogeneous Ensemble",
                 fontsize=12.5, fontweight="bold")
    ax.legend(fontsize=10, loc="upper left")

    for bars in [b1, b2, b3]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    fig3_path = os.path.join(Config.ANALYSIS_DIR, "fig3_per_class_f1_comparison.png")
    plt.savefig(fig3_path, bbox_inches="tight")
    plt.close()
    print(f"[图表导出 3/5] 6 类表情 F1 详细对比图已保存至: {fig3_path}")

    # -------------------- 图 4：验证集混淆矩阵热力图 (绝对数量 + 行归一化召回率百分比) --------------------
    cm_counts = confusion_matrix(val_targets, ens_preds)
    cm_norm = cm_counts.astype("float") / cm_counts.sum(axis=1, keepdims=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8))
    sns.heatmap(cm_counts, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=axes[0], cbar=False)
    axes[0].set_title("(a) Confusion Matrix (Sample Counts, N=2828)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Predicted Emotion")
    axes[0].set_ylabel("True Emotion")

    sns.heatmap(cm_norm, annot=True, fmt=".1f", cmap="YlGnBu",
                xticklabels=class_names, yticklabels=class_names, ax=axes[1], cbar=True)
    axes[1].set_title("(b) Normalized Confusion Matrix (Recall %)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Predicted Emotion")
    axes[1].set_ylabel("True Emotion")

    plt.tight_layout()
    fig4_path = os.path.join(Config.ANALYSIS_DIR, "fig4_ensemble_confusion_matrix.png")
    plt.savefig(fig4_path, bbox_inches="tight")
    plt.close()
    print(f"[图表导出 4/5] 混淆矩阵双视图已保存至: {fig4_path}")

    # -------------------- 图 5：训练集真实分布 vs 测试集预测分布对比图 (验证标签映射无误) --------------------
    if test_sub_df is not None:
        fig, ax = plt.subplots(figsize=(10, 5))
        # 将训练集 ImageFolder 标签转换为官方编号后统计比例
        train_official_labels = [Config.LABEL_MAPPING[y] for y in train_targets]
        train_counts = pd.Series(train_official_labels).value_counts().sort_index()
        train_pct = (train_counts / len(train_official_labels) * 100).values

        test_counts = test_sub_df["Emotion"].value_counts().sort_index()
        test_pct = (test_counts / len(test_sub_df) * 100).values

        official_names = [Config.OFFICIAL_CLASS_NAMES[i] for i in range(6)]
        x = np.arange(6)
        w = 0.35
        b_tr = ax.bar(x - w / 2, train_pct, w, label="Training Set True Distribution (N=28,275)", color="#6366f1")
        b_te = ax.bar(x + w / 2, test_pct, w, label="Test Set Predicted Distribution (N=7,065)", color="#f59e0b")

        ax.set_xticks(x)
        ax.set_xticklabels(official_names, fontsize=10.5, fontweight="bold")
        ax.set_ylabel("Percentage (%)", fontsize=11)
        ax.set_ylim(0, 32)
        ax.set_title("Class Distribution Alignment: Training Ground Truth vs. Test Predictions",
                     fontsize=12.5, fontweight="bold")
        ax.legend(fontsize=10)

        for bar, cnt in zip(b_tr, train_counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                    f"{bar.get_height():.1f}%\n({cnt})", ha="center", va="bottom", fontsize=8)
        for bar, cnt in zip(b_te, test_counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                    f"{bar.get_height():.1f}%\n({cnt})", ha="center", va="bottom", fontsize=8)

        plt.tight_layout()
        fig5_path = os.path.join(Config.ANALYSIS_DIR, "fig5_train_vs_test_distribution.png")
        plt.savefig(fig5_path, bbox_inches="tight")
        plt.close()
        print(f"[图表导出 5/5] 训练集与测试集类别分布对齐图已保存至: {fig5_path}")



# ==============================================================================
# 主流程入口
# ==============================================================================
def main():
    os.makedirs(Config.WORK_DIR, exist_ok=True)
    os.makedirs(Config.ANALYSIS_DIR, exist_ok=True)
    seed_everything(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用计算设备: {device}")

    # 1. 加载数据集并执行分层抽样 (90% 训练, 10% 验证, random_state=42)
    base_dataset = datasets.ImageFolder(root=Config.TRAIN_DIR)
    class_names = base_dataset.classes
    print("ImageFolder 类别顺序:", base_dataset.class_to_idx)

    indices = list(range(len(base_dataset)))
    all_labels = [base_dataset.targets[i] for i in indices]
    train_idx, val_idx = train_test_split(
        indices, test_size=0.1, random_state=42, stratify=all_labels
    )

    train_dataset = SubsetWithTransform(base_dataset, train_idx, transform=train_transform)
    val_dataset = SubsetWithTransform(base_dataset, val_idx, transform=val_single_transform)
    val_tencrop_dataset = SubsetWithTransform(base_dataset, val_idx, transform=tencrop_transform)

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    val_tencrop_loader = DataLoader(val_tencrop_dataset, batch_size=Config.TENCROP_BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    all_history = []

    # 2. 训练或加载模型 1：ResNet18_v4
    model_r18 = build_fer_resnet18(num_classes=6).to(device)
    if (not Config.FORCE_RETRAIN) and os.path.exists(Config.SAVE_R18_PATH) and os.path.exists(Config.R18_PROBS_PATH):
        print("检测到已保存的 ResNet18_v4 权重与预测概率，直接加载...")
        model_r18.load_state_dict(torch.load(Config.SAVE_R18_PATH, map_location=device))
        r18_val_probs = np.load(Config.R18_PROBS_PATH)
        val_targets_np = np.load(Config.VAL_TARGETS_PATH)
    else:
        seed_everything(Config.SEED)
        criterion_r18 = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)
        hist_r18 = train_model_with_history(
            model_r18, "ResNet18_v4", Config.SAVE_R18_PATH, criterion_r18,
            backbone_lr=1.5e-4, head_lr=5e-4, head_key="fc",
            train_loader=train_loader, val_loader=val_loader,
            val_dataset_len=len(val_dataset), device=device
        )
        all_history.extend(hist_r18)
        model_r18.load_state_dict(torch.load(Config.SAVE_R18_PATH, map_location=device))
        r18_val_probs, val_targets_np = evaluate_tencrop(model_r18, val_tencrop_loader, device, desc="ResNet18 TenCrop 验证")
        np.save(Config.R18_PROBS_PATH, r18_val_probs)
        np.save(Config.VAL_TARGETS_PATH, val_targets_np)

    # 3. 训练或加载模型 2：DenseNet121_v4 (带难样本类别权重)
    model_dn = build_fer_densenet121(num_classes=6).to(device)
    if (not Config.FORCE_RETRAIN) and os.path.exists(Config.SAVE_DN121_PATH) and os.path.exists(Config.DN121_PROBS_PATH):
        print("检测到已保存的 DenseNet121_v4 权重与预测概率，直接加载...")
        model_dn.load_state_dict(torch.load(Config.SAVE_DN121_PATH, map_location=device))
        dn_val_probs = np.load(Config.DN121_PROBS_PATH)
    else:
        seed_everything(Config.SEED + 100)
        class_weights = torch.tensor([1.08, 1.12, 0.88, 0.92, 1.08, 1.00], dtype=torch.float32, device=device)
        criterion_dn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=Config.LABEL_SMOOTHING)
        hist_dn = train_model_with_history(
            model_dn, "DenseNet121_v4", Config.SAVE_DN121_PATH, criterion_dn,
            backbone_lr=1.2e-4, head_lr=5e-4, head_key="classifier",
            train_loader=train_loader, val_loader=val_loader,
            val_dataset_len=len(val_dataset), device=device
        )
        all_history.extend(hist_dn)
        model_dn.load_state_dict(torch.load(Config.SAVE_DN121_PATH, map_location=device))
        dn_val_probs, _ = evaluate_tencrop(model_dn, val_tencrop_loader, device, desc="DenseNet121 TenCrop 验证")
        np.save(Config.DN121_PROBS_PATH, dn_val_probs)

    history_df = pd.DataFrame(all_history) if len(all_history) > 0 else (
        pd.read_csv(Config.HISTORY_CSV_PATH) if os.path.exists(Config.HISTORY_CSV_PATH) else None
    )

    # 4. 测试集双模型 TenCrop 集成推理与提交文件生成
    sub_template = pd.read_csv(Config.SUB_TEMPLATE_PATH)
    file_lookup = {}
    for root, _, files in os.walk(Config.TEST_DIR):
        for fname in files:
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                full_path = os.path.join(root, fname)
                stem = os.path.splitext(fname)[0]
                file_lookup[fname] = full_path
                file_lookup[stem] = full_path
                if stem.isdigit():
                    file_lookup[str(int(stem))] = full_path

    test_paths = [file_lookup[str(raw_id).strip()] for raw_id in sub_template["ID"]]
    test_dataset = TestTenCropDataset(test_paths, transform=tencrop_transform)
    test_loader = DataLoader(test_dataset, batch_size=Config.TENCROP_BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model_r18.eval()
    model_dn.eval()
    ensemble_raw_preds = []

    with torch.no_grad():
        for crops in tqdm(test_loader, desc="测试集双模型 TenCrop 集成预测"):
            bs, ncrops, c, h, w = crops.size()
            crops = crops.view(-1, c, h, w).to(device, non_blocking=True)
            with torch.amp.autocast("cuda"):
                probs_r18 = F.softmax(model_r18(crops), dim=1).view(bs, ncrops, -1).mean(dim=1)
                probs_dn = F.softmax(model_dn(crops), dim=1).view(bs, ncrops, -1).mean(dim=1)
                ens_probs = Config.W_R18 * probs_r18 + Config.W_DN121 * probs_dn
            ensemble_raw_preds.extend(ens_probs.argmax(dim=1).cpu().numpy())

    official_preds = [Config.LABEL_MAPPING[p] for p in ensemble_raw_preds]
    sub_df = pd.DataFrame({"ID": sub_template["ID"], "Emotion": official_preds})
    sub_df.to_csv(Config.OUTPUT_SUB_PATH, index=False)

    # 5. 导出指标表格与对比图
    export_analysis(
        history_df=history_df,
        r18_probs=r18_val_probs,
        dn121_probs=dn_val_probs,
        val_targets=val_targets_np,
        class_names=class_names,
        test_sub_df=sub_df,
        train_targets=all_labels
    )

    print(f"\n全部流程结束，结果保存在 {Config.ANALYSIS_DIR}")


if __name__ == "__main__":
    main()