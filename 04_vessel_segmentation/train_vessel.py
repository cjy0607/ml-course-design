#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""2D 眼底图像视网膜血管分割。

两阶段流程：
1. 预处理：绿色通道 CLAHE、Gamma 提亮、灰度 CLAHE 与视野掩码；
2. Stage-1 SA-ResUNet 分割，结合多尺度 D4 TTA 与迟滞阈值后处理；
3. Stage-2 CascadedRefinementUNet 以 512x512 官方 RLE 掩码为监督做级联精炼；
4. 五折结果融合并导出提交文件。
"""

import os
import random
import shutil
import time
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision


# ==============================================================================
# 1. 全局超参数与可复现随机种子配置
# ==============================================================================
class Config:
    DATA_ROOT = "/kaggle/input/competitions/2026task-4/segmentation"
    TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train/image")
    TRAIN_LBL_DIR = os.path.join(DATA_ROOT, "train/label")
    TEST_IMG_DIR = os.path.join(DATA_ROOT, "test/image")

    CKPT_DIR = "/kaggle/working/checkpoints"
    OOF_RAW_PATH = os.path.join(CKPT_DIR, "oof_predictions.npz")
    OOF_MS_PATH = os.path.join(CKPT_DIR, "oof_ms_predictions.npz")
    STAGE2_META_PATH = os.path.join(CKPT_DIR, "stage2_meta.npz")
    PREVIEW_FIG_PATH = "/kaggle/working/val_predictions_preview.png"

    OUTPUT_IMG_DIR = "/kaggle/working/image_stage2"
    SUBMISSION_PATH = "/kaggle/working/submission.csv"

    SEED = 20261107
    N_FOLDS = 5

    # 第一阶段 (SA-ResUNet) 训练超参数
    S1_EPOCHS = 90
    S1_BATCH_SIZE = 4
    S1_PATCH_SIZE = 384
    S1_PATCHES_PER_IMG = 8
    S1_LR = 8e-4
    S1_BASE_CHANNELS = 32

    # 第二阶段 (CascadedRefinementUNet) 训练超参数
    S2_EPOCHS = 55
    S2_BATCH_SIZE = 4
    S2_PATCH_SIZE = 384
    S2_PATCHES_PER_IMG = 10
    S2_LR = 6e-4
    S2_BASE_CHANNELS = 32

    WEIGHT_DECAY = 1e-4
    OFFICIAL_RLE_SIZE = (512, 512)
    MIN_COMPONENT_AREA = 12


def seed_everything(seed: int = Config.SEED):
    """固定全局随机种子，确保实验结果完全可复现"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==============================================================================
# 2. 眼科图像预处理、后处理算子与评测指标计算
# ==============================================================================
def extract_fov_mask(rgb_arr: np.ndarray) -> np.ndarray:
    """提取眼底圆形有效视野掩码 (FOV Mask)，消除眼球外围暗背景假阳性干扰"""
    gray = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2GRAY)
    fov = (gray > 12).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fov = cv2.morphologyEx(fov, cv2.MORPH_CLOSE, kernel)
    fov = cv2.erode(fov, kernel, iterations=1)
    return fov


def preprocess_fundus_image(rgb_arr: np.ndarray) -> np.ndarray:
    """
    构建 3 通道眼科对比度增强特征图 (H, W, 3):
    - Ch 0: 绿色通道 + 精细 CLAHE (clipLimit=2.2, grid=8x8)
    - Ch 1: 绿色通道 + Gamma(1.25) 暗区提亮 + 精细 CLAHE
    - Ch 2: 全局灰度 + 宽域 CLAHE (clipLimit=1.5, grid=12x12)
    """
    green = rgb_arr[:, :, 1]
    gray = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2GRAY)

    clahe_fine = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    clahe_global = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(12, 12))

    ch0 = clahe_fine.apply(green)
    inv_gamma = 1.0 / 1.25
    lut = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    ch1 = clahe_fine.apply(cv2.LUT(green, lut))
    ch2 = clahe_global.apply(gray)

    enhanced = np.stack([ch0, ch1, ch2], axis=-1).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.485, 0.456], dtype=np.float32)
    std = np.array([0.229, 0.229, 0.224], dtype=np.float32)
    return (enhanced - mean) / std


def simulate_official_rle_binary(gray_pil: Image.Image) -> np.ndarray:
    """严格复现官方 segmentation_to_csv.py 中的 512x512 双线性插值与非零二值化逻辑"""
    img_512 = gray_pil.convert("L").resize(Config.OFFICIAL_RLE_SIZE, Image.Resampling.BILINEAR)
    tensor = torchvision.transforms.ToTensor()(img_512)
    tensor[tensor > 0] = 1
    return tensor.numpy()[0].astype(np.uint8)


def compute_segmentation_metrics(pred_bin: np.ndarray, gt_bin: np.ndarray, target_val: int = 1, eps: float = 1e-7) -> dict:
    """计算分割评价指标：Dice 系数、IoU、Precision 与 Recall"""
    p = (pred_bin == target_val).astype(np.float32)
    g = (gt_bin == target_val).astype(np.float32)
    tp = float(np.sum(p * g))
    fp = float(np.sum(p * (1.0 - g)))
    fn = float(np.sum((1.0 - p) * g))

    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    return {"dice": dice, "iou": iou, "precision": precision, "recall": recall}


def postprocess_vessel_mask(vessel_bin: np.ndarray, min_area: int = 12) -> np.ndarray:
    """连通域面积滤波：剔除面积小于 min_area 的孤立假阳性噪点"""
    if min_area <= 0:
        return vessel_bin
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(vessel_bin.astype(np.uint8), connectivity=8)
    cleaned = np.zeros_like(vessel_bin, dtype=np.uint8)
    for idx in range(1, num_labels):
        if stats[idx, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == idx] = 1
    return cleaned


def hysteresis_threshold(prob: np.ndarray, th_low: float, th_high: float, min_area: int = 8) -> np.ndarray:
    """
    双阈值迟滞连接 (Hysteresis Thresholding)：
    保留所有 >= th_high 的高置信度主干血管，并召回与之在 8 邻域连通的 >= th_low 的微细毛细血管分支。
    """
    low_mask = (prob >= th_low).astype(np.uint8)
    high_mask = (prob >= th_high).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(low_mask, connectivity=8)
    out = np.zeros_like(low_mask, dtype=np.uint8)
    valid_labels = np.unique(labels[high_mask == 1])
    for lbl in valid_labels:
        if lbl == 0:
            continue
        if stats[lbl, cv2.CC_STAT_AREA] >= min_area:
            out[labels == lbl] = 1
    return out


def load_train_dataset(train_img_dir: str, train_lbl_dir: str):
    """加载 20 张训练集图像与标签，自动校验前景极性并生成 512x512 官方 RLE 真值"""
    train_files = sorted(
        [f for f in os.listdir(train_img_dir) if not f.startswith(".")],
        key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else x
    )

    train_data, zero_ratios = [], []
    for fname in train_files:
        rgb = np.array(Image.open(os.path.join(train_img_dir, fname)).convert("RGB"))
        lbl_pil = Image.open(os.path.join(train_lbl_dir, fname)).convert("L")
        lbl_raw = np.array(lbl_pil)
        zero_ratios.append(np.mean(lbl_raw < 128))

        train_data.append({
            "id": os.path.splitext(fname)[0],
            "filename": fname,
            "rgb": rgb,
            "enhanced": preprocess_fundus_image(rgb),
            "fov": extract_fov_mask(rgb),
            "lbl_raw": lbl_raw,
            "official_gt_512": simulate_official_rle_binary(lbl_pil),
        })

    mean_zero_ratio = float(np.mean(zero_ratios))
    vessel_is_zero_in_raw = (mean_zero_ratio < 0.5)
    print(f"[数据检查] 训练集共 {len(train_data)} 张, 标签中 0 像素平均占比: {mean_zero_ratio:.4f}")

    for item in train_data:
        if vessel_is_zero_in_raw:
            item["vessel_mask"] = (item["lbl_raw"] < 128).astype(np.float32)
        else:
            item["vessel_mask"] = (item["lbl_raw"] >= 128).astype(np.float32)

    return train_data


# ==============================================================================
# 3. 第一阶段模型定义：SA-ResUNet 与混合损失函数
# ==============================================================================
class FundusPatchDataset(Dataset):
    """第一阶段血管中心引导随机切块与几何/光度数据增强 Dataset"""
    def __init__(self, items, patch_size=384, patches_per_img=8, is_train=True):
        self.items = items
        self.patch_size = patch_size
        self.patches_per_img = patches_per_img if is_train else 1
        self.is_train = is_train

    def __len__(self):
        return len(self.items) * self.patches_per_img

    def __getitem__(self, idx):
        item = self.items[idx % len(self.items)]
        img = item["enhanced"].copy()
        mask = item["vessel_mask"].copy()
        h, w, _ = img.shape

        if self.is_train:
            if random.random() < 0.5:
                scale = random.uniform(0.85, 1.15)
                nh, nw = max(int(h * scale), self.patch_size), max(int(w * scale), self.patch_size)
                img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
                mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
                h, w = nh, nw

            ps = self.patch_size
            vessel_yx = np.argwhere(mask > 0.5)
            if len(vessel_yx) > 0 and random.random() < 0.6:
                cy, cx = vessel_yx[random.randint(0, len(vessel_yx) - 1)]
                y0 = int(np.clip(cy - ps // 2 + random.randint(-48, 48), 0, max(0, h - ps)))
                x0 = int(np.clip(cx - ps // 2 + random.randint(-48, 48), 0, max(0, w - ps)))
            else:
                y0 = random.randint(0, max(0, h - ps))
                x0 = random.randint(0, max(0, w - ps))
            img = img[y0:y0 + ps, x0:x0 + ps]
            mask = mask[y0:y0 + ps, x0:x0 + ps]

            k = random.randint(0, 3)
            if k > 0:
                img = np.rot90(img, k).copy()
                mask = np.rot90(mask, k).copy()
            if random.random() < 0.5:
                img = np.fliplr(img).copy()
                mask = np.fliplr(mask).copy()
            if random.random() < 0.5:
                img = np.flipud(img).copy()
                mask = np.flipud(mask).copy()

            if random.random() < 0.5:
                angle = random.uniform(-25, 25)
                M = cv2.getRotationMatrix2D((ps / 2, ps / 2), angle, 1.0)
                img = cv2.warpAffine(img, M, (ps, ps), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
                mask = cv2.warpAffine(mask, M, (ps, ps), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)

            if random.random() < 0.5:
                img = img * random.uniform(0.9, 1.1) + random.uniform(-0.1, 0.1)
            if random.random() < 0.3:
                img = img + np.random.normal(0, 0.03, img.shape).astype(np.float32)

        return (
            torch.from_numpy(np.transpose(img, (2, 0, 1)).astype(np.float32)),
            torch.from_numpy(mask[None, :, :].astype(np.float32)),
        )


class ChannelSpatialAttention(nn.Module):
    """CBAM 通道与空间双重注意力模块"""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        ca = torch.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))
        x = x * ca
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa = self.spatial(torch.cat([avg_out, max_out], dim=1))
        return x * sa


class ResConvBlock(nn.Module):
    """融合 CBAM 注意力机制的残差双卷积块"""
    def __init__(self, in_ch: int, out_ch: int, dropout_p: float = 0.0, use_att: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)
        self.drop = nn.Dropout2d(p=dropout_p) if dropout_p > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = (
            nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False), nn.BatchNorm2d(out_ch))
            if in_ch != out_ch else nn.Identity()
        )
        self.att = ChannelSpatialAttention(out_ch) if use_att else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        out = self.att(out)
        return self.act(out + residual)


class AttentionGate(nn.Module):
    """门控注意力跳跃连接模块 (Attention Gate)"""
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, kernel_size=1, bias=True), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, kernel_size=1, bias=True), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(
            nn.SiLU(inplace=True),
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, g, x):
        return x * self.psi(self.W_g(g) + self.W_x(x))


class SAResUNet(nn.Module):
    """第一阶段主力分割网络：空间-通道注意力残差 U-Net (含三尺度深监督)"""
    def __init__(self, in_channels: int = 3, base_c: int = 32):
        super().__init__()
        c1, c2, c3, c4, c5 = base_c, base_c * 2, base_c * 4, base_c * 8, base_c * 12

        self.enc1 = ResConvBlock(in_channels, c1, dropout_p=0.05)
        self.enc2 = ResConvBlock(c1, c2, dropout_p=0.05)
        self.enc3 = ResConvBlock(c2, c3, dropout_p=0.10)
        self.enc4 = ResConvBlock(c3, c4, dropout_p=0.15)
        self.pool = nn.MaxPool2d(2, 2)

        self.bottleneck = ResConvBlock(c4, c5, dropout_p=0.20)

        self.ag4 = AttentionGate(c5, c4, c4 // 2)
        self.dec4 = ResConvBlock(c5 + c4, c4, dropout_p=0.10)

        self.ag3 = AttentionGate(c4, c3, c3 // 2)
        self.dec3 = ResConvBlock(c4 + c3, c3, dropout_p=0.05)

        self.ag2 = AttentionGate(c3, c2, c2 // 2)
        self.dec2 = ResConvBlock(c3 + c2, c2, dropout_p=0.05)

        self.ag1 = AttentionGate(c2, c1, c1 // 2)
        self.dec1 = ResConvBlock(c2 + c1, c1, dropout_p=0.0)

        self.head1 = nn.Conv2d(c1, 1, kernel_size=1)
        self.head2 = nn.Conv2d(c2, 1, kernel_size=1)
        self.head3 = nn.Conv2d(c3, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        u4 = F.interpolate(b, size=e4.shape[2:], mode="bilinear", align_corners=False)
        d4 = self.dec4(torch.cat([u4, self.ag4(u4, e4)], dim=1))

        u3 = F.interpolate(d4, size=e3.shape[2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([u3, self.ag3(u3, e3)], dim=1))

        u2 = F.interpolate(d3, size=e2.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([u2, self.ag2(u2, e2)], dim=1))

        u1 = F.interpolate(d2, size=e1.shape[2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([u1, self.ag1(u1, e1)], dim=1))

        out1 = self.head1(d1)
        if self.training:
            out2 = F.interpolate(self.head2(d2), size=x.shape[2:], mode="bilinear", align_corners=False)
            out3 = F.interpolate(self.head3(d3), size=x.shape[2:], mode="bilinear", align_corners=False)
            return out1, out2, out3
        return out1


class VesselComboLoss(nn.Module):
    """第一阶段多尺度复合损失：0.35 BCE + 0.45 Soft Dice + 0.20 Tversky (alpha=0.35, beta=0.65)"""
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def single_scale_loss(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)

        inter = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice_loss = 1.0 - ((2.0 * inter + self.smooth) / (union + self.smooth)).mean()

        fp = (probs * (1.0 - targets)).sum(dim=(2, 3))
        fn = ((1.0 - probs) * targets).sum(dim=(2, 3))
        tversky = (inter + self.smooth) / (inter + 0.35 * fp + 0.65 * fn + self.smooth)
        tversky_loss = (1.0 - tversky).mean()

        return 0.35 * bce_loss + 0.45 * dice_loss + 0.20 * tversky_loss

    def forward(self, outputs, targets):
        if isinstance(outputs, tuple):
            out1, out2, out3 = outputs
            return (
                0.70 * self.single_scale_loss(out1, targets)
                + 0.20 * self.single_scale_loss(out2, targets)
                + 0.10 * self.single_scale_loss(out3, targets)
            )
        return self.single_scale_loss(outputs, targets)


# ==============================================================================
# 4. 第二阶段模型定义：5通道 Auto-Context 级联精炼网络 (CascadedRefinementUNet)
# ==============================================================================
def build_stage2_5ch_input(enhanced_584: np.ndarray, fov_584: np.ndarray, stage1_prob_584: np.ndarray, th_hard: float = 0.37):
    """
    构建 (512, 512, 5) 的第二阶段级联输入特征：
    - Ch 0~2: 512x512 眼科增强图像
    - Ch 3:   第一阶段预测概率图的双线性插值软特征
    - Ch 4:   第一阶段经双阈值迟滞二值化及官方 BILINEAR->(>0) 转换后的同构膨胀先验特征
    """
    img_512 = cv2.resize(enhanced_584, (512, 512), interpolation=cv2.INTER_LINEAR)
    fov_512 = cv2.resize(fov_584.astype(np.float32), (512, 512), interpolation=cv2.INTER_NEAREST)

    p_u8 = np.clip(np.round(stage1_prob_584 * 255.0), 0, 255).astype(np.uint8)
    pil_soft_512 = Image.fromarray(p_u8, mode="L").resize((512, 512), Image.Resampling.BILINEAR)
    ch3_soft = (np.array(pil_soft_512, dtype=np.float32) / 255.0) * fov_512
    ch3_norm = (ch3_soft - 0.5) * 2.0

    v_584 = hysteresis_threshold(stage1_prob_584, th_low=th_hard, th_high=max(th_hard + 0.08, 0.45), min_area=10)
    pil_hard_584 = Image.fromarray((v_584 * 255).astype(np.uint8), mode="L")
    ch4_hard = simulate_official_rle_binary(pil_hard_584).astype(np.float32) * fov_512
    ch4_norm = (ch4_hard - 0.5) * 2.0

    feat_5ch = np.concatenate([img_512, ch3_norm[:, :, None], ch4_norm[:, :, None]], axis=-1)
    return feat_5ch.astype(np.float32), fov_512, ch3_soft, ch4_hard


class Stage2DirectRLEDataset(Dataset):
    """第二阶段直接拟合 512x512 官方 RLE 真值掩码的 Dataset"""
    def __init__(self, items, patch_size=384, patches_per_img=10, is_train=True):
        self.items = items
        self.patch_size = patch_size
        self.patches_per_img = patches_per_img if is_train else 1
        self.is_train = is_train

    def __len__(self):
        return len(self.items) * self.patches_per_img

    def __getitem__(self, idx):
        item = self.items[idx % len(self.items)]
        feat = item["feat_5ch"].copy()
        target = item["official_gt_512"].copy().astype(np.float32)
        h, w, _ = feat.shape

        if self.is_train:
            ps = self.patch_size
            vessel_yx = np.argwhere(target > 0.5)
            if len(vessel_yx) > 0 and random.random() < 0.7:
                cy, cx = vessel_yx[random.randint(0, len(vessel_yx) - 1)]
                y0 = int(np.clip(cy - ps // 2 + random.randint(-48, 48), 0, h - ps))
                x0 = int(np.clip(cx - ps // 2 + random.randint(-48, 48), 0, w - ps))
            else:
                y0 = random.randint(0, h - ps)
                x0 = random.randint(0, w - ps)

            feat = feat[y0:y0 + ps, x0:x0 + ps]
            target = target[y0:y0 + ps, x0:x0 + ps]

            k = random.randint(0, 3)
            if k > 0:
                feat = np.rot90(feat, k).copy()
                target = np.rot90(target, k).copy()
            if random.random() < 0.5:
                feat = np.fliplr(feat).copy()
                target = np.fliplr(target).copy()
            if random.random() < 0.5:
                feat = np.flipud(feat).copy()
                target = np.flipud(target).copy()

            if random.random() < 0.4:
                feat[:, :, :3] = feat[:, :, :3] * random.uniform(0.92, 1.08) + random.uniform(-0.05, 0.05)
            if random.random() < 0.4:
                feat[:, :, 3:] = feat[:, :, 3:] + np.random.normal(0, 0.10, feat[:, :, 3:].shape).astype(np.float32)

        return (
            torch.from_numpy(np.transpose(feat, (2, 0, 1)).astype(np.float32)),
            torch.from_numpy(target[None, :, :].astype(np.float32)),
        )


class DilatedContextBlock(nn.Module):
    """多分支空洞卷积上下文模块 (dilation=1, 2, 4)，用于修复跨度较大的毛细血管断裂"""
    def __init__(self, channels: int):
        super().__init__()
        mid = channels // 2
        self.b1 = nn.Sequential(nn.Conv2d(channels, mid, 3, padding=1, dilation=1, bias=False), nn.BatchNorm2d(mid), nn.SiLU(inplace=True))
        self.b2 = nn.Sequential(nn.Conv2d(channels, mid, 3, padding=2, dilation=2, bias=False), nn.BatchNorm2d(mid), nn.SiLU(inplace=True))
        self.b3 = nn.Sequential(nn.Conv2d(channels, mid, 3, padding=4, dilation=4, bias=False), nn.BatchNorm2d(mid), nn.SiLU(inplace=True))
        self.fuse = nn.Sequential(nn.Conv2d(mid * 3, channels, 1, bias=False), nn.BatchNorm2d(channels), nn.SiLU(inplace=True))

    def forward(self, x):
        return x + self.fuse(torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1))


class ConvResBlock(nn.Module):
    """标准残差双卷积模块"""
    def __init__(self, in_ch: int, out_ch: int, drop_p: float = 0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.Dropout2d(drop_p) if drop_p > 0 else nn.Identity(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.short = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch)) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.conv(x) + self.short(x))


class CascadedRefinementUNet(nn.Module):
    """第二阶段级联精炼 U-Net"""
    def __init__(self, in_channels: int = 5, base_c: int = 32):
        super().__init__()
        c1, c2, c3, c4 = base_c, base_c * 2, base_c * 4, base_c * 8
        self.enc1 = ConvResBlock(in_channels, c1, 0.05)
        self.enc2 = ConvResBlock(c1, c2, 0.05)
        self.enc3 = ConvResBlock(c2, c3, 0.10)
        self.pool = nn.MaxPool2d(2, 2)

        self.bottleneck = nn.Sequential(
            ConvResBlock(c3, c4, 0.15),
            DilatedContextBlock(c4),
        )

        self.dec3 = ConvResBlock(c4 + c3, c3, 0.10)
        self.dec2 = ConvResBlock(c3 + c2, c2, 0.05)
        self.dec1 = ConvResBlock(c2 + c1, c1, 0.0)

        self.head1 = nn.Conv2d(c1, 1, 1)
        self.head2 = nn.Conv2d(c2, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        u3 = F.interpolate(b, size=e3.shape[2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))

        u2 = F.interpolate(d3, size=e2.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = F.interpolate(d2, size=e1.shape[2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        out1 = self.head1(d1)
        if self.training:
            out2 = F.interpolate(self.head2(d2), size=x.shape[2:], mode="bilinear", align_corners=False)
            return out1, out2
        return out1


class DirectRLELoss(nn.Module):
    """第二阶段直接针对 512x512 官方 RLE 掩码的损失函数 (0.25 BCE + 0.75 Soft Dice)"""
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        if isinstance(logits, tuple):
            return 0.80 * self.forward(logits[0], targets) + 0.20 * self.forward(logits[1], targets)
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        inter = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice_loss = 1.0 - ((2.0 * inter + self.smooth) / (union + self.smooth)).mean()
        return 0.25 * bce + 0.75 * dice_loss


# ==============================================================================
# 5. 测试时增强 (TTA) 推理算子与验证集可视化
# ==============================================================================
@torch.no_grad()
def predict_d4_tensor(model: nn.Module, x_tensor: torch.Tensor) -> np.ndarray:
    """对输入 Tensor 执行二面体群 D4 (4个旋转角度 x 水平翻转 = 8倍) TTA 推理"""
    _, _, h, w = x_tensor.shape
    p_sum = np.zeros((h, w), dtype=np.float32)
    for k in range(4):
        for flip in (False, True):
            xa = torch.rot90(x_tensor, k, dims=[2, 3])
            if flip:
                xa = torch.flip(xa, dims=[3])
            pa = torch.sigmoid(model(xa))
            if flip:
                pa = torch.flip(pa, dims=[3])
            pa = torch.rot90(pa, -k, dims=[2, 3])
            p_sum += pa[0, 0].cpu().numpy()
    return p_sum / 8.0


@torch.no_grad()
def predict_full_image_tta(model: nn.Module, enhanced_img: np.ndarray, fov_mask: np.ndarray, device, use_tta: bool = True) -> np.ndarray:
    """单尺度全图推理 (自动反射 Padding 至 16 的整数倍)"""
    model.eval()
    h, w, _ = enhanced_img.shape
    pad_h, pad_w = (16 - h % 16) % 16, (16 - w % 16) % 16
    x_np = np.pad(enhanced_img, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    x_base = torch.from_numpy(np.transpose(x_np, (2, 0, 1)).astype(np.float32)).unsqueeze(0).to(device)

    if not use_tta:
        prob = torch.sigmoid(model(x_base))[0, 0].cpu().numpy()[:h, :w]
        return prob * fov_mask

    return predict_d4_tensor(model, x_base)[:h, :w] * fov_mask


@torch.no_grad()
def predict_ms_sliding(model: nn.Module, enhanced_img: np.ndarray, fov_mask: np.ndarray, device) -> np.ndarray:
    """
    三尺度融合 TTA 推理：
    - 0.45 权重：384x384 高斯加权滑窗 TTA（匹配训练 Patch 感受野，避免四周黑边稀释 CBAM 通道池化）
    - 0.35 权重：584x565 原尺寸全图 TTA
    - 0.20 权重：1.15倍放大尺度 (672x656) 全图 TTA（增强 1~2 像素宽毛细血管响应）
    """
    model.eval()
    h, w, _ = enhanced_img.shape

    ph, pw = (16 - h % 16) % 16, (16 - w % 16) % 16
    x_pad = np.pad(enhanced_img, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    t_full = torch.from_numpy(np.transpose(x_pad, (2, 0, 1)).astype(np.float32)).unsqueeze(0).to(device)
    p_full = predict_d4_tensor(model, t_full)[:h, :w]

    ps = 384
    ys, xs = [0, (h - ps) // 2, h - ps], [0, (w - ps) // 2, w - ps]
    yy, xx = np.meshgrid(np.linspace(-1, 1, ps), np.linspace(-1, 1, ps), indexing="ij")
    gw = np.exp(-0.5 * (yy ** 2 + xx ** 2) / (0.6 ** 2)).astype(np.float32)
    p_sw, w_sw = np.zeros((h, w), dtype=np.float32), np.zeros((h, w), dtype=np.float32)
    for y0 in ys:
        for x0 in xs:
            patch = enhanced_img[y0:y0 + ps, x0:x0 + ps]
            tp = torch.from_numpy(np.transpose(patch, (2, 0, 1)).astype(np.float32)).unsqueeze(0).to(device)
            p_sw[y0:y0 + ps, x0:x0 + ps] += predict_d4_tensor(model, tp) * gw
            w_sw[y0:y0 + ps, x0:x0 + ps] += gw
    p_sw /= np.maximum(w_sw, 1e-6)

    img_up = cv2.resize(enhanced_img, (656, 672), interpolation=cv2.INTER_CUBIC)
    t_up = torch.from_numpy(np.transpose(img_up, (2, 0, 1)).astype(np.float32)).unsqueeze(0).to(device)
    p_up = cv2.resize(predict_d4_tensor(model, t_up), (w, h), interpolation=cv2.INTER_LINEAR)

    return (0.45 * p_sw + 0.35 * p_full + 0.20 * p_up) * fov_mask


@torch.no_grad()
def predict_stage2_d4_tta(model: nn.Module, feat_5ch: np.ndarray, fov_512: np.ndarray, device) -> np.ndarray:
    """第二阶段 512x512 级联精炼网络的 8倍 D4 TTA 推理"""
    model.eval()
    x_base = torch.from_numpy(np.transpose(feat_5ch, (2, 0, 1)).astype(np.float32)).unsqueeze(0).to(device)
    return predict_d4_tensor(model, x_base) * fov_512


def save_validation_visualization(train_data, oof_prob_maps, best_th: float, save_path: str):
    """生成验证集四列对比分析图：原图 | 真值标签 | 预测掩码 | TP(绿)/FP(红)/FN(蓝) 误差分布图"""
    n_show = min(4, len(train_data))
    fig, axes = plt.subplots(n_show, 4, figsize=(16, 4 * n_show))
    if n_show == 1:
        axes = np.expand_dims(axes, axis=0)

    for row in range(n_show):
        item = train_data[row]
        prob = oof_prob_maps[row]
        gt_v = item["vessel_mask"].astype(np.uint8)
        pred_v = postprocess_vessel_mask((prob >= best_th).astype(np.uint8), Config.MIN_COMPONENT_AREA)
        m = compute_segmentation_metrics(pred_v, gt_v, target_val=1)

        err_map = np.zeros((*gt_v.shape, 3), dtype=np.uint8)
        err_map[(pred_v == 1) & (gt_v == 1)] = [0, 230, 118]    # TP: 绿色
        err_map[(pred_v == 1) & (gt_v == 0)] = [255, 82, 82]    # FP: 红色
        err_map[(pred_v == 0) & (gt_v == 1)] = [68, 138, 255]   # FN: 蓝色

        axes[row, 0].imshow(item["rgb"])
        axes[row, 0].set_title(f"Image ID: {item['id']} (Original)")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(np.where(gt_v == 1, 0, 255), cmap="gray", vmin=0, vmax=255)
        axes[row, 1].set_title("Ground Truth (Vessel=0)")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(np.where(pred_v == 1, 0, 255), cmap="gray", vmin=0, vmax=255)
        axes[row, 2].set_title(f"OOF Pred (Vessel Dice={m['dice']:.4f})")
        axes[row, 2].axis("off")

        axes[row, 3].imshow(err_map)
        axes[row, 3].set_title(f"Green=TP | Red=FP | Blue=FN (Rec={m['recall']:.3f})")
        axes[row, 3].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ==============================================================================
# 6. 主流程阶段一：SA-ResUNet 五折交叉验证与多尺度滑窗 OOF 评估
# ==============================================================================
def run_stage1_training_and_ms_oof(train_data, fold_splits, indices, device):
    os.makedirs(Config.CKPT_DIR, exist_ok=True)
    n_samples = len(train_data)
    oof_raw_probs = [None] * n_samples
    oof_ms_probs = [None] * n_samples
    criterion = VesselComboLoss()

    for fold in range(Config.N_FOLDS):
        val_idx = fold_splits[fold]
        train_idx = np.setdiff1d(indices, val_idx)
        train_items = [train_data[i] for i in train_idx]
        val_items = [train_data[i] for i in val_idx]

        train_loader = DataLoader(
            FundusPatchDataset(train_items, patch_size=Config.S1_PATCH_SIZE, patches_per_img=Config.S1_PATCHES_PER_IMG, is_train=True),
            batch_size=Config.S1_BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )

        model = SAResUNet(in_channels=3, base_c=Config.S1_BASE_CHANNELS).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=Config.S1_LR, weight_decay=Config.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.S1_EPOCHS, eta_min=1e-5)
        scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

        best_val_dice = 0.0
        best_ckpt_path = os.path.join(Config.CKPT_DIR, f"sa_resunet_fold{fold + 1}.pth")
        t0 = time.time()

        val_ids_str = ", ".join([train_data[i]["id"] for i in val_idx])
        print(f"\n>>> [Stage 1] 训练 Fold {fold + 1}/{Config.N_FOLDS} | 验证集 ID: [{val_ids_str}] <<<")

        for epoch in range(1, Config.S1_EPOCHS + 1):
            model.train()
            running_loss = 0.0
            for imgs, masks in train_loader:
                imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                    loss = criterion(model(imgs), masks)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item()

            scheduler.step()
            avg_loss = running_loss / max(len(train_loader), 1)

            if epoch % 5 == 0 or epoch >= Config.S1_EPOCHS - 10 or epoch == 1:
                val_dices = [
                    compute_segmentation_metrics(
                        (predict_full_image_tta(model, item["enhanced"], item["fov"], device, use_tta=False) >= 0.5).astype(np.uint8),
                        item["vessel_mask"].astype(np.uint8),
                        target_val=1,
                    )["dice"]
                    for item in val_items
                ]
                mean_val_dice = float(np.mean(val_dices))
                if mean_val_dice > best_val_dice:
                    best_val_dice = mean_val_dice
                    torch.save(model.state_dict(), best_ckpt_path)

                if epoch % 15 == 0 or epoch == Config.S1_EPOCHS or epoch == 1:
                    print(
                        f"  [Fold {fold + 1} | Epoch {epoch:03d}/{Config.S1_EPOCHS:03d}] "
                        f"Loss: {avg_loss:.4f} | Val Vessel Dice: {mean_val_dice:.4f} "
                        f"(Best: {best_val_dice:.4f}) | Time: {time.time() - t0:.1f}s"
                    )

        # 加载当前折最优模型，分别提取单尺度 TTA 与三尺度滑窗 TTA 的验证集 OOF 预测
        model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
        for i in val_idx:
            oof_raw_probs[i] = predict_full_image_tta(model, train_data[i]["enhanced"], train_data[i]["fov"], device, use_tta=True)
            oof_ms_probs[i] = predict_ms_sliding(model, train_data[i]["enhanced"], train_data[i]["fov"], device)

    np.savez_compressed(Config.OOF_RAW_PATH, ids=[it["id"] for it in train_data], probs=np.array(oof_raw_probs, dtype=np.float32))
    np.savez_compressed(Config.OOF_MS_PATH, probs=np.array(oof_ms_probs, dtype=np.float32))

    # 评估原图 (584x565) 最优阈值并生成可视化误差图
    best_th_raw, best_raw_dice = 0.465, -1.0
    for th in np.linspace(0.28, 0.68, 81):
        dices = [
            compute_segmentation_metrics(
                postprocess_vessel_mask((oof_raw_probs[i] >= th).astype(np.uint8), Config.MIN_COMPONENT_AREA),
                train_data[i]["vessel_mask"].astype(np.uint8),
                target_val=1,
            )["dice"]
            for i in range(n_samples)
        ]
        if np.mean(dices) > best_raw_dice:
            best_raw_dice, best_th_raw = float(np.mean(dices)), float(th)

    save_validation_visualization(train_data, oof_raw_probs, best_th_raw, Config.PREVIEW_FIG_PATH)
    print(f"\n[Stage 1 总结] 原图 (584x565) 最优阈值={best_th_raw:.4f} -> 5折 OOF 纯血管前景 Dice: {best_raw_dice:.4f}")
    return oof_ms_probs


# ==============================================================================
# 7. 主流程阶段二：CascadedRefinementUNet 五折训练与两阶段融合参数寻优
# ==============================================================================
def run_stage2_cascaded_refinement(train_data, oof_ms_probs, fold_splits, indices, device):
    items = []
    for i, item in enumerate(train_data):
        feat_5ch, fov_512, ch3_soft, ch4_hard = build_stage2_5ch_input(item["enhanced"], item["fov"], oof_ms_probs[i])
        items.append({
            "id": item["id"],
            "feat_5ch": feat_5ch,
            "fov_512": fov_512,
            "ch3_soft": ch3_soft,
            "ch4_hard": ch4_hard,
            "official_gt_512": item["official_gt_512"],
            "stage1_prob_584": oof_ms_probs[i],
        })

    # 搜索第一阶段多尺度 OOF 在双阈值迟滞连接 + 原图二值同构膨胀下的最优参数
    best_s1_dice, best_s1_params = 0.0, (0.37, 0.49, 8)
    for th_l in np.linspace(0.26, 0.40, 15):
        for th_h in [th_l, th_l + 0.06, th_l + 0.12]:
            dices = []
            for it in items:
                v_584 = hysteresis_threshold(it["stage1_prob_584"], th_low=th_l, th_high=th_h, min_area=8)
                p_rle = simulate_official_rle_binary(Image.fromarray((v_584 * 255).astype(np.uint8), mode="L"))
                dices.append(compute_segmentation_metrics(p_rle, it["official_gt_512"], target_val=1)["dice"])
            if np.mean(dices) > best_s1_dice:
                best_s1_dice = float(np.mean(dices))
                best_s1_params = (float(th_l), float(th_h), 8)

    print(f"[Stage 1 同构膨胀 + Hysteresis 寻优] 参数={best_s1_params} -> OOF 官方 RLE Dice: {best_s1_dice:.4f}")

    # 使用最优一阶段参数更新第 5 通道先验特征
    s1_rle_preds = []
    for it in items:
        v_584 = hysteresis_threshold(it["stage1_prob_584"], best_s1_params[0], best_s1_params[1], best_s1_params[2])
        p_rle = simulate_official_rle_binary(Image.fromarray((v_584 * 255).astype(np.uint8), mode="L")).astype(np.float32) * it["fov_512"]
        s1_rle_preds.append(p_rle)
        it["feat_5ch"][:, :, 4] = (p_rle - 0.5) * 2.0

    stage2_oof_probs = [None] * len(items)
    criterion = DirectRLELoss()

    for fold in range(Config.N_FOLDS):
        val_idx = fold_splits[fold]
        train_idx = np.setdiff1d(indices, val_idx)
        train_ds = Stage2DirectRLEDataset([items[i] for i in train_idx], patch_size=Config.S2_PATCH_SIZE, patches_per_img=Config.S2_PATCHES_PER_IMG, is_train=True)
        train_loader = DataLoader(train_ds, batch_size=Config.S2_BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

        model = CascadedRefinementUNet(in_channels=5, base_c=Config.S2_BASE_CHANNELS).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=Config.S2_LR, weight_decay=Config.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.S2_EPOCHS, eta_min=1e-5)
        scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

        best_fold_dice = 0.0
        ckpt_path = os.path.join(Config.CKPT_DIR, f"stage2_refine_fold{fold + 1}.pth")

        for epoch in range(1, Config.S2_EPOCHS + 1):
            model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                    loss = criterion(model(x), y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            scheduler.step()

            if epoch % 5 == 0 or epoch == Config.S2_EPOCHS:
                val_dices = [
                    compute_segmentation_metrics(
                        (predict_stage2_d4_tta(model, items[idx]["feat_5ch"], items[idx]["fov_512"], device) >= 0.5).astype(np.uint8),
                        items[idx]["official_gt_512"],
                        target_val=1,
                    )["dice"]
                    for idx in val_idx
                ]
                m_dice = float(np.mean(val_dices))
                if m_dice > best_fold_dice:
                    best_fold_dice = m_dice
                    torch.save(model.state_dict(), ckpt_path)

        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        for idx in val_idx:
            stage2_oof_probs[idx] = predict_stage2_d4_tta(model, items[idx]["feat_5ch"], items[idx]["fov_512"], device)
        print(f"  -> [Stage 2] Fold {fold + 1}/5 完成 | 验证集 512x512 官方 RLE Dice: {best_fold_dice:.4f}")

    # 网格搜索一、二阶段最优融合权重与迟滞阈值
    best_blend_dice, best_w, best_th = best_s1_dice, 0.65, 0.36
    for w_s2 in np.linspace(0.0, 1.0, 21):
        for th in np.linspace(0.35, 0.65, 31):
            dices = []
            for i, it in enumerate(items):
                p_blend = w_s2 * stage2_oof_probs[i] + (1.0 - w_s2) * s1_rle_preds[i]
                pred_bin = hysteresis_threshold(p_blend, th_low=th, th_high=min(th + 0.05, 0.85), min_area=8)
                dices.append(compute_segmentation_metrics(pred_bin, it["official_gt_512"], target_val=1)["dice"])
            if np.mean(dices) > best_blend_dice:
                best_blend_dice = float(np.mean(dices))
                best_w = float(w_s2)
                best_th = float(th)

    print("==========================================================================")
    print(f"第一阶段 OOF 官方 RLE Dice:                 {best_s1_dice:.4f}")
    print(f"一+二阶段级联融合后 OOF 官方 RLE Dice:       {best_blend_dice:.4f} (w_s2={best_w:.2f}, th={best_th:.3f})")
    print("==========================================================================")

    np.savez_compressed(
        Config.STAGE2_META_PATH,
        best_s1_params=np.array(best_s1_params),
        best_w=best_w,
        best_th=best_th,
    )


# ==============================================================================
# 8. 主流程阶段三：测试集五折两阶段级联推理、无损 PNG 保存与官方 RLE CSV 导出
# ==============================================================================
def run_test_inference_and_export_csv(device):
    os.makedirs(Config.OUTPUT_IMG_DIR, exist_ok=True)
    meta = np.load(Config.STAGE2_META_PATH)
    best_s1_params = meta["best_s1_params"]
    best_w = float(meta["best_w"])
    best_th = float(meta["best_th"])
    th_low, th_high, min_area = float(best_s1_params[0]), float(best_s1_params[1]), int(best_s1_params[2])

    test_files = sorted(
        [f for f in os.listdir(Config.TEST_IMG_DIR) if not f.startswith(".")],
        key=lambda x: int(os.path.splitext(x)[0])
    )

    test_data = []
    for fname in test_files:
        rgb = np.array(Image.open(os.path.join(Config.TEST_IMG_DIR, fname)).convert("RGB"))
        test_data.append({
            "id": os.path.splitext(fname)[0],
            "rgb": rgb,
            "enhanced": preprocess_fundus_image(rgb),
            "fov": extract_fov_mask(rgb),
        })

    final_probs = []
    for fold in range(1, Config.N_FOLDS + 1):
        print(f"\n========== 测试集推理 Fold {fold}/{Config.N_FOLDS} ==========")
        model1 = SAResUNet(in_channels=3, base_c=Config.S1_BASE_CHANNELS).to(device)
        model1.load_state_dict(torch.load(os.path.join(Config.CKPT_DIR, f"sa_resunet_fold{fold}.pth"), map_location=device))
        model1.eval()

        model2 = CascadedRefinementUNet(in_channels=5, base_c=Config.S2_BASE_CHANNELS).to(device)
        model2.load_state_dict(torch.load(os.path.join(Config.CKPT_DIR, f"stage2_refine_fold{fold}.pth"), map_location=device))
        model2.eval()

        fold_probs = []
        for item in test_data:
            p1 = predict_full_image_tta(model1, item["enhanced"], item["fov"], device, use_tta=True)
            feat_5ch, fov_512, _, _ = build_stage2_5ch_input(item["enhanced"], item["fov"], p1)

            v_584 = hysteresis_threshold(p1, th_low=th_low, th_high=th_high, min_area=min_area)
            p1_rle = simulate_official_rle_binary(Image.fromarray((v_584 * 255).astype(np.uint8), mode="L")).astype(np.float32) * fov_512
            feat_5ch[:, :, 4] = (p1_rle - 0.5) * 2.0

            p2 = predict_stage2_d4_tta(model2, feat_5ch, fov_512, device)
            p_blend = best_w * p2 + (1.0 - best_w) * p1_rle
            fold_probs.append(p_blend)

        final_probs.append(fold_probs)
        del model1, model2
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    final_probs = np.mean(np.asarray(final_probs, dtype=np.float32), axis=0)

    # 后处理保存 512x512 无损二值 PNG，并调用官方 RLE 编码逻辑导出 submission.csv
    imagelist, namelist = [], []
    for i, item in enumerate(test_data):
        pred = hysteresis_threshold(final_probs[i], th_low=best_th, th_high=min(best_th + 0.05, 0.85), min_area=8)
        save_path = os.path.join(Config.OUTPUT_IMG_DIR, f"{item['id']}.png")
        Image.fromarray((pred.astype(np.uint8) * 255), mode="L").save(save_path, format="PNG")
        imagelist.append(save_path)
        namelist.append(f"{item['id']}.png")

    outputs = []
    for image_path in imagelist:
        image = Image.open(image_path).convert("L").resize((512, 512), Image.Resampling.BILINEAR)
        tensor = torchvision.transforms.ToTensor()(image)
        tensor[tensor > 0] = 1
        dots = np.where(tensor.flatten() == 1)[0]
        run_lengths, prev = [], -2
        for b in dots:
            if b > prev + 1:
                run_lengths.extend((b + 1, 0))
            run_lengths[-1] += 1
            prev = b
        outputs.append(" ".join([str(r) for r in run_lengths]))

    df_sub = pd.DataFrame({"Id": [i.split(".")[0] for i in namelist], "Predicted": outputs})
    df_sub.to_csv(Config.SUBMISSION_PATH, index=None)
    print(f"\n[全部流程结束] 已保存 20 张测试集分割掩码至 {Config.OUTPUT_IMG_DIR}，并导出 {Config.SUBMISSION_PATH}！")


# ==============================================================================
# 9. 程序主入口
# ==============================================================================
def main():
    seed_everything(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[运行环境] PyTorch {torch.__version__} | Device: {device}")

    train_data = load_train_dataset(Config.TRAIN_IMG_DIR, Config.TRAIN_LBL_DIR)
    seed_everything(Config.SEED)
    indices = np.arange(len(train_data))
    np.random.shuffle(indices)
    fold_splits = np.array_split(indices, Config.N_FOLDS)

    # 1. 第一阶段：训练 5折 SA-ResUNet 并提取多尺度滑窗 OOF
    oof_ms_probs = run_stage1_training_and_ms_oof(train_data, fold_splits, indices, device)

    # 2. 第二阶段：训练 5折 CascadedRefinementUNet 并寻优两阶段融合权重
    run_stage2_cascaded_refinement(train_data, oof_ms_probs, fold_splits, indices, device)

    # 3. 第三阶段：测试集五折两阶段级联推理并生成官方 RLE 提交文件
    run_test_inference_and_export_csv(device)


if __name__ == "__main__":
    main()