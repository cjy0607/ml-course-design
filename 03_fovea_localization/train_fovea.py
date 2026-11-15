"""眼底图像中心凹定位。

Stage-1 用 ResNet34 热图回归完成粗定位，Stage-2 在局部窗口内用 FPN 精炼，
5 折交叉验证后按坐标一致性自适应融合。
"""

import os
import time
import zipfile
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2

# =====================================================================
# 2D 眼底图像中心凹检测定位 
# =====================================================================

class Config:
    BASE_INPUT = "/kaggle/input"
    DATA_DIR = None
    for root, dirs, files in os.walk(BASE_INPUT):
        if "fovea_localization_train_GT.csv" in files:
            DATA_DIR = root
            break
    if DATA_DIR is None:
        DATA_DIR = "./detection"

    TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train")
    TEST_IMG_DIR = os.path.join(DATA_DIR, "test")
    TRAIN_XML_DIR = os.path.join(DATA_DIR, "train_location")
    TRAIN_CSV = os.path.join(DATA_DIR, "fovea_localization_train_GT.csv")
    SUB_CSV = os.path.join(DATA_DIR, "sample_submission.csv")

    STAGE1_SIZE = 512
    STAGE2_SIZE = 384
    BATCH_SIZE = 8
    N_SPLITS = 5
    SEED = 2026
    LR = 1e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_everything(seed=2026):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(Config.SEED)
print(f"当前运行设备: {Config.DEVICE} | 数据集路径: {Config.DATA_DIR}")

# ---------------------------------------------------------------------
# 1. 基础图像处理与眼底绿光通道增强
# ---------------------------------------------------------------------
def clean_id_candidates(data_id):
    s = str(data_id).strip()
    if s.lower().endswith(('.jpg', '.png', '.bmp', '.jpeg', '.xml')):
        s = os.path.splitext(s)[0]
    candidates = [s]
    try:
        num = int(float(s))
        candidates.extend([f"{num:04d}", f"{num:03d}", f"{num:02d}", str(num)])
    except ValueError:
        pass
    return list(dict.fromkeys(candidates))

def find_image_path(img_dir, data_id):
    candidates = clean_id_candidates(data_id)
    for cand in candidates:
        for ext in ['.jpg', '.JPG', '.png', '.PNG', '.jpeg', '.bmp']:
            p = os.path.join(img_dir, cand + ext)
            if os.path.exists(p):
                return p
    for f in os.listdir(img_dir):
        if os.path.splitext(f)[0] in candidates:
            return os.path.join(img_dir, f)
    raise FileNotFoundError(f"未找到图片: {data_id}")

def parse_xml_bbox(xml_dir, data_id):
    if not os.path.exists(xml_dir):
        return None
    for cand in clean_id_candidates(data_id):
        for ext in ['.xml', '.XML']:
            p = os.path.join(xml_dir, cand + ext)
            if os.path.exists(p):
                try:
                    bndbox = ET.parse(p).getroot().find(".//bndbox")
                    if bndbox is not None:
                        return [float(bndbox.find("xmin").text), float(bndbox.find("ymin").text),
                                float(bndbox.find("xmax").text), float(bndbox.find("ymax").text)]
                except Exception:
                    pass
    return None

def crop_fundus_fov(image, threshold=10):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = gray > threshold
    if not np.any(mask):
        h, w = image.shape[:2]
        return image, 0, 0, w, h
    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return image[y0:y1, x0:x1], x0, y0, x1 - x0, y1 - y0

def enhance_fundus_green(img_rgb):
    img = img_rgb.copy()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g_enh = clahe.apply(img[:, :, 1])
    img[:, :, 1] = cv2.addWeighted(img[:, :, 1], 0.55, g_enh, 0.45, 0)
    return img

def extract_safe_patch(img, center_x, center_y, half_r):
    h, w = img.shape[:2]
    x1, y1 = int(round(center_x - half_r)), int(round(center_y - half_r))
    x2, y2 = x1 + 2 * half_r, y1 + 2 * half_r
    pad_l, pad_t = max(0, -x1), max(0, -y1)
    pad_r, pad_b = max(0, x2 - w), max(0, y2 - h)
    if pad_l > 0 or pad_t > 0 or pad_r > 0 or pad_b > 0:
        img_padded = cv2.copyMakeBorder(img, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    else:
        img_padded = img
    patch = img_padded[y1 + pad_t : y2 + pad_t, x1 + pad_l : x2 + pad_l]
    return patch, x1, y1, 2 * half_r

# ---------------------------------------------------------------------
# 2. 模型网络架构定义
# ---------------------------------------------------------------------
class SpatialSoftArgmax2d(nn.Module):
    def __init__(self, temperature=0.05):
        super().__init__()
        self.temperature = temperature

    def forward(self, heatmap):
        B, C, H, W = heatmap.shape
        prob = torch.softmax(heatmap.view(B, C, -1) / self.temperature, dim=-1).view(B, C, H, W)
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, 1, H, device=heatmap.device),
            torch.linspace(0, 1, W, device=heatmap.device), indexing='ij'
        )
        exp_x = torch.sum(prob * grid_x, dim=[2, 3])
        exp_y = torch.sum(prob * grid_y, dim=[2, 3])
        return torch.stack([exp_x, exp_y], dim=-1)

@torch.no_grad()
def decode_heatmap_dark(heatmap_logits, soft_coord, kernel_radius=13):
    B, _, H, W = heatmap_logits.shape
    device = heatmap_logits.device
    prob = torch.sigmoid(heatmap_logits).squeeze(1)
    gy, gx = torch.meshgrid(torch.linspace(-1, 1, H, device=device), torch.linspace(-1, 1, W, device=device), indexing='ij')
    center_prior = torch.exp(-0.20 * (gx**2 + gy**2)).unsqueeze(0)
    weighted_prob = prob * center_prior
    flat_idx = torch.argmax(weighted_prob.view(B, -1), dim=-1)
    max_y = (flat_idx // W).view(B, 1, 1).float()
    max_x = (flat_idx % W).view(B, 1, 1).float()
    idx_y, idx_x = torch.meshgrid(torch.arange(H, device=device).float(), torch.arange(W, device=device).float(), indexing='ij')
    dist_sq = (idx_x.unsqueeze(0) - max_x)**2 + (idx_y.unsqueeze(0) - max_y)**2
    local_mask = (dist_sq <= (kernel_radius ** 2)).float()
    masked_logits = heatmap_logits.squeeze(1) - (1.0 - local_mask) * 1e4
    local_prob = torch.softmax(masked_logits.view(B, -1) / 0.04, dim=-1).view(B, H, W)
    norm_y, norm_x = torch.meshgrid(torch.linspace(0, 1, H, device=device), torch.linspace(0, 1, W, device=device), indexing='ij')
    dark_x = torch.sum(local_prob * norm_x, dim=[1, 2])
    dark_y = torch.sum(local_prob * norm_y, dim=[1, 2])
    dark_coord = torch.stack([dark_x, dark_y], dim=-1)
    return 0.65 * dark_coord + 0.35 * soft_coord

def make_gaussian_target(coords, H, W, sigma=2.8):
    B = coords.shape[0]
    device = coords.device
    gy, gx = torch.meshgrid(torch.linspace(0, 1, H, device=device), torch.linspace(0, 1, W, device=device), indexing='ij')
    gx, gy = gx.unsqueeze(0).expand(B, -1, -1), gy.unsqueeze(0).expand(B, -1, -1)
    cx, cy = coords[:, 0].view(B, 1, 1), coords[:, 1].view(B, 1, 1)
    dist_sq = ((gx - cx) * W) ** 2 + ((gy - cy) * H) ** 2
    return torch.exp(-dist_sq / (2 * sigma ** 2)).unsqueeze(1)

class Stage1MaculaDetector(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = base.layer1, base.layer2, base.layer3, base.layer4
        self.lat4, self.lat3 = nn.Conv2d(512, 128, 1), nn.Conv2d(256, 128, 1)
        self.hm_head = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.SiLU(), nn.Conv2d(128, 2, 1))
        self.soft_argmax = SpatialSoftArgmax2d(temperature=0.08)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc_vis = nn.Linear(512, 1)

    def forward(self, x):
        c1 = self.layer1(self.stem(x))
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        p3 = self.lat3(c3) + F.interpolate(self.lat4(c4), size=c3.shape[-2:], mode='bilinear', align_corners=False)
        hm = self.hm_head(p3)
        coords = self.soft_argmax(hm)
        vis_logit = self.fc_vis(self.pool(c4).flatten(1)).squeeze(-1)
        return coords[:, 0, :], coords[:, 1, :], vis_logit, hm[:, 0:1, :, :]

class HighResFPNRefiner(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = base.layer1, base.layer2, base.layer3, base.layer4
        self.lat4, self.lat3, self.lat2, self.lat1 = nn.Conv2d(512, 128, 1), nn.Conv2d(256, 128, 1), nn.Conv2d(128, 128, 1), nn.Conv2d(64, 128, 1)
        self.smooth = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.SiLU(),
                                    nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.SiLU(), nn.Conv2d(64, 1, 1))
        self.soft_argmax = SpatialSoftArgmax2d(temperature=0.05)

    def forward(self, x):
        c1 = self.layer1(self.stem(x))
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        p4 = self.lat4(c4)
        p3 = self.lat3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode='bilinear', align_corners=False)
        p2 = self.lat2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode='bilinear', align_corners=False)
        p1 = self.lat1(c1) + F.interpolate(p2, size=c1.shape[-2:], mode='bilinear', align_corners=False)
        hm_logits = self.smooth(p1)
        coord = self.soft_argmax(hm_logits).squeeze(1)
        return coord, hm_logits

# ---------------------------------------------------------------------
# 3. 数据集与增强管道定义
# ---------------------------------------------------------------------
class UltraFastStage1Dataset(Dataset):
    def __init__(self, records, transform=None):
        self.records = records
        self.transform = transform
    def __len__(self):
        return len(self.records)
    def __getitem__(self, idx):
        rec = self.records[idx]
        img_512, cw, ch, x0, y0 = rec['img_512'], rec['cw'], rec['ch'], rec['x0'], rec['y0']
        fx, fy, is_vis, bbox = rec['fx'], rec['fy'], rec['is_vis'], rec['bbox']
        sx, sy = Config.STAGE1_SIZE / cw, Config.STAGE1_SIZE / ch
        if bbox is not None and is_vis > 0.5:
            bx = np.clip((0.5 * (bbox[0] + bbox[2]) - x0) * sx, 0, Config.STAGE1_SIZE - 1)
            by = np.clip((0.5 * (bbox[1] + bbox[3]) - y0) * sy, 0, Config.STAGE1_SIZE - 1)
        else:
            bx = np.clip((fx - x0) * sx, 0, Config.STAGE1_SIZE - 1) if is_vis > 0.5 else Config.STAGE1_SIZE / 2.0
            by = np.clip((fy - y0) * sy, 0, Config.STAGE1_SIZE - 1) if is_vis > 0.5 else Config.STAGE1_SIZE / 2.0
        adj_fx = np.clip((fx - x0) * sx, 0, Config.STAGE1_SIZE - 1) if is_vis > 0.5 else Config.STAGE1_SIZE / 2.0
        adj_fy = np.clip((fy - y0) * sy, 0, Config.STAGE1_SIZE - 1) if is_vis > 0.5 else Config.STAGE1_SIZE / 2.0
        trans = self.transform(image=img_512, keypoints=[(adj_fx, adj_fy), (bx, by)])
        img_t = trans['image']
        kps = trans['keypoints']
        n_fx, n_fy = np.clip(kps[0][0] / Config.STAGE1_SIZE, 0.0, 1.0), np.clip(kps[0][1] / Config.STAGE1_SIZE, 0.0, 1.0)
        n_bx, n_by = np.clip(kps[1][0] / Config.STAGE1_SIZE, 0.0, 1.0), np.clip(kps[1][1] / Config.STAGE1_SIZE, 0.0, 1.0)
        return img_t, torch.tensor([n_fx, n_fy, n_bx, n_by, is_vis], dtype=torch.float32), idx

class Stage2ZoomDataset(Dataset):
    def __init__(self, records, transform=None):
        self.records = records
        self.transform = transform
    def __len__(self):
        return len(self.records)
    def __getitem__(self, idx):
        rec = self.records[idx]
        img, fx, fy, max_dim = rec['img_enhanced'], rec['fx'], rec['fy'], max(rec['cw'], rec['ch'])
        ratio = np.random.uniform(0.22, 0.40)
        half_r = int(max_dim * ratio * 0.5)
        max_shift = int(half_r * 0.36)
        cx = fx + np.random.randint(-max_shift, max_shift + 1)
        cy = fy + np.random.randint(-max_shift, max_shift + 1)
        patch, x1, y1, side = extract_safe_patch(img, cx, cy, half_r)
        local_x, local_y = np.clip(fx - x1, 0, side - 1), np.clip(fy - y1, 0, side - 1)
        trans = self.transform(image=patch, keypoints=[(local_x, local_y)])
        patch_t = trans['image']
        kp_x = trans['keypoints'][0][0] if len(trans['keypoints']) > 0 else local_x * (Config.STAGE2_SIZE / side)
        kp_y = trans['keypoints'][0][1] if len(trans['keypoints']) > 0 else local_y * (Config.STAGE2_SIZE / side)
        return patch_t, torch.tensor([np.clip(kp_x / Config.STAGE2_SIZE, 0.0, 1.0), np.clip(kp_y / Config.STAGE2_SIZE, 0.0, 1.0)], dtype=torch.float32)

@torch.no_grad()
def predict_patch_4tta(model, img, cx, cy, half_r, val_tf, device):
    patch, x1, y1, side = extract_safe_patch(img, cx, cy, half_r)
    pt = val_tf(image=patch, keypoints=[(0, 0)])['image'].unsqueeze(0).to(device)
    batch_pt = torch.cat([pt, torch.flip(pt, dims=[3]), torch.flip(pt, dims=[2]), torch.flip(pt, dims=[2, 3])], dim=0)
    soft_c, hm_logits = model(batch_pt)
    coords = decode_heatmap_dark(hm_logits, soft_c, kernel_radius=13)
    coords[1, 0] = 1.0 - coords[1, 0]
    coords[2, 1] = 1.0 - coords[2, 1]
    coords[3, 0] = 1.0 - coords[3, 0]
    coords[3, 1] = 1.0 - coords[3, 1]
    c_mean = coords.mean(dim=0).cpu().numpy()
    return float(x1 + c_mean[0] * side), float(y1 + c_mean[1] * side)

@torch.no_grad()
def evaluate_stage2_progressive(model, val_records, val_tf, device):
    model.eval()
    sq_errors, details = [], []
    for rec in val_records:
        true_x, true_y = rec['fx'], rec['fy']
        if rec['vis_prob'] < 0.35:
            final_x, final_y = 0.0, 0.0
        else:
            img, max_dim, cx, cy = rec['img_enhanced'], max(rec['cw'], rec['ch']), rec['coarse_x'], rec['coarse_y']
            z1_x, z1_y = predict_patch_4tta(model, img, cx, cy, int(max_dim * 0.38 * 0.5), val_tf, device)
            z2_x1, z2_y1 = predict_patch_4tta(model, img, z1_x, z1_y, int(max_dim * 0.24 * 0.5), val_tf, device)
            z2_x2, z2_y2 = predict_patch_4tta(model, img, z1_x, z1_y, int(max_dim * 0.28 * 0.5), val_tf, device)
            final_x = 0.45 * z2_x1 + 0.40 * z2_x2 + 0.15 * z1_x
            final_y = 0.45 * z2_y1 + 0.40 * z2_y2 + 0.15 * z1_y
        sq_errors.extend([(final_x - true_x)**2, (final_y - true_y)**2])
        dist = np.sqrt((final_x - true_x)**2 + (final_y - true_y)**2)
        details.append((rec['data_id'], true_x, true_y, final_x, final_y, dist))
    return np.sqrt(np.mean(sq_errors)), details

def get_zoom_transforms():
    sz = Config.STAGE2_SIZE
    train_tf = A.Compose([
        A.Resize(sz, sz), A.CLAHE(clip_limit=2.5, tile_grid_size=(8, 8), p=0.6),
        A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5),
        A.Affine(scale=(0.92, 1.08), rotate=(-25, 25), p=0.6),
        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    val_tf = A.Compose([
        A.Resize(sz, sz), A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    return train_tf, val_tf

# ---------------------------------------------------------------------
# 4. 主流程：自动训练并生成图表、数据表与 submission.csv
# ---------------------------------------------------------------------
def main_pipeline():
    train_df = pd.read_csv(Config.TRAIN_CSV)
    print("正在加载训练图像...")
    all_records = []
    for idx, row in train_df.iterrows():
        did = clean_id_candidates(row['data'])[0]
        fx, fy = float(row['Fovea_X']), float(row['Fovea_Y'])
        img_raw = cv2.cvtColor(cv2.imread(find_image_path(Config.TRAIN_IMG_DIR, did)), cv2.COLOR_BGR2RGB)
        img_enh = enhance_fundus_green(img_raw)
        cropped, x0, y0, cw, ch = crop_fundus_fov(img_enh)
        img_512 = cv2.resize(cropped, (Config.STAGE1_SIZE, Config.STAGE1_SIZE))
        bbox = parse_xml_bbox(Config.TRAIN_XML_DIR, did)
        all_records.append({
            'data_id': did, 'img_enhanced': img_enh, 'cropped_img': cropped, 'img_512': img_512,
            'x0': x0, 'y0': y0, 'cw': cw, 'ch': ch, 'fx': fx, 'fy': fy,
            'is_vis': 0.0 if (fx <= 1e-3 and fy <= 1e-3) else 1.0, 'bbox': bbox,
            'coarse_x': 0.0, 'coarse_y': 0.0, 'vis_prob': 1.0
        })

    s1_train_tf = A.Compose([
        A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
        A.Affine(scale=(0.92, 1.08), translate_percent=(-0.05, 0.05), rotate=(-20, 20), p=0.7),
        A.ColorJitter(brightness=0.15, contrast=0.15, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    s1_val_tf = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    s2_train_tf, s2_val_tf = get_zoom_transforms()
    kf = KFold(n_splits=Config.N_SPLITS, shuffle=True, random_state=Config.SEED)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    oof_list = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_records)):
        train_recs = [all_records[i] for i in train_idx]
        val_recs = [all_records[i] for i in val_idx]
        s1_path = f"val_model_fold{fold}.pth"
        s2_path = f"fine_model_fold{fold}.pth"

        # Stage-1 训练或加载
        s1_model = Stage1MaculaDetector().to(Config.DEVICE)
        if not (os.path.exists(s1_path) and os.path.exists(s2_path)):
            print(f"\n[Fold {fold + 1}/5] 训练 Stage-1 模型...")
            s1_loader = DataLoader(UltraFastStage1Dataset(train_recs * 2, s1_train_tf), batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2)
            s1_val_loader = DataLoader(UltraFastStage1Dataset(val_recs, s1_val_tf), batch_size=Config.BATCH_SIZE, shuffle=False)
            opt1 = torch.optim.AdamW(s1_model.parameters(), lr=Config.LR, weight_decay=1e-3)
            sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=40, eta_min=1e-6)
            best_s1 = float("inf")
            for ep in range(1, 41):
                s1_model.train()
                for imgs, targets, _ in s1_loader:
                    imgs, targets = imgs.to(Config.DEVICE), targets.to(Config.DEVICE)
                    opt1.zero_grad()
                    with torch.cuda.amp.autocast():
                        pf, pb, pvis, hm_f = s1_model(imgs)
                        vmask = (targets[:, 4] > 0.5).float().unsqueeze(1)
                        gt_hm1 = make_gaussian_target(targets[:, 0:2], H=hm_f.shape[2], W=hm_f.shape[3], sigma=2.0)
                        lf = (F.smooth_l1_loss(pf, targets[:, 0:2], reduction='none', beta=0.015) * vmask).sum() / (vmask.sum()*2 + 1e-6)
                        lb = (F.smooth_l1_loss(pb, targets[:, 2:4], reduction='none', beta=0.015) * vmask).sum() / (vmask.sum()*2 + 1e-6)
                        lhm = (F.mse_loss(torch.sigmoid(hm_f), gt_hm1, reduction='none').mean(dim=[1,2,3], keepdim=True) * vmask).sum() / (vmask.sum() + 1e-6)
                        loss = lf + 0.4 * lb + 0.4 * lhm + 0.05 * F.binary_cross_entropy_with_logits(pvis, targets[:, 4])
                    scaler.scale(loss).backward()
                    scaler.step(opt1)
                    scaler.update()
                sched1.step()
                if ep in [30, 40]:
                    s1_model.eval()
                    sq = []
                    with torch.no_grad():
                        for imgs, _, idxs in s1_val_loader:
                            imgs = imgs.to(Config.DEVICE)
                            cf1, cb1, v1, _ = s1_model(imgs)
                            cf2, cb2, v2, _ = s1_model(torch.flip(imgs, dims=[3]))
                            cf2[:, 0], cb2[:, 0] = 1.0 - cf2[:, 0], 1.0 - cb2[:, 0]
                            c_n = (0.40 * (cf1 + cf2) + 0.10 * (cb1 + cb2)).cpu().numpy()
                            vp = torch.sigmoid(0.5 * (v1 + v2)).cpu().numpy()
                            for b_i, r_idx in enumerate(idxs.numpy()):
                                rec = val_recs[r_idx]
                                px = 0.0 if vp[b_i] < 0.35 else float(c_n[b_i, 0] * rec['cw'] + rec['x0'])
                                py = 0.0 if vp[b_i] < 0.35 else float(c_n[b_i, 1] * rec['ch'] + rec['y0'])
                                sq.extend([(px - rec['fx'])**2, (py - rec['fy'])**2])
                    if np.sqrt(np.mean(sq)) < best_s1:
                        best_s1 = np.sqrt(np.mean(sq))
                        torch.save(s1_model.state_dict(), s1_path)

        s1_model.load_state_dict(torch.load(s1_path, map_location=Config.DEVICE))
        s1_model.eval()
        with torch.no_grad():
            for rec in val_recs:
                inp = s1_val_tf(image=rec['img_512'], keypoints=[(0,0)])['image'].unsqueeze(0).to(Config.DEVICE)
                b_inp = torch.cat([inp, torch.flip(inp, dims=[3]), torch.flip(inp, dims=[2]), torch.flip(inp, dims=[2, 3])], dim=0)
                cf, cb, v, hm1 = s1_model(b_inp)
                cf_dark = decode_heatmap_dark(hm1, cf, kernel_radius=7)
                cf_dark[1, 0], cb[1, 0] = 1.0 - cf_dark[1, 0], 1.0 - cb[1, 0]
                cf_dark[2, 1], cb[2, 1] = 1.0 - cf_dark[2, 1], 1.0 - cb[2, 1]
                cf_dark[3, 0], cb[3, 0] = 1.0 - cf_dark[3, 0], 1.0 - cb[3, 0]
                cf_dark[3, 1], cb[3, 1] = 1.0 - cf_dark[3, 1], 1.0 - cb[3, 1]
                f_n = cf_dark.mean(dim=0).cpu().numpy()
                b_n = cb.mean(dim=0).cpu().numpy()
                c_n = 0.80 * f_n + 0.20 * b_n
                vp = torch.sigmoid(v.mean()).cpu().item()
                rec['vis_prob'] = vp
                rec['coarse_x'] = 0.0 if vp < 0.35 else float(c_n[0] * rec['cw'] + rec['x0'])
                rec['coarse_y'] = 0.0 if vp < 0.35 else float(c_n[1] * rec['ch'] + rec['y0'])

        # Stage-2 训练或加载
        s2_model = HighResFPNRefiner().to(Config.DEVICE)
        if not os.path.exists(s2_path):
            print(f"[Fold {fold + 1}/5] 训练 Stage-2 模型...")
            vis_train_recs = [r for r in train_recs if r['is_vis'] > 0.5]
            s2_loader = DataLoader(Stage2ZoomDataset(vis_train_recs * 3, s2_train_tf), batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2)
            opt2 = torch.optim.AdamW(s2_model.parameters(), lr=Config.LR, weight_decay=1e-3)
            sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=40, eta_min=1e-6)
            best_s2 = float("inf")
            for ep in range(1, 41):
                s2_model.train()
                for patches, targets in s2_loader:
                    patches, targets = patches.to(Config.DEVICE), targets.to(Config.DEVICE)
                    opt2.zero_grad()
                    with torch.cuda.amp.autocast():
                        p_coord, hm_logits = s2_model(patches)
                        gt_hm = make_gaussian_target(targets, H=hm_logits.shape[2], W=hm_logits.shape[3], sigma=2.8)
                        loss = F.smooth_l1_loss(p_coord, targets, beta=0.012) + 0.6 * F.mse_loss(torch.sigmoid(hm_logits), gt_hm)
                    scaler.scale(loss).backward()
                    scaler.step(opt2)
                    scaler.update()
                sched2.step()
                if ep in [30, 40]:
                    rmse2, _ = evaluate_stage2_progressive(s2_model, val_recs, s2_val_tf, Config.DEVICE)
                    if rmse2 < best_s2:
                        best_s2 = rmse2
                        torch.save(s2_model.state_dict(), s2_path)

        s2_model.load_state_dict(torch.load(s2_path, map_location=Config.DEVICE))
        _, details = evaluate_stage2_progressive(s2_model, val_recs, s2_val_tf, Config.DEVICE)

        for rec, (_, tx, ty, s2_x, s2_y, _) in zip(val_recs, details):
            oof_list.append({'fold': fold, 'data_id': rec['data_id'], 'true_x': tx, 'true_y': ty,
                             's1_x': rec['coarse_x'], 's1_y': rec['coarse_y'], 's2_x': s2_x, 's2_y': s2_y})

    oof_df = pd.DataFrame(oof_list)
    diff_dist = np.sqrt((oof_df['s2_x'] - oof_df['s1_x'])**2 + (oof_df['s2_y'] - oof_df['s1_y'])**2)
    w_vec = np.where(diff_dist > 35.0, 0.20, 0.75)
    oof_df['pred_x'] = w_vec * oof_df['s2_x'] + (1.0 - w_vec) * oof_df['s1_x']
    oof_df['pred_y'] = w_vec * oof_df['s2_y'] + (1.0 - w_vec) * oof_df['s1_y']
    oof_df['dist_error'] = np.sqrt((oof_df['pred_x'] - oof_df['true_x'])**2 + (oof_df['pred_y'] - oof_df['true_y'])**2)
    oof_df.to_csv("oof_predictions.csv", index=False)

    # ---------------------------------------------------------------------
    # 5. 自动生成测试集 submission.csv
    # ---------------------------------------------------------------------
    print("\n正在生成测试集预测结果 (submission.csv)...")
    sub_df = pd.read_csv(Config.SUB_CSV)
    sub_df['img_prefix'] = sub_df['ImageID'].apply(lambda s: str(s).strip().split("_Fovea_")[0] if "_Fovea_" in str(s) else str(s).strip().rsplit("_", 1)[0])
    unique_test_ids = sub_df['img_prefix'].unique()

    c_models, f_models = [], []
    for f in range(5):
        cm = Stage1MaculaDetector().to(Config.DEVICE)
        cm.load_state_dict(torch.load(f"val_model_fold{f}.pth", map_location=Config.DEVICE))
        cm.eval()
        c_models.append(cm)
        fm = HighResFPNRefiner().to(Config.DEVICE)
        fm.load_state_dict(torch.load(f"fine_model_fold{f}.pth", map_location=Config.DEVICE))
        fm.eval()
        f_models.append(fm)

    coarse_val_tf = A.Compose([A.Resize(Config.STAGE1_SIZE, Config.STAGE1_SIZE), A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()])
    pred_dict = {}

    for did in unique_test_ids:
        img_path = find_image_path(Config.TEST_IMG_DIR, did)
        img_raw = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        img_enh = enhance_fundus_green(img_raw)
        cropped, x0, y0, cw, ch = crop_fundus_fov(img_enh)
        max_dim = max(cw, ch)

        img_512 = cv2.resize(cropped, (Config.STAGE1_SIZE, Config.STAGE1_SIZE))
        inp = coarse_val_tf(image=img_512)['image'].unsqueeze(0).to(Config.DEVICE)
        batch_inp = torch.cat([inp, torch.flip(inp, dims=[3]), torch.flip(inp, dims=[2]), torch.flip(inp, dims=[2, 3])], dim=0)

        c_preds, vis_scores = [], []
        for cm in c_models:
            cf, cb, v, hm1 = cm(batch_inp)
            cf_dark = decode_heatmap_dark(hm1, cf, kernel_radius=7)
            cf_dark[1, 0], cb[1, 0] = 1.0 - cf_dark[1, 0], 1.0 - cb[1, 0]
            cf_dark[2, 1], cb[2, 1] = 1.0 - cf_dark[2, 1], 1.0 - cb[2, 1]
            cf_dark[3, 0], cb[3, 0] = 1.0 - cf_dark[3, 0], 1.0 - cb[3, 0]
            cf_dark[3, 1], cb[3, 1] = 1.0 - cf_dark[3, 1], 1.0 - cb[3, 1]
            f_n = cf_dark.mean(dim=0).cpu().numpy()
            b_n = cb.mean(dim=0).cpu().numpy()
            c_n = 0.80 * f_n + 0.20 * b_n
            c_preds.append([c_n[0] * cw + x0, c_n[1] * ch + y0])
            vis_scores.append(torch.sigmoid(v.mean()).cpu().item())

        if np.mean(vis_scores) < 0.35:
            pred_dict[str(did)] = (0.0, 0.0)
            continue

        coarse_median = np.median(c_preds, axis=0)
        cx, cy = float(coarse_median[0]), float(coarse_median[1])

        r_wide = int(max_dim * 0.36 * 0.5)
        z1_x, z1_y = predict_patch_4tta(f_models[0], img_enh, cx, cy, r_wide, s2_val_tf, Config.DEVICE)

        s2_candidates = []
        for scale in [0.24, 0.28]:
            r_scale = int(max_dim * scale * 0.5)
            patch, px1, py1, pside = extract_safe_patch(img_enh, z1_x, z1_y, r_scale)
            pt = s2_val_tf(image=patch, keypoints=[(0,0)])['image'].unsqueeze(0).to(Config.DEVICE)
            b_pt = torch.cat([pt, torch.flip(pt, dims=[3]), torch.flip(pt, dims=[2]), torch.flip(pt, dims=[2, 3])], dim=0)
            for fm in f_models:
                soft_c, hm_l = fm(b_pt)
                coords = decode_heatmap_dark(hm_l, soft_c, kernel_radius=13)
                coords[1, 0] = 1.0 - coords[1, 0]
                coords[2, 1] = 1.0 - coords[2, 1]
                coords[3, 0] = 1.0 - coords[3, 0]
                coords[3, 1] = 1.0 - coords[3, 1]
                cm_pt = coords.mean(dim=0).cpu().numpy()
                s2_candidates.append([px1 + cm_pt[0] * pside, py1 + cm_pt[1] * pside])

        s2_median = np.median(s2_candidates, axis=0)
        s2_x, s2_y = float(s2_median[0]), float(s2_median[1])

        diff = np.hypot(s2_x - cx, s2_y - cy)
        w = 0.20 if diff > 35.0 else 0.75
        pred_dict[str(did)] = (w * s2_x + (1.0 - w) * cx, w * s2_y + (1.0 - w) * cy)

    final_values = []
    for _, row in sub_df.iterrows():
        px, py = pred_dict[str(row['img_prefix'])]
        final_values.append(px if str(row['ImageID']).endswith(("_X", "Fovea_X")) else py)

    submission = pd.DataFrame({'ImageID': sub_df['ImageID'], 'value': final_values})
    submission.to_csv("submission.csv", index=False)
    print("submission.csv 已生成")

    # ---------------------------------------------------------------------
    # 6. 绘制误差分析图与统计表
    # ---------------------------------------------------------------------
    print("\n正在生成误差分析图与统计表...")
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False

    # 图表 1: 误差分布直方图与 KDE 曲线
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    errors = oof_df['dist_error'].values
    ax.hist(errors, bins=15, color='#4c72b0', edgecolor='white', alpha=0.8, density=True)
    from scipy.stats import gaussian_kde
    density = gaussian_kde(errors)
    xs = np.linspace(0, max(errors)+5, 200)
    ax.plot(xs, density(xs), color='#c44e52', linewidth=2.5, label='KDE Curve')
    ax.axvline(np.mean(errors), color='#55a868', linestyle='--', linewidth=2, label=f'Mean Error ({np.mean(errors):.2f} px)')
    ax.axvline(np.median(errors), color='#8172b3', linestyle='-.', linewidth=2, label=f'Median Error ({np.median(errors):.2f} px)')
    ax.set_xlabel('Euclidean Distance Error (Pixels)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Density', fontsize=11, fontweight='bold')
    ax.set_title('Distribution of Localization Error on Validation Set (OOF)', fontsize=12, fontweight='bold', pad=12)
    ax.legend(fontsize=10, frameon=True)
    plt.tight_layout()
    plt.savefig('fig4_error_distribution_histogram.png')
    plt.close()

    # 图表 2: 各折误差箱线图
    fold_groups = [oof_df[oof_df['fold'] == f]['dist_error'].values for f in range(5)]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.boxplot(fold_groups, patch_artist=True, notch=True,
               boxprops=dict(facecolor='#4c72b0', color='black', alpha=0.7),
               medianprops=dict(color='orange', linewidth=2))
    ax.set_xticklabels([f'Fold {f+1}' for f in range(5)], fontsize=11, fontweight='bold')
    ax.set_ylabel('Pixel Error Distribution', fontsize=11, fontweight='bold')
    ax.set_title('Boxplot of Localization Errors Across 5 Folds', fontsize=12, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig('fig5_fold_error_boxplot.png')
    plt.close()

    # 导出统计汇总表 CSV
    stats_data = {
        "Metric_Name": ["Mean Error (px)", "Standard Deviation (px)", "Median Error (px)", "Minimum Error (px)", "Maximum Error (px)"],
        "Final_Optimized_Model": [float(np.mean(errors)), float(np.std(errors)), float(np.median(errors)), float(np.min(errors)), float(np.max(errors))]
    }
    pd.DataFrame(stats_data).to_csv("detailed_statistics.csv", index=False)
    print("误差分析图与统计表已生成")

if __name__ == "__main__":
    main_pipeline()