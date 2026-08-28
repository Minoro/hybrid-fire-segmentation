import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchmetrics.classification import BinaryConfusionMatrix
from torchmetrics import Precision, Recall, F1Score

from torch import optim, nn, utils, Tensor
import lightning as L
import torchvision.transforms as T

from lightning.pytorch import Trainer, seed_everything


from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from tqdm.auto import tqdm

from my_dataset import MultiSatelliteDataset, calculate_mean_std

from datetime import datetime
import random
from torchvision.transforms import functional as TF

import json

import scipy.ndimage as ndi
from scipy.ndimage import uniform_filter
import numpy as np
from typing import List, Dict, Any, Optional, Callable

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from scipy.ndimage import label

from torchvision import models

import shutil

seed_everything(42, workers=True)

GDAL_CROPED = True

BACKBONE = 'DAFDM'

DISABLE_THRESHOLD_BRANCH = True

OVERRIDE_TRAINED = False

USE_FILM = False
NUM_SENSORS_THREHOLD_EQUATIONS = 2


FOLD = 1
LR = 1e-4
BATCH_SIZE = 16
EPOCHS = 50
ES_PATIENCE = 10
WORKERS = 6
DEVICES = [1]

LOSS_B_WEIGHT = 1.0
LOSS_DT_WEIGHT = 1.0
LOSS_FUSED_WEIGHT = 1.0


LANDSAT_DATASET_PATH = '/dataset/Landsat/'
LANDSAT_ANNOTATION_FOLDER = 'manual_annotations_patches'
LANDSAT_IMG_FOLDER = 'landsat_patches'
LANDSAT_BANDS = (7,6,5)
LANDSAT_QUANTIFICATION = 65535.0

SENTINEL_DATASET_PATH = '/dataset/Sentinel/'
SENTINEL_IMG_FOLDER = 'imgs'
SENTINEL_ANNOTATION_FOLDER = 'mask1'
SENTINEL_BANDS = (6,5,4)
SENTINEL_QUANTIFICATION = 10000.0

LANDSAT_DATAFRAME_PATH = f'/hybrid-fire-segmentation/resources/dataframes/folds/landsat-sentinel/landsat_extracted_folds.csv'
SENTINEL_DATAFRAME_PATH =  '/hybrid-fire-segmentation/resources/dataframes/folds/landsat-sentinel/sentinel_extracted_folds.csv'
LANDSAT_SENTINEL_DATAFRAME_PATH = '/hybrid-fire-segmentation/resources/dataframes/folds/landsat-sentinel/landsat-sentinel_folds.csv'


LOG_DIR = f'/hybrid-fire-segmentation/resources/dafdm/logs'



# ---- utils ----
def softplus_param(x: nn.Parameter, min_val: float = 1e-6):
    """Return positive parameter from raw value using softplus."""
    return F.softplus(x) + min_val


def soft_thresh(x: torch.Tensor, thr: torch.Tensor, k: float):
    """differentiable threshold (probability-like)"""
    return safe_sigmoid(k * (x - thr))


def safe_sigmoid(x):
    return torch.sigmoid(torch.clamp(x, -20, 20))


def logistic(x):
    # Converte para float e substitui NaN/inf por 0
    x = np.nan_to_num(x.astype(np.float64), nan=0.0, posinf=60.0, neginf=-60.0)

    # Clipping mais seguro para evitar overflow no exp
    x_clipped = np.clip(x, -60.0, 60.0)

    # Implementação estável da função logística
    pos_mask = x_clipped >= 0
    neg_mask = ~pos_mask

    # Cria array vazio de saída
    y = np.empty_like(x_clipped)

    # Regiões positivas (evita exp(-x) muito grande)
    y[pos_mask] = 1.0 / (1.0 + np.exp(-x_clipped[pos_mask]))

    # Regiões negativas (evita exp(x) muito grande)
    exp_x = np.exp(x_clipped[neg_mask])
    y[neg_mask] = exp_x / (1.0 + exp_x)

    # Substitui quaisquer valores numéricos inválidos resultantes
    y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0)
    return y


# ==== Parâmetros por domínio (sensor) ====
def param_list(num_domains, init_val=1.0):
    return nn.ParameterList([nn.Parameter(torch.tensor(init_val, dtype=torch.float32))
                                for _ in range(num_domains)])


class RandomFlipRotate:
    """
    Aplica rotação aleatória e flip aleatório nas imagens e máscaras.
    """
    def __init__(self, degrees=(0, 360), p_flip=0.5):
        self.rotation = T.RandomRotation(degrees)
        self.h_flip = T.RandomHorizontalFlip(p=p_flip)
        self.v_flip = T.RandomVerticalFlip(p=p_flip)

    def __call__(self, image, mask):
        # Aplica rotação aleatória
        angle = self.rotation.get_params(self.rotation.degrees)
        image = T.functional.rotate(image, angle)
        mask = T.functional.rotate(mask, angle)

        # Aplica flip horizontal aleatório
        if torch.rand(1).item() < self.h_flip.p:
            image = T.functional.hflip(image)
            mask = T.functional.hflip(mask)

        # Aplica flip vertical aleatório
        if torch.rand(1).item() < self.v_flip.p:
            image = T.functional.vflip(image)
            mask = T.functional.vflip(mask)

        return image, mask
    




class ECALayer(nn.Module):
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size,
                              padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)  # [B, C, 1, 1]
        y = self.conv(y.squeeze(-1).transpose(1, 2))  # [B, 1, C]
        y = self.sigmoid(y).transpose(1, 2).unsqueeze(-1)
        return x * y.expand_as(x)


class EFPMBlock(nn.Module):

    def __init__(self,in_ch, out_ch, attn_k=3):
        super().__init__()
        self.eca = ECALayer(in_ch, k_size=attn_k)
        self.refine = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x_att = self.eca(x)
        return self.refine(x * x_att)



class EFPM(nn.Module):
    def __init__(self, low_ch, high_ch, out_ch, attn_k=3):
        super().__init__()

        self.efpm_block = EFPMBlock(low_ch + high_ch, out_ch=out_ch, attn_k=attn_k)

    def forward(self, f_low, f_high):
        x = torch.cat([f_low, f_high], dim=1)
        return self.efpm_block(x)

class DAFDM(nn.Module):
    def __init__(self, input_channels=3, out_channels=1, weights=None):
        super().__init__()
        self.input_conv = (
            nn.Conv2d(input_channels, 3, kernel_size=1)
            if input_channels > 3 else nn.Identity()
        )

        # Encoder (VGG16)

        vgg = models.vgg16_bn(weights=None).features
        self.enc1 = vgg[:6]    # 64
        self.enc2 = vgg[6:13]  # 128
        self.enc3 = vgg[13:23] # 256
        self.enc4 = vgg[23:33] # 512
        self.enc5 = vgg[33:43] # 512

        self.efpm1 = EFPMBlock(512, 512)
        self.efpm2 = EFPM(512, 512, 256)
        self.efpm3 = EFPM(256, 256, 128)
        self.efpm4 = EFPM(128, 128, 64)
        self.efpm5 = EFPM(64, 64, 32)


        self.up1 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.up4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
      

        # Output (mapa final binário)
        self.head = nn.Sequential(
            nn.Conv2d(32, out_channels, 1),
            # nn.Sigmoid()
        )

    def forward(self, x):
        x = self.input_conv(x)

        # Encoder
        f1 = self.enc1(x)  # H
        f2 = self.enc2(f1) # H/2
        f3 = self.enc3(f2) # H/4
        f4 = self.enc4(f3) # H/8
        f5 = self.enc5(f4) # H/16


        # print(f'Shapes: - F1: {f1.shape} - F2: {f2.shape} - F3: {f3.shape} - F4: {f4.shape} - F5: {f5.shape}')

        # Decoder
        # d1 = self.efpm1(f4, f5)        # [B, 512, H/8, W/8]
        d1 = self.efpm1(f5)
        u1 = self.up1(d1)

        d2 = self.efpm2(f4, u1)       # [B, 256, H/4, W/4]
        u2 = self.up2(d2)             # → H/2

        d3 = self.efpm3(f3, u2)       # [B, 128, H/2, W/2]
        u3 = self.up3(d3)             # → H

        d4 = self.efpm4(f2, u3)       # [B, 64, H, W]
        u4 = self.up4(d4)             # → H*2

        d5 = self.efpm5(f1, u4)       # [B, 32, H*2, W*2]

        return self.head(d5)




class SensorNormalizerFiLM(nn.Module):
    """
    Normalizador por-sensor usando FiLM (Feature-wise Linear Modulation).
    Para cada sensor: aplica x' = gamma * x + beta (por canal).
    """
    def __init__(self, num_sensors: int = 2, num_channels: int = 3):
        super().__init__()
        self.num_sensors = num_sensors
        self.num_channels = num_channels

        # parâmetros FiLM (gamma e beta) por sensor e canal
        self.gamma = nn.ParameterList([
            nn.Parameter(torch.ones(num_channels)) for _ in range(num_sensors)
        ])
        self.beta = nn.ParameterList([
            nn.Parameter(torch.zeros(num_channels)) for _ in range(num_sensors)
        ])

    def forward(self, x: torch.Tensor, domain_id=0) -> torch.Tensor:
        """
        x: (B,C,H,W)
        domain_id: int ou tensor (B,) com IDs de sensor
        """
        B, C, H, W = x.shape
        device, dtype = x.device, x.dtype

        # normaliza domain_id
        if isinstance(domain_id, int):
            domain_id = torch.full((B,), domain_id, dtype=torch.long, device=device)
        else:
            domain_id = domain_id.to(device).long()

        out = torch.empty_like(x)
        for sid in range(self.num_sensors):
            mask = (domain_id == sid)
            if mask.any():
                gamma = self.gamma[sid].to(device=device, dtype=dtype).view(1, C, 1, 1)
                beta = self.beta[sid].to(device=device, dtype=dtype).view(1, C, 1, 1)
                out[mask] = gamma * x[mask] + beta
        return out

    def export_rules_numeric(self) -> Dict[str, Any]:
        rules = {}
        for sid in range(self.num_sensors):
            rules[f"sensor_{sid}"] = {
                "gamma": self.gamma[sid].detach().cpu().numpy().tolist(),
                "beta": self.beta[sid].detach().cpu().numpy().tolist(),
            }
        return rules

    def export_numpy_rule(self, domain_id: int = 0):
        gamma = self.gamma[domain_id].detach().cpu().numpy().astype(np.float32)
        beta = self.beta[domain_id].detach().cpu().numpy().astype(np.float32)

        def apply_rule(img: np.ndarray):
            # img: (H,W,C)
            return gamma.reshape(1, 1, -1) * img + beta.reshape(1, 1, -1)

        return apply_rule




class ThermalAnomalyBranchPP(nn.Module):
    def __init__(self, num_domains: int = 2):
        super().__init__()
        self.num_domains = num_domains


        # Índices espectrais — thresholds
        self.thr_stgi = nn.Parameter(torch.ones(num_domains) * 0.5)
        self.thr_acst = nn.Parameter(torch.ones(num_domains) * 0.5)
        self.thr_sai  = nn.Parameter(torch.ones(num_domains) * 0.5)
        self.thr_sati = nn.Parameter(torch.ones(num_domains) * 0.5)

        # Pesos de combinação (treináveis)
        self.w_stgi = nn.Parameter(torch.ones(num_domains) * 0.25)
        self.w_acst = nn.Parameter(torch.ones(num_domains) * 0.25)
        self.w_sai  = nn.Parameter(torch.ones(num_domains) * 0.25)
        self.w_sati = nn.Parameter(torch.ones(num_domains) * 0.25)

        # Parâmetros para função logística
        self.gamma_fire = nn.Parameter(torch.ones(num_domains) * 5.0)
        self.delta_fire = nn.Parameter(torch.zeros(num_domains) + 0.5)

        # Parâmetros para saturação térmica (antes eram constantes)
        self.alpha_sati = nn.Parameter(torch.ones(num_domains) * 3.0)
        self.beta_sati  = nn.Parameter(torch.ones(num_domains) * 0.3)

        # Parâmetro de contraste
        self.k_contrast = nn.Parameter(torch.ones(num_domains) * 10.0)

    # =============================================================
    # Forward
    # =============================================================
    def forward(self, x: torch.Tensor, domain_id: torch.Tensor):
        """
        Args:
            x: tensor [B, C, H, W] com bandas (espera SWIR2, SWIR1, NIR)
            domain_id: tensor [B] com id do domínio (0..num_domains-1)
        """
        swir2, swir1, nir = x[:, 0], x[:, 1], x[:, 2]
        B, H, W = swir2.shape

        # Seleciona parâmetros específicos de cada domínio
        thr_stgi = self.thr_stgi[domain_id].view(B, 1, 1)
        thr_acst = self.thr_acst[domain_id].view(B, 1, 1)
        thr_sai  = self.thr_sai[domain_id].view(B, 1, 1)
        thr_sati = self.thr_sati[domain_id].view(B, 1, 1)

        w_stgi = self.w_stgi[domain_id].view(B, 1, 1)
        w_acst = self.w_acst[domain_id].view(B, 1, 1)
        w_sai  = self.w_sai[domain_id].view(B, 1, 1)
        w_sati = self.w_sati[domain_id].view(B, 1, 1)

        gamma_fire = self.gamma_fire[domain_id].view(B, 1, 1)
        delta_fire = self.delta_fire[domain_id].view(B, 1, 1)

        alpha_sati = self.alpha_sati[domain_id].view(B, 1, 1)
        beta_sati  = self.beta_sati[domain_id].view(B, 1, 1)
        k_contrast = self.k_contrast[domain_id].view(B, 1, 1)

        # ===============================
        # Cálculo dos índices
        # ===============================
        stgi = (swir2 - swir1) / (swir2 + swir1 + 1e-6)
        acst = (swir2 - nir) / (swir2 + nir + 1e-6)
        sai  = (swir1 - nir) / (swir1 + nir + 1e-6)
        sati_raw = torch.tanh(alpha_sati * (swir2 - beta_sati * swir1))

        # ===============================
        # Máscaras binárias por índice
        # ===============================
        p_stgi = torch.sigmoid(k_contrast * (stgi - thr_stgi))
        p_acst = torch.sigmoid(k_contrast * (acst - thr_acst))
        p_sai  = torch.sigmoid(k_contrast * (sai - thr_sai))
        p_sati = torch.sigmoid(k_contrast * (sati_raw - thr_sati))

        # ===============================
        # Combinação ponderada
        # ===============================
        p_fire = (w_stgi * p_stgi + w_acst * p_acst + w_sai * p_sai + w_sati * p_sati)
        # p_fire = p_fire / (w_stgi + w_acst + w_sai + w_sati + 1e-6)

        # Camada de calibração final (aprende a suavizar a saída)
        final_prob = torch.sigmoid(gamma_fire * (p_fire - delta_fire))
        final_logits = torch.logit(final_prob.clamp(1e-6, 1 - 1e-6))

        return {
            "final_logits": final_logits.unsqueeze(1),  # [B,1,H,W]
            "final_prob": final_prob.unsqueeze(1),      # [B,1,H,W]
            "indices": {"stgi": stgi, "acst": acst, "sai": sai, "sati": sati_raw}
        }


    def export_numpy_rule(self, domain_id: int = 0):
        p = self.export_rules_numeric()[domain_id]

        def numpy_fire_mask(img):
            # img[..., 0]=SWIR2, img[..., 1]=SWIR1, img[..., 2]=NIR
            swir2 = img[..., 0].astype(np.float32)
            swir1 = img[..., 1].astype(np.float32)
            nir   = img[..., 2].astype(np.float32)

            # índices
            stgi = (swir2 - swir1) / (swir2 + swir1 + 1e-6)
            acst = (swir2 - nir) / (swir2 + nir + 1e-6)
            sai  = (swir1 - nir) / (swir1 + nir + 1e-6)
            sati = np.tanh(p["alpha_sati"] * (swir2 - p["beta_sati"] * swir1))

            # função sigmoide estável
            def sigmoid_stable_np(x):
                x = np.clip(x, -20, 20)
                return 1.0 / (1.0 + np.exp(-x))

            # probabilidades por índice
            p_stgi = sigmoid_stable_np(p["k_contrast"] * (stgi - p["thr_stgi"]))
            p_acst = sigmoid_stable_np(p["k_contrast"] * (acst - p["thr_acst"]))
            p_sai  = sigmoid_stable_np(p["k_contrast"] * (sai - p["thr_sai"]))
            p_sati = sigmoid_stable_np(p["k_contrast"] * (sati - p["thr_sati"]))

            # denom = (p["w_stgi"] + p["w_acst"] + p["w_sai"] + p["w_sati"] + 1e-6)
            # p_fire = (
            #     p["w_stgi"] * p_stgi + 
            #     p["w_acst"] * p_acst + 
            #     p["w_sai"] * p_sai + 
            #     p["w_sati"] * p_sati
            # ) / denom

            p_fire = (
                p["w_stgi"] * p_stgi + 
                p["w_acst"] * p_acst + 
                p["w_sai"] * p_sai + 
                p["w_sati"] * p_sati
            )

            # print(f'[DEBUG] p_fire min: {p_fire.min()}, max: {p_fire.max()}, mean: {p_fire.mean()}')
            final_prob = sigmoid_stable_np(p["gamma_fire"] * (p_fire - p["delta_fire"]))
            # print(f'[DEBUG] final_prob min: {final_prob.min()}, max: {final_prob.max()}, mean: {final_prob.mean()}')
            return final_prob
            # return p_fire
        
        return numpy_fire_mask

    # ============================================================
    # EXPORTAÇÃO DOS PARÂMETROS
    # ============================================================
    def export_rules_numeric(self):
        params = {}
        for d in range(self.num_domains):
            params[d] = {
                "thr_stgi": float(self.thr_stgi[d]),
                "thr_acst": float(self.thr_acst[d]),
                "thr_sai": float(self.thr_sai[d]),
                "thr_sati": float(self.thr_sati[d]),
                "w_stgi": float(self.w_stgi[d]),
                "w_acst": float(self.w_acst[d]),
                "w_sai": float(self.w_sai[d]),
                "w_sati": float(self.w_sati[d]),
                "alpha_sati": float(self.alpha_sati[d]),
                "beta_sati": float(self.beta_sati[d]),
                "gamma_fire": float(self.gamma_fire[d]),
                "delta_fire": float(self.delta_fire[d]),
                "k_contrast": float(self.k_contrast[d]),
            }
        return params



class ThermalAnomalyBranchWithNormalizer(nn.Module):
    def __init__(self, num_sensors: int = 2, num_channels: int = 3, use_normalizer : bool = True):
        super().__init__()
    
        self.normalizer = None
        if use_normalizer:
            self.normalizer = SensorNormalizerFiLM(2, num_channels)
        self.branch = ThermalAnomalyBranchPP(num_sensors)

    def forward(self, x: torch.Tensor, domain_id=0) -> torch.Tensor:
        if self.normalizer is not None:
            x_norm = self.normalizer(x, domain_id)
            # print('DEBUG', x_norm.shape)
            return self.branch(x_norm, domain_id)
        
        return self.branch(x, domain_id)
    
    def export_rules_numeric(self):
        return {
            "normalizer": self.normalizer.export_rules_numeric() if self.normalizer is not None else 'No Normalization',
            "branch": self.branch.export_rules_numeric(),
        }

    def export_numpy_rule(self, domain_id: int = 0):
        norm_fn = None
        if self.normalizer is not None:
            norm_fn = self.normalizer.export_numpy_rule(domain_id)
        branch_fn = self.branch.export_numpy_rule(domain_id)

        def apply_rule(img: np.ndarray):
            if norm_fn is not None:
                img_norm = norm_fn(img)
                return branch_fn(img_norm)
            
            return branch_fn(img)

        return apply_rule
    

class LitDTBranchAdvanced(L.LightningModule):
    def __init__(self, backbone='DAFDM', lr=1e-3, alpha=0.5, loss_b_weight=1.0, loss_dt_weight=1.0, loss_fused_weight=1.0):
        super().__init__()
        self.backbone_type = backbone
        self.lr = lr
        self.loss_b_weight = loss_b_weight
        self.loss_dt_weight = loss_dt_weight
        self.loss_fused_weight = loss_fused_weight
        
        self.save_hyperparameters()

        if self.backbone_type is None:
            self.backbone = None
        elif self.backbone_type.lower() == 'dafdm':
            self.backbone = DAFDM(input_channels=3, out_channels=1)
        else:
            raise ValueError(f'Backbone {self.backbone_type} não implementada.')

        self.threshold_branch = None
        if not DISABLE_THRESHOLD_BRANCH:
            self.threshold_branch = ThermalAnomalyBranchWithNormalizer(num_sensors=NUM_SENSORS_THREHOLD_EQUATIONS, use_normalizer=USE_FILM)
        
        self.loss_fn = nn.BCEWithLogitsLoss()
        # self.loss_fn = nn.BCELoss()

        self.alpha = nn.Parameter(torch.tensor(alpha))  # learnable weight for fusion

    def forward(self, x, domain_id=None):

        if self.threshold_branch is None:
            logits_backbone = self.backbone(x)                 # logits (B,1,H,W)
            probs_backbone = torch.sigmoid(logits_backbone)
            return {
                'backbone_logits': logits_backbone,
                'backbone_prob': probs_backbone,
                'dt_logits': None,
                'dt_prob': None,
                'fused_logits': None,
                'fused_prob': None
            }


        dt_out = self.threshold_branch(x, domain_id)            # dict {'final_logits', 'final_prob'}
        dt_logits = dt_out['final_logits']
        dt_prob = dt_out['final_prob']
    
        if self.backbone is None:
            return {
                'backbone_logits': None,
                'backbone_prob': None,
                'dt_logits': dt_logits,
                'dt_prob': dt_prob,
                'fused_logits': dt_logits,
                'fused_prob': dt_prob
            }
        
        logits_unet = self.backbone(x)                 # logits (B,1,H,W)
        probs_unet = torch.sigmoid(logits_unet)
        # fused logits: combine logits in probability space then convert to logits
        fused_prob = torch.sigmoid(self.alpha) * probs_unet + (1.0 - torch.sigmoid(self.alpha)) * dt_prob
        # for loss fused -> convert fused_prob to logits via logit (clamp)
        fused_prob_clamped = fused_prob.clamp(1e-6, 1-1e-6)
        fused_logits = torch.log(fused_prob_clamped / (1 - fused_prob_clamped))
        return {
            'backbone_logits': logits_unet,
            'backbone_prob': probs_unet,
            'dt_logits': dt_logits,
            'dt_prob': dt_prob,
            'fused_logits': fused_logits,
            'fused_prob': fused_prob
        }

    def training_step(self, batch, batch_idx):
        x, y, domain_id = batch['image'], batch['mask'], batch['domain_id']
        out = self(x, domain_id)

        if self.threshold_branch is None:
            loss = self.loss_fn(out['backbone_logits'], y)
            f1 = self.f1_score(y, torch.sigmoid(out['backbone_logits']))
            self.log("train_loss", loss, prog_bar=True)
            self.log("train_f1", f1, prog_bar=True)
            return loss
        
        if self.backbone is None:
            loss = self.loss_fn(out['dt_logits'], y)
            f1 = self.f1_score(y, torch.sigmoid(out['dt_logits']))
            self.log("train_loss", loss, prog_bar=True)
            self.log("train_f1", f1, prog_bar=True)
            return loss
        
        loss_backbone = self.loss_fn(out['backbone_logits'], y)
        loss_threshold = self.loss_fn(out['dt_logits'], y)
        loss_fused = self.loss_fn(out['fused_logits'], y)

        loss = self.loss_b_weight*loss_backbone + self.loss_dt_weight*loss_threshold + self.loss_fused_weight*loss_fused

        if self.threshold_branch.normalizer is not None:
            # small regularizers: keep normalizer near identity
            gammas = self.threshold_branch.normalizer.gamma
            betas = self.threshold_branch.normalizer.beta
            reg_norm = 0.0
            for g in gammas:
                reg_norm = reg_norm + ((g - 1.0) ** 2).mean()
            for b in betas:
                reg_norm = reg_norm + (b ** 2).mean()
            loss = loss + 1e-3 * reg_norm

        f1 = self.f1_score(y, torch.sigmoid(out['fused_logits']))
        self.log("train_f1", f1, prog_bar=True)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_loss_backbone", loss_backbone, prog_bar=True)
        self.log("train_loss_threshold", loss_threshold, prog_bar=True)
        self.log("train_loss_fused", loss_fused, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y, domain_id = batch['image'], batch['mask'], batch['domain_id']
        out = self(x, domain_id)

        if self.threshold_branch is None:
            loss = self.loss_fn(out['backbone_logits'], y)
            f1 = self.f1_score(y, torch.sigmoid(out['backbone_logits']))
            self.log("val_loss", loss, prog_bar=True)
            self.log("val_f1", f1, prog_bar=True)
            return loss
        
        if self.backbone is None:
            loss = self.loss_fn(out['dt_logits'], y)
            f1 = self.f1_score(y, torch.sigmoid(out['dt_logits']))
            self.log("val_loss", loss, prog_bar=True)
            self.log("val_f1", f1, prog_bar=True)
            return loss

        
        loss_backbone = self.loss_fn(out['backbone_logits'], y)
        loss_threshold = self.loss_fn(out['dt_logits'], y)
        loss_fused = self.loss_fn(out['fused_logits'], y)

        loss = self.loss_b_weight*loss_backbone + self.loss_dt_weight*loss_threshold + self.loss_fused_weight*loss_fused
        
        if self.threshold_branch.normalizer is not None:
            # small regularizers: keep normalizer near identity
            gammas = self.threshold_branch.normalizer.gamma
            betas = self.threshold_branch.normalizer.beta
            reg_norm = 0.0
            for g in gammas:
                reg_norm = reg_norm + ((g - 1.0) ** 2).mean()
            for b in betas:
                reg_norm = reg_norm + (b ** 2).mean()
            loss = loss + 1e-3 * reg_norm

        f1 = self.f1_score(y, torch.sigmoid(out['fused_logits']))
        self.log("val_f1", f1, prog_bar=True)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_loss_backbone", loss_backbone, prog_bar=True)
        self.log("val_loss_threshold", loss_threshold, prog_bar=True)
        self.log("val_loss_fused", loss_fused, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y, domain_id = batch['image'], batch['mask'], batch['domain_id']
        out = self(x, domain_id)
        if self.threshold_branch is None:
            loss = self.loss_fn(out['backbone_logits'], y)
            f1 = self.f1_score(y, torch.sigmoid(out['backbone_logits']))
            self.log("test_loss", loss, prog_bar=True)
            self.log("test_f1", f1, prog_bar=True)
            return loss
        
        if self.backbone is None:
            loss = self.loss_fn(out['dt_logits'], y)
            f1 = self.f1_score(y, torch.sigmoid(out['dt_logits']))
            self.log("test_loss", loss, prog_bar=True)
            self.log("test_f1", f1, prog_bar=True)
            return loss

        loss = self.loss_fn(out['fused_logits'], y)
        loss_backbone = self.loss_fn(out['backbone_logits'], y)
        loss_threshold = self.loss_fn(out['dt_logits'], y)
        loss_fused = self.loss_fn(out['fused_logits'], y)

        loss = self.loss_b_weight*loss_backbone + self.loss_dt_weight*loss_threshold + self.loss_fused_weight*loss_fused

        if self.threshold_branch.normalizer is not None:
            # small regularizers: keep normalizer near identity
            gammas = self.threshold_branch.normalizer.gamma
            betas = self.threshold_branch.normalizer.beta
            reg_norm = 0.0
            for g in gammas:
                reg_norm = reg_norm + ((g - 1.0) ** 2).mean()
            for b in betas:
                reg_norm = reg_norm + (b ** 2).mean()
            loss = loss + 1e-3 * reg_norm

        f1 = self.f1_score(y, torch.sigmoid(out['fused_logits']))
        self.log("test_f1", f1, prog_bar=True)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_loss_backbone", loss_backbone, prog_bar=True)
        self.log("test_loss_threshold", loss_threshold, prog_bar=True)
        self.log("test_loss_fused", loss_fused, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}
        
    
    def __call__(self, x, domain_id=None):
        return self.forward(x, domain_id=domain_id)
    
    @staticmethod
    def f1_score(y_true, y_pred):
        y_pred = (y_pred > 0.5).float()
        tp = (y_true * y_pred).sum().to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)

        if precision + recall == 0:
            return torch.tensor(0.0)

        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        return f1

def build_datasets():
    print('[INFO] Preprando dataset landsat-sentinel')
    transform_train = RandomFlipRotate()
    transform = None 

    root_dir_map = {'landsat': LANDSAT_DATASET_PATH, 'sentinel': SENTINEL_DATASET_PATH, }
    mask_folder_map = {'landsat': LANDSAT_ANNOTATION_FOLDER, 'sentinel': SENTINEL_ANNOTATION_FOLDER,}
    img_folder_map = {'landsat': LANDSAT_IMG_FOLDER, 'sentinel': SENTINEL_IMG_FOLDER,}
    bands = {'landsat': LANDSAT_BANDS, 'sentinel': SENTINEL_BANDS,}
    
    quantification = {'landsat': LANDSAT_QUANTIFICATION, 'sentinel': SENTINEL_QUANTIFICATION}
    mean_std = None
   
    train_dataset = MultiSatelliteDataset(LANDSAT_SENTINEL_DATAFRAME_PATH, root_dir_map, mask_folder_map, img_folder_map, bands, quantification, fold=FOLD, set='train', transform=transform_train, means_stds=mean_std)
    val_dataset = MultiSatelliteDataset(LANDSAT_SENTINEL_DATAFRAME_PATH, root_dir_map, mask_folder_map, img_folder_map, bands, quantification, fold=FOLD, set='validation', transform=transform, means_stds=mean_std)


    test_dataset_full = MultiSatelliteDataset(LANDSAT_SENTINEL_DATAFRAME_PATH, root_dir_map, mask_folder_map, img_folder_map, bands, quantification, fold=FOLD, set='test', transform=transform, means_stds=mean_std)
    test_dataset_landsat = MultiSatelliteDataset(LANDSAT_DATAFRAME_PATH, root_dir_map, mask_folder_map, img_folder_map, bands, quantification, fold=FOLD, set='test', transform=transform, means_stds=mean_std)
    test_dataset_sentinel = MultiSatelliteDataset(SENTINEL_DATAFRAME_PATH, root_dir_map, mask_folder_map, img_folder_map, bands, quantification, fold=FOLD, set='test', transform=transform, means_stds=mean_std)


    return train_dataset, val_dataset, test_dataset_full, test_dataset_landsat, test_dataset_sentinel




def evaluate_dataset(trainer, module, test_dataset, batch_size=1, workers=4, keys=['fused_prob', 'dt_prob', 'backbone_prob']):
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=workers)
    module.eval()

    if BACKBONE is None:
        keys = ['dt_prob']

    if DISABLE_THRESHOLD_BRANCH and 'dt_prob' in keys:
        keys = ['backbone_prob']


    final_metrics = None
    if trainer is not None:
        print('[INFO] Iniciando avaliação...')
        final_metrics = trainer.test(module, test_dataloader)
        print('[INFO] Avaliação finalizada')
    
    print(f'[INFO] Computando métricas...')
    # cm_matrix = BinaryConfusionMatrix()    
    # test_precision = Precision(task="binary")
    # test_recall = Recall(task="binary")
    # test_f1 = F1Score(task="binary")

    precision = {}
    recall = {}
    f1 = {}
    cm = {}

    for key in keys:
        precision[key] = Precision(task="binary")
        recall[key] = Recall(task="binary")
        f1[key] = F1Score(task="binary")
        cm[key] = BinaryConfusionMatrix()

    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            images = batch['image'].to(module.device)
            masks_batch = (batch['mask'].to(module.device) > 0.5).float()
            # sensor_id = batch['satellite'].argmax(dim=1).to(module.device)  # Converte para o tipo correto
            # masks_batch = masks_batch.unsqueeze(1)  # Adiciona a dimensão de canal
            domain_id = batch['domain_id'].to(module.device)

            y_hat = module(images, domain_id=domain_id)
            # y_hat = torch.sigmoid(y_hat).squeeze(1)  # Remove a dimensão de canal
            # y_hat = torch.sigmoid(y_hat)
            for key in keys:
                probs = y_hat[key]
                preds = (probs > 0.5).float()

                precision[key].update(preds, masks_batch)
                recall[key].update(preds, masks_batch)
                f1[key].update(preds, masks_batch)
                cm[key].update(preds, masks_batch)

                # comparações entre branches
                for key2 in keys:
                    if key == key2:
                        continue

                    pair_key = f"{key}_vs_{key2}"
                    if pair_key not in precision:
                        precision[pair_key] = Precision(task="binary")
                        recall[pair_key] = Recall(task="binary")
                        f1[pair_key] = F1Score(task="binary")
                        cm[pair_key] = BinaryConfusionMatrix()
                    
                    preds2 = (y_hat[key2] > 0.5).float()
                    precision[pair_key].update(preds, preds2)
                    recall[pair_key].update(preds, preds2)
                    f1[pair_key].update(preds, preds2)
                    cm[pair_key].update(preds, preds2)



    precision = {k: v.compute() for k, v in precision.items()}
    recall = {k: v.compute() for k, v in recall.items()}
    f1 = {k: v.compute() for k, v in f1.items()}
    cm = {k: v.compute() for k, v in cm.items()}
    
    module.train()

    return precision, recall, f1, cm, final_metrics


def evaluate_numpy_rule(module, test_dataset):
    """
    Avalia regras numpy exportadas para todos os sensores presentes no test_dataset.
    Assume que cada sample contém ['image', 'mask', 'sensor_id'].
    """
    all_preds = []
    all_gts = []

    for i in tqdm(range(len(test_dataset)), total=len(test_dataset)):
        sample = test_dataset[i]
        img = sample['image'].numpy().transpose(1,2,0)  # (H,W,3)
        gt = sample['mask'].numpy().squeeze(0)          # (H,W)

        gt = (gt > 0.5).astype(np.uint8)  # binariza ground truth

        sensor_id = int(sample['domain_id'])     # 0 = Sentinel, 1 = Landsat (por exemplo)

        # exporta regra específica para o sensor
        rule_fn = module.threshold_branch.export_numpy_rule(sensor_id)
        pred_mask = rule_fn(img)  # (H,W) float [0,1]

        # binariza (threshold fixo, pode calibrar depois)
        pred_bin = (pred_mask > 0.5).astype(np.uint8)

        all_preds.append(pred_bin.flatten())
        all_gts.append(gt.flatten())

    all_preds = np.concatenate(all_preds)
    all_gts = np.concatenate(all_gts)

    print("[DEBUG] Valores únicos em all_gts:", np.unique(all_gts), 'Shape:', all_gts.shape)
    print("[DEBUG] Valores únicos em all_preds:", np.unique(all_preds), 'Shape:', all_preds.shape)

    # Garante que os valores estão no intervalo esperado (0 ou 1)
    assert set(np.unique(all_gts)).issubset({0, 1}), "all_gts contém valores inesperados!"
    assert set(np.unique(all_preds)).issubset({0, 1}), "all_preds contém valores inesperados!"

    # Corrige valores inesperados em all_gts e all_preds
    all_gts = np.clip(all_gts, 0, 1)
    all_preds = np.clip(all_preds, 0, 1)

    precision = precision_score(all_gts, all_preds, zero_division=0)
    recall = recall_score(all_gts, all_preds, zero_division=0)
    f1 = f1_score(all_gts, all_preds, zero_division=0)
    cm = confusion_matrix(all_gts, all_preds)

    return precision, recall, f1, cm, None




def save_results_to_file(file_path, dataset_name, final_metrics, precision, recall, f1, cm_matrix):
    with open(file_path, 'a+') as f:
        f.write('-'*80 + '\n')
        f.write(f'Dataset: {dataset_name}\n')
        f.write(str(final_metrics))
        f.write(f'\nSklearn Metrics:\n')
        f.write(f'Precision: {str(precision)}\n')	
        f.write(f'Recall: {str(recall)}\n')
        f.write(f'F1: {str(f1)}\n')
        f.write(f'Confusion Matrix:\n')
        f.write(str(cm_matrix))
        f.write('\n')

if __name__ == "__main__":
   
    log_dir = os.path.join(LOG_DIR, f'fold-{FOLD}')
    os.makedirs(log_dir, exist_ok=True)
    LOG_DIR = log_dir
   
    train_dataset, val_dataset, test_dataset_full, test_dataset_landsat, test_dataset_sentinel = build_datasets()

    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    val_sampler = torch.utils.data.SequentialSampler(val_dataset)


    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        sampler=train_sampler,
        num_workers=WORKERS, 
        drop_last=True
    )

    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler, num_workers=WORKERS)


    module = LitDTBranchAdvanced(backbone=BACKBONE, lr=LR, loss_b_weight=LOSS_B_WEIGHT, loss_dt_weight=LOSS_DT_WEIGHT, loss_fused_weight=LOSS_FUSED_WEIGHT)

    early_stop_callback = EarlyStopping(monitor="val_loss", patience=ES_PATIENCE, verbose=False, mode="min")
    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(LOG_DIR, 'checkpoints'), save_top_k=1, monitor="val_loss", verbose=False, mode="min")


    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    trainer = Trainer(default_root_dir=LOG_DIR, check_val_every_n_epoch=1, log_every_n_steps=1, accelerator=accelerator, max_epochs=EPOCHS, devices=DEVICES, callbacks=[early_stop_callback, checkpoint_callback], deterministic=True)
    
    print('[INFO] Iniciando treinamento...')
    

    if OVERRIDE_TRAINED or not os.path.exists(os.path.join(LOG_DIR, 'checkpoints', 'final_model.ckpt')):
        start_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        trainer.fit(model=module, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
        end_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print('[INFO] Treinamento finalizado')
        
        trainer.save_checkpoint(os.path.join(LOG_DIR, 'checkpoints', 'last_trained_epoch.ckpt'))
        best_ckpt_path = checkpoint_callback.best_model_path
        if best_ckpt_path and os.path.exists(best_ckpt_path):
            print(f'[INFO] Melhor modelo encontrado em: {best_ckpt_path}')
            
            # Define o caminho final
            final_path = os.path.join(log_dir, 'checkpoints', 'final_model.ckpt')
            
            # Copia o arquivo do melhor checkpoint para o nome 'final_model.ckpt'
            shutil.copy(best_ckpt_path, final_path)
            print(f'[INFO] Melhor modelo copiado para: {final_path}')
        else:
            print('[WARN] Nenhum checkpoint encontrado.')

    else:
        print('[INFO] Carregando modelo treinado...')
        module = LitDTBranchAdvanced.load_from_checkpoint(os.path.join(LOG_DIR, 'checkpoints', 'final_model.ckpt'))
        print('[INFO] Modelo treinado carregado!')
        trainer = None  # Não precisamos do trainer para avaliação  


    if module.threshold_branch is not None:
        print(f'[INFO] Exportando regras numéricas...')
        rules = module.threshold_branch.export_rules_numeric()
        with open(os.path.join(LOG_DIR, 'exported_rules.json'), 'w') as f:
            json.dump(rules, f, indent=4)

    print(f'[INFO] Avaliando dataset Landsat-Sentinel...')  
    precision, recall, f1, cm_matrix, final_metrics = evaluate_dataset(trainer, module, test_dataset_full, batch_size=BATCH_SIZE, workers=WORKERS)
    print(f'[INFO] Salvando resultados Landsat-Sentinel...')
    save_results_to_file(os.path.join(LOG_DIR, 'final_results.txt'), 'Landsat-Sentinel', final_metrics, precision, recall, f1, cm_matrix)
        
    
    print(f'[INFO] Avaliando dataset Landsat...')
    precision, recall, f1, cm_matrix, final_metrics = evaluate_dataset(trainer, module, test_dataset_landsat, batch_size=BATCH_SIZE, workers=WORKERS)
    print(f'[INFO] Salvando resultados Landsat...')
    save_results_to_file(os.path.join(LOG_DIR, 'final_results.txt'), 'Landsat', final_metrics, precision, recall, f1, cm_matrix)


    print(f'[INFO] Avaliando dataset Sentinel...')
    precision, recall, f1, cm_matrix, final_metrics = evaluate_dataset(trainer, module, test_dataset_sentinel, batch_size=BATCH_SIZE, workers=WORKERS)
    print(f'[INFO] Salvando resultados - Sentinel...')
    save_results_to_file(os.path.join(LOG_DIR, 'final_results.txt'), 'Sentinel', final_metrics, precision, recall, f1, cm_matrix)


    print(f'[INFO] Processo finalizado!')


