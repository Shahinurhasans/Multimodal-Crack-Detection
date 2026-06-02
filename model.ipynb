"""
models.py — All model architecture definitions.

Models:
  ─ AudioCNN              : lightweight CNN over mel-spectrograms
  ─ FusionModel           : gated multimodal fusion (original)
  ─ ImageBranch           : EfficientNetB0 image feature extractor
  ─ AudioBranch           : multi-block mel-CNN feature extractor
  ─ CrossAttentionFusion  : bidirectional cross-attention fusion head
  ─ MultimodalFusionModel : full cross-attention model (image+audio)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

from config import cfg


# ══════════════════════════════════════════════════════════════════════════════
# Original gated-fusion pipeline
# ══════════════════════════════════════════════════════════════════════════════

class AudioCNN(nn.Module):
    """
    Small CNN over a (1, N_MELS, T) mel-spectrogram.
    When used as a feature extractor, replace fc2 with nn.Identity().
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.adapt = nn.AdaptiveAvgPool2d((8, 8))
        self.fc1   = nn.Linear(32 * 8 * 8, 128)
        self.fc2   = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.adapt(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class FusionModel(nn.Module):
    """
    Gated multimodal fusion.
      - image_proj : 2048 → 256
      - audio_proj :  128 → 256
      - gate       : 512  → 256  (sigmoid)
      - fc1/fc2    : 512 → 256 → 2
    """
    def __init__(self):
        super().__init__()
        self.image_proj = nn.Linear(2048, 256)
        self.audio_proj = nn.Linear(128, 256)
        self.relu        = nn.ReLU()
        self.dropout     = nn.Dropout(0.3)
        self.gate        = nn.Linear(512, 256)
        self.fc1         = nn.Linear(512, 256)
        self.fc2         = nn.Linear(256, 2)

    def forward(self, img_feat, aud_feat):
        img_feat = self.relu(self.image_proj(img_feat))
        aud_feat = self.relu(self.audio_proj(aud_feat))
        fused    = torch.cat((img_feat, aud_feat), dim=1)
        gate     = torch.sigmoid(self.gate(fused))
        img_feat = img_feat * gate
        aud_feat = aud_feat * (1 - gate)
        x        = torch.cat((img_feat, aud_feat), dim=1)
        x        = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)


def build_resnet50_encoder():
    """
    ResNet50 with fc replaced by Identity (outputs 2048-d features).
    layer4 is unfrozen for fine-tuning; all other layers are frozen.
    """
    resnet = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V1)
    resnet.fc = nn.Identity()
    for p in resnet.parameters():
        p.requires_grad = False
    for p in resnet.layer4.parameters():
        p.requires_grad = True
    return resnet


def build_audio_encoder():
    """AudioCNN with fc2 replaced by Identity (outputs 128-d features)."""
    model = AudioCNN()
    model.fc2 = nn.Identity()
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Advanced cross-attention model
# ══════════════════════════════════════════════════════════════════════════════

class ImageBranch(nn.Module):
    """EfficientNetB0 backbone → 1280-d global average-pooled features."""
    def __init__(self, out_dim: int = cfg.IMG_FEAT_DIM, dropout: float = cfg.DROPOUT):
        super().__init__()
        base = tv_models.efficientnet_b0(
            weights=tv_models.EfficientNet_B0_Weights.DEFAULT)
        self.backbone = base.features
        self.pool     = nn.AdaptiveAvgPool2d(1)
        self.dropout  = nn.Dropout(dropout)
        self.out_dim  = out_dim

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x).flatten(1)
        return self.dropout(x)


class AudioBranch(nn.Module):
    """Multi-block residual-style CNN for (1, N_MELS, T) mel-spectrograms."""
    def __init__(self, out_dim: int = cfg.AUD_FEAT_DIM, dropout: float = cfg.DROPOUT):
        super().__init__()

        def _block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            )

        self.enc = nn.Sequential(
            _block(1,   32),  nn.MaxPool2d(2),
            _block(32,  64),  nn.MaxPool2d(2),
            _block(64,  128), nn.MaxPool2d(2),
            _block(128, 256),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 16, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.head(self.enc(x))


class CrossAttentionFusion(nn.Module):
    """
    Bidirectional cross-attention fusion:
      - image queries attend over audio  → image-enriched features
      - audio queries attend over image  → audio-enriched features
    Fused representation → binary classifier head.
    """
    def __init__(
        self,
        img_dim:    int = cfg.IMG_FEAT_DIM,
        aud_dim:    int = cfg.AUD_FEAT_DIM,
        fusion_dim: int = cfg.FUSION_DIM,
        num_heads:  int = 8,
        dropout:    float = cfg.DROPOUT,
    ):
        super().__init__()
        self.img_proj = nn.Sequential(nn.Linear(img_dim, fusion_dim),
                                       nn.LayerNorm(fusion_dim))
        self.aud_proj = nn.Sequential(nn.Linear(aud_dim, fusion_dim),
                                       nn.LayerNorm(fusion_dim))

        self.img2aud = nn.MultiheadAttention(fusion_dim, num_heads,
                                              dropout=dropout, batch_first=True)
        self.aud2img = nn.MultiheadAttention(fusion_dim, num_heads,
                                              dropout=dropout, batch_first=True)

        self.norm_img = nn.LayerNorm(fusion_dim)
        self.norm_aud = nn.LayerNorm(fusion_dim)

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),      # used with BCEWithLogitsLoss
        )

    def forward(self, img_feat, aud_feat):
        img = self.img_proj(img_feat).unsqueeze(1)   # [B, 1, D]
        aud = self.aud_proj(aud_feat).unsqueeze(1)

        img_enr, _ = self.img2aud(query=img, key=aud, value=aud)
        aud_enr, _ = self.aud2img(query=aud, key=img, value=img)

        img_enr = self.norm_img(img + img_enr).squeeze(1)   # [B, D]
        aud_enr = self.norm_aud(aud + aud_enr).squeeze(1)

        fused = torch.cat([img_enr, aud_enr], dim=-1)       # [B, 2D]
        return self.classifier(fused).squeeze(-1)            # [B]


class MultimodalFusionModel(nn.Module):
    """Complete cross-attention multimodal model."""
    def __init__(self):
        super().__init__()
        self.image_branch = ImageBranch()
        self.audio_branch = AudioBranch()
        self.fusion = CrossAttentionFusion(
            img_dim    = cfg.IMG_FEAT_DIM,
            aud_dim    = cfg.AUD_FEAT_DIM,
            fusion_dim = cfg.FUSION_DIM,
        )

    def forward(self, img, mel):
        img_feat = self.image_branch(img)
        aud_feat = self.audio_branch(mel)
        return self.fusion(img_feat, aud_feat)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
