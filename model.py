import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.swin_transformer import swin_t

# ---------------- Residual Block ----------------
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch)
        )
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.relu(self.conv(x) + self.shortcut(x))


# ---------------- UNet + Swin (Pretrain, CAM-ready) ----------------
class UNet_Swin_Pretrain(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        # -------- Encoder --------
        self.R1 = ResidualBlock(3, 32)
        self.R2 = ResidualBlock(32, 64)
        self.R3 = ResidualBlock(64, 128)
        self.R4 = ResidualBlock(128, 256)

        # --------Residual Semantic Block--------
        self.RS = ResidualBlock(256, 512)

        # -------- Swin Transformer --------
        self.swin = swin_t(weights=None)
        self.swin.head = nn.Identity()
        self.swin_out_dim = 768

        # -------- Decoder (Segmentation) --------
        self.U1 = nn.Conv2d(512, 256, 1)
        self.U2 = nn.Conv2d(256, 128, 1)
        self.U3 = nn.Conv2d(128, 64, 1)
        self.U4 = nn.Conv2d(64, 32, 1)

        self.CD1 = ResidualBlock(512, 256)
        self.CD2 = ResidualBlock(256, 128)
        self.CD3 = ResidualBlock(128, 64)
        self.CD4 = ResidualBlock(64, 32)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.seg_head = nn.Conv2d(32, 1, 1)

        # -------- Classification / SSL head --------
        self.gem = nn.AdaptiveAvgPool2d(1)
        cls_in_dim = (32 + 64 + 128 + 256 + 512) + self.swin_out_dim
        self.cls_head = nn.Sequential(
            nn.Linear(cls_in_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        # -------- Encoder --------
        R1 = self.R1(x)
        R2 = self.R2(self.pool(R1))
        R3 = self.R3(self.pool(R2))
        R4 = self.R4(self.pool(R3))

        # -------- Residual Semantic Block --------
        RS = self.RS(self.pool(R4))

        # -------- Swin --------
        swin_input = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=True)
        swin_feat = self.swin(swin_input)  # [B, 768]

        # -------- Decoder --------
        D1 = self.CD1(torch.cat([self.up(self.U1(RS)), R4], dim=1))
        D2 = self.CD2(torch.cat([self.U2(self.up(D1)), R3], dim=1))
        D3 = self.CD3(torch.cat([self.U3(self.up(D2)), R2], dim=1))
        D4 = self.CD4(torch.cat([self.U4(self.up(D3)), R1], dim=1))
        seg_logits = self.seg_head(D4)

        # -------- Global Features (for SSL / cls) --------
        pooled_feats = [
            self.gem(f).flatten(1)
            for f in [R1, R2, R3, R4, RS]
        ]
        pooled_feats = torch.cat(pooled_feats + [swin_feat], dim=1)

        cls_output = self.cls_head(pooled_feats)

        # -------- CAM --------
        cam_feats = {
            'R1': R1,
            'R2': R2,
            'R3': R3,
            'R4': R4,
            'RS': RS,
            'Fused': pooled_feats
        }

        return seg_logits, cls_output, pooled_feats, swin_feat, cam_feats


# ---------------- Projection Head (SSL) ----------------
class ProjectionHead(nn.Module):
    def __init__(self, in_dim, proj_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, proj_dim)
        )

    def forward(self, x):
        return self.net(x)


# ---------------- Test ----------------
if __name__ == "__main__":
    model = UNet_Swin_Pretrain()

