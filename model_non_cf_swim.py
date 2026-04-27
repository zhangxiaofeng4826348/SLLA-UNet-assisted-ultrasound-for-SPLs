import torch
import torch.nn as nn
import torch.nn.functional as F

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

# ---------------- GeM ----------------
class GeM(nn.Module):
    """Generalized Mean Pooling"""
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        x = F.adaptive_avg_pool2d(x, 1).pow(1.0 / self.p)
        return x.view(x.size(0), -1)

# ---------------- UNet Ablation----------------
class UNet_Ablation_non_cf(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        # -------- Encoder --------
        self.R1 = ResidualBlock(3, 32)
        self.R2 = ResidualBlock(32, 64)
        self.R3 = ResidualBlock(64, 128)
        self.R4 = ResidualBlock(128, 256)

        # -------- Bottleneck --------
        self.Bott = ResidualBlock(256, 512)

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
        self.seg_head = nn.Conv2d(32, 1, 1)  # 分割输出


        self.gem = GeM()
        cls_in_dim = 512
        self.cls_head = nn.Sequential(
            nn.Linear(cls_in_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # -------- Encoder --------
        R1 = self.R1(x)
        R2 = self.R2(self.pool(R1))
        R3 = self.R3(self.pool(R2))
        R4 = self.R4(self.pool(R3))

        # -------- Bottleneck --------
        Bott = self.Bott(self.pool(R4))

        # -------- Decoder (Segmentation) --------
        D1 = self.CD1(torch.cat([self.up(self.U1(Bott)), R4], dim=1))
        D2 = self.CD2(torch.cat([self.U2(self.up(D1)), R3], dim=1))
        D3 = self.CD3(torch.cat([self.U3(self.up(D2)), R2], dim=1))
        D4 = self.CD4(torch.cat([self.U4(self.up(D3)), R1], dim=1))
        seg_logits = torch.sigmoid(self.seg_head(D4))


        pooled_feats = self.gem(Bott).flatten(1)
        cls_output = self.cls_head(pooled_feats)

        return seg_logits, cls_output, pooled_feats


# ---------------- Projection Head ----------------
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


if __name__ == "__main__":
    model = UNet_Ablation_non_cf()

