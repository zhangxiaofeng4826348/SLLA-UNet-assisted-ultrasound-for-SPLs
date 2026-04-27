# ===============================================
# loss.py
# ===============================================

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------- Class-Balanced Focal Loss ----------------
def effective_num_weight(counts, beta=0.999):
    counts = torch.tensor(counts, dtype=torch.float32)
    eff_num = 1.0 - torch.pow(beta, counts)
    weights = (1.0 - beta) / (eff_num + 1e-12)
    weights = weights / weights.sum() * len(counts)
    return weights


class CBFocalLoss(nn.Module):
    def __init__(self, num_benign, num_malignant, gamma=1.0, beta=0.999, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        w = effective_num_weight([num_benign, num_malignant], beta)
        self.register_buffer('class_weight', w)

    def forward(self, inputs, targets):
        # inputs: [B,2] logits
        # targets: [B] long
        log_prob = F.log_softmax(inputs, dim=1)
        prob = log_prob.exp()

        logpt = log_prob.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt    = prob.gather(1, targets.unsqueeze(1)).squeeze(1)

        class_weight = self.class_weight.to(inputs.device)
        weights = class_weight[targets]

        focal = (1.0 - pt).clamp(min=1e-6).pow(self.gamma)
        loss = -weights * focal * logpt

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# ---------------- Dice Loss ----------------
class DiceLoss(nn.Module):
    """Soft Dice Loss"""
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        targets = targets.float()
        num = 2.0 * (probs * targets).sum(dim=(1, 2, 3)) + self.eps
        den = (probs.pow(2).sum(dim=(1, 2, 3)) + targets.pow(2).sum(dim=(1, 2, 3))) + self.eps
        dice = num / den
        loss = 1.0 - dice
        return loss.mean()


class SegLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.dice = DiceLoss(eps=eps)

    def forward(self, seg_logits: torch.Tensor, seg_targets: torch.Tensor) -> torch.Tensor:
        seg_targets = seg_targets.float()
        return self.dice(seg_logits, seg_targets)

    def forward_with_components(self, seg_logits, seg_targets):
        seg_targets = seg_targets.float()
        dice_loss = self.dice(seg_logits, seg_targets)
        return dice_loss, {'dice': float(dice_loss.detach())}


# ---------------- Combined Loss ----------------
class CombinedLoss(nn.Module):
    """

    """
    def __init__(self, num_benign, num_malignant, alpha=1.0, beta=1.0, gamma=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.cls_loss = CBFocalLoss(num_benign=num_benign, num_malignant=num_malignant, gamma=gamma)
        self.seg_loss = SegLoss()

    def forward(self, cls_pred, cls_target, seg_pred, seg_target):
        cls_loss_val = self.cls_loss(cls_pred, cls_target)
        seg_loss_val = self.seg_loss(seg_pred, seg_target)
        total_loss = self.alpha * cls_loss_val + self.beta * seg_loss_val
        return total_loss
