# ===============================================
# NT-Xent Loss (SimCLR / SSL)
# ===============================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss
    (SimCLR-style contrastive loss)

    Input:
        z1: [B, D]  projection of view 1
        z2: [B, D]  projection of view 2

    Output:
        scalar loss
    """

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z1, z2: shape [B, D]
        """

        assert z1.shape == z2.shape, "z1 and z2 must have same shape"

        batch_size = z1.size(0)
        device = z1.device

        # -------- 1. Normalize --------
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # -------- 2. Concatenate --------
        z = torch.cat([z1, z2], dim=0)  # [2B, D]

        # -------- 3. Similarity matrix --------
        sim = torch.matmul(z, z.T) / self.temperature  # [2B, 2B]

        # -------- 4. Mask self-similarity --------
        mask = torch.eye(2 * batch_size, device=device, dtype=torch.bool)
        sim.masked_fill_(mask, -1e4)

        # -------- 5. Positive pairs --------
        # positives: (i, i+B) and (i+B, i)
        pos_sim = torch.cat([
            torch.diag(sim, batch_size),
            torch.diag(sim, -batch_size)
        ], dim=0)  # [2B]

        # -------- 6. NT-Xent loss --------
        exp_sim = torch.exp(sim)
        denom = exp_sim.sum(dim=1)  # [2B]

        loss = -torch.log(torch.exp(pos_sim) / denom)
        return loss.mean()
