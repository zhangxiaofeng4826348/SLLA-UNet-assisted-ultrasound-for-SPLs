# ===============================================
# optimizer_fine.py
# Optimizer & Scheduler for SSL fine-tuning / downstream fine-tune
# Compatible with UNet_Swin_Pretrain & UNet_Ablation
# ===============================================

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

# =================================================
# Optimizer
# =================================================

def build_finetune_optimizer(
    model,
    lr_encoder=1e-4,
    lr_decoder=5e-4,
    lr_classifier=5e-4,
    weight_decay=1e-5,
    freeze_encoder=False,
    freeze_swin=False
):
    """
    Build AdamW optimizer with layer-wise learning rates
    (encoder < decoder/classifier)

    Args:
        model: UNet_Swin_Pretrain or UNet_Ablation
        lr_encoder: learning rate for encoder
        lr_decoder: learning rate for segmentation decoder
        lr_classifier: learning rate for classification head
        weight_decay: AdamW weight decay
        freeze_encoder: whether to freeze encoder (R1-R4 + BN)
        freeze_swin: whether to freeze Swin Transformer (only if exists)

    Returns:
        optimizer
    """

    param_groups = []

    # ----------------- Encoder -----------------
    encoder_params = list(model.R1.parameters()) + \
                     list(model.R2.parameters()) + \
                     list(model.R3.parameters()) + \
                     list(model.R4.parameters()) + \
                     list(model.BN.parameters())


    if hasattr(model, "swin"):
        encoder_params += list(model.swin.parameters())


    if freeze_encoder:
        for p in list(model.R1.parameters()) + list(model.R2.parameters()) + \
                 list(model.R3.parameters()) + list(model.R4.parameters()) + list(model.BN.parameters()):
            p.requires_grad = False

    if hasattr(model, "swin") and freeze_swin:
        for p in model.swin.parameters():
            p.requires_grad = False


    encoder_params = [p for p in encoder_params if p.requires_grad]
    if encoder_params:
        param_groups.append({"params": encoder_params, "lr": lr_encoder})

    # ----------------- Decoder -----------------
    decoder_params = list(model.U1.parameters()) + \
                     list(model.U2.parameters()) + \
                     list(model.U3.parameters()) + \
                     list(model.U4.parameters()) + \
                     list(model.CD1.parameters()) + \
                     list(model.CD2.parameters()) + \
                     list(model.CD3.parameters()) + \
                     list(model.CD4.parameters()) + \
                     list(model.seg_head.parameters())

    decoder_params = [p for p in decoder_params if p.requires_grad]
    if decoder_params:
        param_groups.append({"params": decoder_params, "lr": lr_decoder})

    # ----------------- Classifier -----------------
    classifier_params = [p for p in model.cls_head.parameters() if p.requires_grad]
    if classifier_params:
        param_groups.append({"params": classifier_params, "lr": lr_classifier})

    # ----------------- Build Optimizer -----------------
    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    return optimizer


# =================================================
# Scheduler
# =================================================

def build_finetune_scheduler(
    optimizer,
    num_epochs,
    min_lr=1e-6
):
    """
    Cosine annealing scheduler (epoch-level)
    Args:
        optimizer: optimizer
        num_epochs: total training epochs
        min_lr: minimum learning rate
    Returns:
        scheduler
    """
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=min_lr
    )
    return scheduler
