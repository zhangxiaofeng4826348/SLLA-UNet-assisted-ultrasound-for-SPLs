import torch

def build_optimizer(model, proj_head, lr=5e-5, weight_decay=1e-4,
                    freeze_encoder=False, freeze_swin=False):


    for name, param in model.named_parameters():

        if any(k in name for k in ["CD", "U", "seg_head", "cls_head"]):
            param.requires_grad = False

        elif freeze_encoder and any(k in name for k in ["R1", "R2", "R3", "R4", "BN"]):
            param.requires_grad = False

        elif freeze_swin and "swin" in name:
            param.requires_grad = False
        else:
            param.requires_grad = True


    params_to_optimize = list(filter(lambda p: p.requires_grad, model.parameters())) + list(proj_head.parameters())

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=lr,
        weight_decay=weight_decay
    )
    return optimizer


def build_scheduler(optimizer, max_epoch):

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max_epoch
    )
    return scheduler
