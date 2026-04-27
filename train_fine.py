# ===============================================
# finetune.py




import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms as T
from sklearn.metrics import accuracy_score, roc_auc_score

from model_non_cf import UNet_Swin_Pretrain_non_cf
from Loss_fine import CombinedLoss
from dataset_fine import CustomDataset
from optimizer_fine import build_finetune_optimizer, build_finetune_scheduler


def run_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer=None,
    device="cuda",
    train=True
):
    model.train() if train else model.eval()

    total_loss = 0.0
    n_samples = 0

    all_labels = []
    all_preds = []
    all_seg_logits = []
    all_seg_masks = []

    with torch.set_grad_enabled(train):
        for batch in dataloader:
            image = batch["image"].to(device)     # [B,3,H,W]
            mask  = batch["mask"].to(device)      # [B,1,H,W]
            label = batch["label"].to(device)     # [B]
            clin  = batch.get("clin", None)

            if train:
                optimizer.zero_grad()

            # -------- forward --------
            seg_logits, cls_logits, _, _ = model(image)
            loss = criterion(cls_logits, label, seg_logits, mask)

            if train:
                loss.backward()
                optimizer.step()

            bs = image.size(0)
            total_loss += loss.item() * bs
            n_samples += bs


            all_labels.append(label.detach().cpu())
            all_preds.append(
                torch.softmax(cls_logits, dim=1)[:, 1].detach().cpu()
            )
            all_seg_logits.append(
                torch.sigmoid(seg_logits).detach().cpu()
            )
            all_seg_masks.append(mask.detach().cpu())


    all_labels = torch.cat(all_labels)
    all_preds = torch.cat(all_preds)
    all_seg_logits = torch.cat(all_seg_logits)
    all_seg_masks = torch.cat(all_seg_masks)


    acc = accuracy_score(
        all_labels.numpy(),
        (all_preds.numpy() > 0.5).astype(int)
    )
    try:
        auc = roc_auc_score(all_labels.numpy(), all_preds.numpy())
    except ValueError:
        auc = float("nan")


    smooth = 1e-6
    intersection = (all_seg_logits * all_seg_masks).sum(dim=(1, 2, 3))
    union = all_seg_logits.sum(dim=(1, 2, 3)) + all_seg_masks.sum(dim=(1, 2, 3))
    dice = ((2 * intersection + smooth) / (union + smooth)).mean().item()

    return total_loss / n_samples, dice, acc, auc



# ===============================================
def finetune(
    train_loader,
    val_loader,
    model,
    criterion,
    lr_encoder,
    lr_decoder,
    lr_classifier,
    weight_decay,
    scheduler_min_lr,
    device,
    num_epochs,
    save_dir,
    ssl_pretrain_path=None,
    freeze_encoder=False,
    freeze_swin=False
):
    os.makedirs(save_dir, exist_ok=True)


    if ssl_pretrain_path is not None:
        ckpt = torch.load(ssl_pretrain_path, map_location=device)

        if isinstance(ckpt, dict):
            if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
                pretrained_dict = ckpt
            elif "encoder" in ckpt:
                pretrained_dict = ckpt["encoder"]
            elif "model" in ckpt:
                pretrained_dict = ckpt["model"]
            else:
                raise KeyError(f"Unrecognized checkpoint keys: {ckpt.keys()}")
        else:
            raise TypeError("Unsupported checkpoint format")

        model_dict = model.state_dict()


        pretrained_dict_filtered = {}
        for k, v in pretrained_dict.items():
            if k in model_dict:
                if v.shape == model_dict[k].shape and "cls_head" not in k:
                    pretrained_dict_filtered[k] = v
                else:
                    print(f"[INFO] Skip loading {k}: shape mismatch or excluded")


        model_dict.update(pretrained_dict_filtered)
        model.load_state_dict(model_dict)

        print(f"[INFO] Loaded SSL pretrained weights (cls_head excluded) from:\n{ssl_pretrain_path}")

    # ===== Optimizer / Scheduler =====
    optimizer = build_finetune_optimizer(
        model,
        lr_encoder=lr_encoder,
        lr_decoder=lr_decoder,
        lr_classifier=lr_classifier,
        weight_decay=weight_decay,
        freeze_encoder=freeze_encoder,
        freeze_swin=freeze_swin
    )

    scheduler = build_finetune_scheduler(
        optimizer,
        num_epochs=num_epochs,
        min_lr=scheduler_min_lr
    )


    best_val_loss = np.inf
    best_val_auc = 0.0
    best_val_dice = 0.0

    # ===== Training Loop =====
    for epoch in range(num_epochs):
        train_loss, train_dice, train_acc, train_auc = run_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer=optimizer,
            device=device,
            train=True
        )

        val_loss, val_dice, val_acc, val_auc = run_one_epoch(
            model,
            val_loader,
            criterion,
            optimizer=None,
            device=device,
            train=False
        )

        print(
            f"Epoch [{epoch+1:03d}/{num_epochs}] | "
            f"Train Loss: {train_loss:.4f} | Dice: {train_dice:.4f} | "
            f"Acc: {train_acc:.4f} | AUC: {train_auc:.4f} || "
            f"Val Loss: {val_loss:.4f} | Dice: {val_dice:.4f} | "
            f"Acc: {val_acc:.4f} | AUC: {val_auc:.4f}"
        )

        scheduler.step()


        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                model.state_dict(),
                os.path.join(save_dir, "finetune_best_loss_non_cf.pth")
            )
            print("  ↳ Best multi-task (Loss) model saved.")


        if not np.isnan(val_auc) and val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(
                model.state_dict(),
                os.path.join(save_dir, "finetune_best_auc_non_cf.pth")
            )
            print("  ↳ Best classification (AUC) model saved.")


        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(
                model.state_dict(),
                os.path.join(save_dir, "finetune_best_dice_non_cf.pth")
            )
            print("  ↳ Best segmentation (Dice) model saved.")

    print("Training finished.")


# ===============================================
# ---------- main ----------
# ===============================================
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    train_image_dir = r"data_fine/train/train_images"
    train_mask_dir  = r"data_fine/train/train_masks"
    train_label_xls = r"data_fine/train/train_labels.xlsx"

    val_image_dir   = r"data_fine/val/val_images"
    val_mask_dir    = r"data_fine/val/val_masks"
    val_label_xls   = r"data_fine/val/val_labels.xlsx"

    ssl_pretrain_path = r"out/result/ssl_pretrain_final.pth"
    save_dir = r"out/result"


    batch_size = 8
    num_epochs = 100

    lr_encoder = 5e-6
    lr_decoder = 1e-5
    lr_classifier = 1e-5
    weight_decay = 1e-5
    min_lr = 1e-6

    freeze_encoder = False
    freeze_swin = False

    num_benign = 210
    num_malignant = 484

    # -------- Transform --------
    transform = T.ToTensor()

    # -------- Dataset / Loader --------
    train_dataset = CustomDataset(
        train_image_dir, train_mask_dir, train_label_xls, transform=transform
    )
    val_dataset = CustomDataset(
        val_image_dir, val_mask_dir, val_label_xls, transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # -------- Model --------
    model = UNet_Swin_Pretrain_non_cf().to(device)

    # -------- Loss --------
    criterion = CombinedLoss(
        num_benign=num_benign,
        num_malignant=num_malignant,
        alpha=1.0,
        beta=1.0
    )


    finetune(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        criterion=criterion,
        lr_encoder=lr_encoder,
        lr_decoder=lr_decoder,
        lr_classifier=lr_classifier,
        weight_decay=weight_decay,
        scheduler_min_lr=min_lr,
        device=device,
        num_epochs=num_epochs,
        save_dir=save_dir,
        ssl_pretrain_path=ssl_pretrain_path,
        freeze_encoder=freeze_encoder,
        freeze_swin=freeze_swin
    )
