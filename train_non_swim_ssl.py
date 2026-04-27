import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms as T

from model_non_swim import UNet_Ablation, ProjectionHead
from Loss_ssl import NTXentLoss
from dataset_ssl import SSLDataset
from optimizer_ssl import build_optimizer, build_scheduler
from utils import set_seed, save_checkpoint, AverageMeter
from config_ssl import SSL_CONFIG


# View1
transform1 = T.Compose([
    T.RandomResizedCrop(256, scale=(0.7, 1.0)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(15),
    T.ToTensor()
])

# View2
transform2 = T.Compose([
    T.RandomResizedCrop(256, scale=(0.8, 1.0)),
    T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
    T.RandomHorizontalFlip(),
    T.ToTensor()
])


def train_ssl(image_folder, freeze_encoder=False):


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(SSL_CONFIG["seed"])


    os.makedirs("out/result", exist_ok=True)
    os.makedirs("out/picture", exist_ok=True)

    # ===== Dataset / Dataloader =====
    dataset = SSLDataset(
        image_folder=image_folder,
        mask_folder=None,
        transform1=transform1,
        transform2=transform2
    )
    dataloader = DataLoader(
        dataset,
        batch_size=SSL_CONFIG["batch_size"],
        shuffle=True,
        drop_last=True
    )

    # ===== Model / Projection Head =====
    model = UNet_Ablation().to(device)
    proj_head = ProjectionHead(
        in_dim=32+64+128+256+512,
        proj_dim=SSL_CONFIG["proj_dim"]
    ).to(device)

    # ===== Optimizer / Scheduler / Loss =====
    optimizer = build_optimizer(
        model, proj_head,
        lr=SSL_CONFIG["lr"],
        weight_decay=SSL_CONFIG["weight_decay"],
        freeze_encoder=freeze_encoder,
        freeze_swin=False
    )
    scheduler = build_scheduler(optimizer, SSL_CONFIG["epochs"])
    criterion = NTXentLoss(temperature=SSL_CONFIG["temperature"])

    scaler = torch.amp.GradScaler()
    loss_meter = AverageMeter()
    loss_curve = []
    best_loss = float('inf')

    # ================= Training Loop =================
    for epoch in range(SSL_CONFIG["epochs"]):
        model.train()
        proj_head.train()
        loss_meter.reset()

        for batch in dataloader:
            img1 = batch["image1"].to(device)
            img2 = batch["image2"].to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type='cuda'):
                feat1 = model(img1)[2]  # pooled_feats
                feat2 = model(img2)[2]

                z1 = proj_head(torch.cat(feat1, dim=1))
                z2 = proj_head(torch.cat(feat2, dim=1))

                loss = criterion(z1, z2)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(loss.item(), img1.size(0))

        scheduler.step()
        loss_curve.append(loss_meter.avg)

        print(f"Epoch [{epoch+1:03d}/{SSL_CONFIG['epochs']}] SSL Loss: {loss_meter.avg:.4f}")


        if loss_meter.avg < best_loss:
            best_loss = loss_meter.avg
            save_checkpoint(
                {
                    "model": model.state_dict(),
                    "proj_head": proj_head.state_dict(),
                    "epoch": epoch,
                    "loss": best_loss
                },
                "out/result/ssl_best_non_swim.pth"
            )


        if (epoch + 1) % SSL_CONFIG["save_freq"] == 0:
            save_checkpoint(
                {
                    "model": model.state_dict(),
                    "proj_head": proj_head.state_dict(),
                    "epoch": epoch,
                    "loss": loss_meter.avg
                },
                f"out/result/ssl_epoch_non_swim_{epoch+1}.pth"
            )


    save_checkpoint(
        {"model": model.state_dict(),
         "proj_head": proj_head.state_dict()},
        "out/result/ssl_pretrain_final_non_swim.pth"
    )


    plt.figure()
    plt.plot(loss_curve)
    plt.xlabel("Epoch")
    plt.ylabel("SSL Loss")
    plt.title("SSL Pretraining Curve")
    plt.savefig("out/picture/ssl_loss_non_swim.png")
    plt.show()
    plt.close()



if __name__ == "__main__":
    train_ssl(
        r"C:\Users\Administrator\PycharmProjects\unet+\pre\data_pre\onlineDataset\images",
        freeze_encoder=False
    )
