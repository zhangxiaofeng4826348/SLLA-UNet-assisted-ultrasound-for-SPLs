import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import nibabel as nib
from torchvision import transforms as T

from model import UNet_Swin_Pretrain

# ---------------- Grad-CAM ----------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None


        if self.target_layer in ['R1','R2', 'R3', 'R4', 'RS']:
            module = getattr(self.model, self.target_layer)
            module.register_forward_hook(self._forward_hook)
            module.register_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        self.model.zero_grad()
        _, cls_logits, pooled_feats, _, cam_feats = self.model(x)

        if class_idx is None:
            class_idx = cls_logits.argmax(dim=1)

        score = cls_logits.gather(1, class_idx.view(-1, 1)).squeeze()
        score.backward(torch.ones_like(score), retain_graph=True)

        if self.target_layer in ['R1','R2', 'R3', 'R4', 'RS']:
            act = self.activations
            grad = self.gradients
            weights = grad.mean(dim=(2,3), keepdim=True)
            cam = F.relu((weights * act).sum(dim=1, keepdim=True))
            cam = F.interpolate(cam, size=x.shape[2:], mode='bilinear', align_corners=False)
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
            return cam.detach().cpu().numpy()
        elif self.target_layer == 'Fused':

            pooled_feats.requires_grad_(True)
            grad = torch.autograd.grad(score.sum(), pooled_feats, retain_graph=True)[0]
            contrib = (grad * pooled_feats).detach().cpu().numpy()[0]
            return contrib
        else:
            raise ValueError(f"Unknown layer {self.target_layer}")


# ---------------- Transform ----------------
class ImgMaskTransform:
    def __init__(self, transforms=None, mask_transforms=None):
        self.transforms = transforms or []
        self.mask_transforms = mask_transforms or []

    def __call__(self, image, mask):
        for t in self.transforms:
            image = t(image)
        for t in self.mask_transforms:
            mask = t(mask)
        return image, mask

transform = ImgMaskTransform(
    transforms=[T.Resize((256,256))],
    mask_transforms=[T.Resize((256,256))]
)



import os

def visualize_cam(
    model_cls,
    img_path,
    checkpoint_path,
    device='cuda',
    save_dir='cam_results'
):
    import gc
    os.makedirs(save_dir, exist_ok=True)


    model = model_cls().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    new_ckpt = {}
    for k, v in ckpt.items():
        if k.startswith("BN."):
            new_k = k.replace("BN.", "RS.")
            new_ckpt[new_k] = v
        else:
            new_ckpt[k] = v
    model.load_state_dict(new_ckpt, strict=True)
    model.eval()


    img = Image.open(img_path).convert('RGB')
    img_resized = img.resize((256, 256))
    img_arr = np.array(img_resized)

    x_img, _ = transform(img, img)
    x = T.ToTensor()(x_img).unsqueeze(0).to(device)

    with torch.no_grad():
        seg_logits, cls_logits, pooled_feats, _, cam_feats = model(x)
        seg_prob = torch.sigmoid(seg_logits)[0, 0].cpu().numpy()
        seg_bin = (seg_prob > 0.5).astype(np.uint8)


    conv_layers = ['R1', 'R2', 'R3', 'R4', 'RS']
    cams = {}
    for layer in conv_layers:
        gradcam = GradCAM(model, layer)
        cam = gradcam(x)
        cams[layer] = cam[0, 0]
        del cam
        torch.cuda.empty_cache()


    fused_contrib = GradCAM(model, 'Fused')(x)
    channel_splits = {'R1':32, 'R2':64, 'R3':128, 'R4':256, 'RS':512, 'Swin':768}
    contrib_summary = {}
    start = 0
    for k, c in channel_splits.items():
        contrib_summary[k] = fused_contrib[start:start + c].sum()
        start += c
    del fused_contrib, pooled_feats
    torch.cuda.empty_cache()

    base_name = os.path.splitext(os.path.basename(img_path))[0]


    plt.figure(figsize=(4, 4))
    plt.imshow(img_arr)
    plt.axis('off')
    plt.savefig(os.path.join(save_dir, f"{base_name}_original.png"), dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()


    mask_red = np.zeros_like(img_arr)
    mask_red[..., 0] = seg_bin * 255
    overlay = (0.5 * img_arr + 0.5 * mask_red).astype(np.uint8)

    plt.figure(figsize=(4, 4))
    plt.imshow(overlay)
    plt.axis('off')
    plt.savefig(os.path.join(save_dir, f"{base_name}_mask_overlay.png"), dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    del x, x_img, seg_prob, seg_bin
    torch.cuda.empty_cache()
    gc.collect()


    for layer in conv_layers:
        plt.figure(figsize=(4, 4))
        plt.imshow(img_arr)
        plt.imshow(cams[layer], cmap='jet', alpha=0.5)
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, f"{base_name}_gradcam_{layer}.png"), dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()
        del cams[layer]
        torch.cuda.empty_cache()
    del cams
    gc.collect()


    plt.figure(figsize=(6, 4))
    plt.bar(contrib_summary.keys(), contrib_summary.values())
    plt.ylabel("Contribution")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{base_name}_fused_contribution.png"), dpi=300)
    plt.close()
    del contrib_summary
    torch.cuda.empty_cache()
    gc.collect()

    print(f"[INFO] All CAM results saved to: {save_dir}")




if __name__ == "__main__":

    test_dir = r"data_fine/extra_test/extra_test_images2"
    checkpoint_path = r"out/result_model/finetune_best_loss.pth"

    save_dir = r"cam_results_all"
    os.makedirs(save_dir, exist_ok=True)

    img_list = [
        f for f in os.listdir(test_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    print(f"[INFO] Found {len(img_list)} images.")

    for i, name in enumerate(img_list):

        img_path = os.path.join(test_dir, name)

        print(f"[{i+1}/{len(img_list)}] Processing: {name}")

        visualize_cam(
            UNet_Swin_Pretrain,
            img_path,
            checkpoint_path,
            save_dir=save_dir
        )

    print("[DONE] All test images processed.")

