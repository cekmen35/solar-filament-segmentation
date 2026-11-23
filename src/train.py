"""Train the segmentation model.  Usage: python -m src.train"""
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader

from .config import Config
from .data import SegDataset, load_split
from .model import build_model


def val_iou(model, loader, device):
    model.eval()
    inter = union = 0
    with torch.no_grad(), torch.autocast("cuda"):
        for x, y in loader:
            p = (torch.sigmoid(model(x.to(device))) > 0.5).float().cpu()
            t = (y > 0.5).float()
            inter += (p * t).sum().item()
            union += ((p + t) > 0).float().sum().item()
    return inter / max(union, 1)


def main(cfg: Config = Config()):
    device = "cuda"
    train_data, val_data, *_ = load_split(cfg)
    train_loader = DataLoader(SegDataset(train_data, cfg, True), batch_size=cfg.batch_size,
                              shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(SegDataset(val_data, cfg, False), batch_size=4, num_workers=2)

    model = build_model(cfg).to(device)
    dice, bce = smp.losses.DiceLoss("binary"), torch.nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.lr, total_steps=cfg.epochs * len(train_loader))
    scaler = torch.cuda.amp.GradScaler()

    best = 0.0
    for ep in range(cfg.epochs):
        model.train()
        total = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            with torch.autocast("cuda"):
                out = model(x)
                loss = bce(out, y) + dice(out, y)   # BCE + Dice: masks cover <1% of pixels
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item()
        iou = val_iou(model, val_loader, device)
        print(f"ep {ep + 1:02d} loss {total / len(train_loader):.4f} val IoU {iou:.4f}")
        if iou > best:
            best = iou
            torch.save(model.state_dict(), f"{cfg.work_dir}/best.pt")


if __name__ == "__main__":
    main()
