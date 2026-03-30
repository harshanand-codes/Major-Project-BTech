import argparse
import os
import time

import numpy as np
import torch
import yaml
from tqdm import tqdm

from data.dataset import get_dataloaders
from models.segmentation_model import PolypSegmentationModel
from models.losses import CombinedLoss
from utils.metrics import compute_metrics


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_loss1 = 0.0
    total_loss4 = 0.0
    num_batches = 0

    for images, masks, prompts in tqdm(loader, desc="Train", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        pred_mask, guided_features, text_embedding = model(images, prompts)

        vl_feat = guided_features["f4"]
        loss, loss1, loss4 = criterion(pred_mask, masks, vl_feat, text_embedding)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_loss1 += loss1.item()
        total_loss4 += loss4.item()
        num_batches += 1

    return {
        "loss": total_loss / num_batches,
        "loss_dice_bce": total_loss1 / num_batches,
        "loss_vl": total_loss4 / num_batches,
    }


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_loss1 = 0.0
    total_loss4 = 0.0
    num_batches = 0

    all_preds = []
    all_targets = []

    for images, masks, prompts in tqdm(loader, desc="Val", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        pred_mask, guided_features, text_embedding = model(images, prompts)

        vl_feat = guided_features["f4"]
        loss, loss1, loss4 = criterion(pred_mask, masks, vl_feat, text_embedding)

        total_loss += loss.item()
        total_loss1 += loss1.item()
        total_loss4 += loss4.item()
        num_batches += 1

        pred_sigmoid = torch.sigmoid(pred_mask).cpu().numpy()
        all_preds.append(pred_sigmoid)
        all_targets.append(masks.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    metrics = compute_metrics(all_preds, all_targets)

    metrics["loss"] = total_loss / num_batches
    metrics["loss_dice_bce"] = total_loss1 / num_batches
    metrics["loss_vl"] = total_loss4 / num_batches

    return metrics


def save_checkpoint(path, epoch, model, optimizer, scheduler, criterion, best_dice, cfg):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "criterion_state_dict": criterion.state_dict(),
        "best_dice": best_dice,
        "config": cfg,
    }, path)


def main():
    parser = argparse.ArgumentParser(description="Train Polyp Segmentation Model")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = get_dataloaders(cfg)
    test_count = len(test_loader.dataset) if test_loader else 0
    print(f"Train samples: {len(train_loader.dataset)}, "
          f"Val samples: {len(val_loader.dataset)}, "
          f"Test samples: {test_count}")

    model = PolypSegmentationModel(cfg).to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params:,} / Total: {total_params:,}")

    criterion = CombinedLoss(
        visual_dim=cfg["model"]["encoder_dim"],
        text_dim=cfg["model"]["text_dim"],
        lambda_1=cfg["loss"]["lambda_1"],
        lambda_4=cfg["loss"]["lambda_4"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["training"]["epochs"]
    )

    save_dir = cfg["training"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    start_epoch = 1
    best_dice = 0.0

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "criterion_state_dict" in checkpoint:
            criterion.load_state_dict(checkpoint["criterion_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_dice = checkpoint.get("best_dice", 0.0)
        print(f"Resumed from epoch {checkpoint['epoch']} "
              f"(best Dice: {best_dice:.4f})")

    patience = cfg["training"].get("early_stopping_patience", 0)
    epochs_without_improvement = 0

    for epoch in range(start_epoch, cfg["training"]["epochs"] + 1):
        start = time.time()

        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer,
                                        device)
        val_metrics = validate(model, val_loader, criterion, device)

        scheduler.step()

        elapsed = time.time() - start
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d}/{cfg['training']['epochs']} "
            f"({elapsed:.1f}s) | "
            f"LR: {lr:.2e} | "
            f"Train Loss: {train_metrics['loss']:.4f} "
            f"(Dice+BCE: {train_metrics['loss_dice_bce']:.4f}, "
            f"VL: {train_metrics['loss_vl']:.4f}) | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Dice: {val_metrics['dice']:.4f} | "
            f"IoU: {val_metrics['iou']:.4f} | "
            f"F1: {val_metrics['f1']:.4f} | "
            f"HD: {val_metrics['hausdorff']:.2f}"
        )

        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            epochs_without_improvement = 0
            save_checkpoint(
                os.path.join(save_dir, "best_model.pth"),
                epoch, model, optimizer, scheduler, criterion, best_dice, cfg,
            )
            print(f"  -> New best model saved (Dice: {best_dice:.4f})")
        else:
            epochs_without_improvement += 1

        if epoch % 10 == 0:
            save_checkpoint(
                os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pth"),
                epoch, model, optimizer, scheduler, criterion, best_dice, cfg,
            )

        if patience > 0 and epochs_without_improvement >= patience:
            print(f"\nEarly stopping: Val Dice did not improve for {patience} epochs.")
            break

    print(f"\nTraining complete. Best validation Dice: {best_dice:.4f}")

    if test_loader is not None and len(test_loader.dataset) > 0:
        print("\nLoading best model for test evaluation...")
        best_ckpt = torch.load(os.path.join(save_dir, "best_model.pth"),
                               map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt["model_state_dict"])

        test_metrics = validate(model, test_loader, criterion, device)

        print("=" * 50)
        print(f"TEST RESULTS ({len(test_loader.dataset)} images)")
        print("=" * 50)
        print(f"  Dice:              {test_metrics['dice']:.4f}")
        print(f"  IoU:               {test_metrics['iou']:.4f}")
        print(f"  F1:                {test_metrics['f1']:.4f}")
        print(f"  Hausdorff Dist:    {test_metrics['hausdorff']:.2f}")
        print(f"  Loss:              {test_metrics['loss']:.4f}")
        print("=" * 50)


if __name__ == "__main__":
    main()
