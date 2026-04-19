import argparse
import json
import os
import time

import numpy as np
import torch
import yaml
from tqdm import tqdm

from data.dataset import get_dataloaders, get_mixed_dataloaders
from models.segmentation_model import PolypSegmentationModel
from models.losses import CombinedLoss
from utils.metrics import compute_metrics


def train_one_epoch_mixed(model, seg_loader, video_loader, criterion, optimizer, device):
    """Train one epoch with mixed image segmentation + video segmentation batches."""
    model.train()
    totals = {"loss": 0, "loss_1": 0, "loss_2": 0, "loss_3": 0, "loss_4": 0}
    num_batches = 0

    video_iter = iter(video_loader)

    for seg_batch in tqdm(seg_loader, desc="Train", leave=False):
        images, masks, prompts = seg_batch
        images = images.to(device)
        masks = masks.to(device)

        try:
            vid_batch = next(video_iter)
        except StopIteration:
            video_iter = iter(video_loader)
            vid_batch = next(video_iter)

        vid_imgs1, vid_imgs2, vid_masks1, vid_masks2, vid_prompts = vid_batch
        vid_imgs1 = vid_imgs1.to(device)
        vid_imgs2 = vid_imgs2.to(device)
        vid_masks1 = vid_masks1.to(device)

        # --- Segmentation path (image_seg_dataset): Loss 1 + Loss 4 ---
        pred_mask, guided_features, text_embedding, _ = model(
            images, prompts, images2=None
        )

        seg_loss, seg_loss_dict = criterion(
            pred_mask=pred_mask,
            target_mask=masks,
            visual_features=guided_features["f4"],
            text_embedding=text_embedding,
        )

        # --- Video path: Loss 1 + Loss 2 + Loss 3 + Loss 4 ---
        vid_pred_mask, vid_guided_features, vid_text_embedding, corr_outputs = model(
            vid_imgs1, vid_prompts, images2=vid_imgs2
        )

        vid_loss, vid_loss_dict = criterion(
            pred_mask=vid_pred_mask,
            target_mask=vid_masks1,
            visual_features=vid_guided_features["f4"],
            text_embedding=vid_text_embedding,
            f_corr=corr_outputs["f_corr"],
            features2=corr_outputs["features2"],
            f_enhanced=corr_outputs["f_enhanced"],
            features1=corr_outputs["features1"],
        )

        loss = seg_loss + vid_loss
        loss_dict = {}
        for k in ["loss_1", "loss_2", "loss_3", "loss_4"]:
            s = seg_loss_dict.get(k)
            v = vid_loss_dict.get(k)
            if s is not None and v is not None:
                loss_dict[k] = s + v
            elif s is not None:
                loss_dict[k] = s
            elif v is not None:
                loss_dict[k] = v

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        totals["loss"] += loss.item()
        for k in ["loss_1", "loss_2", "loss_3", "loss_4"]:
            if k in loss_dict:
                totals[k] += loss_dict[k].item()
        num_batches += 1

    return {k: v / max(num_batches, 1) for k, v in totals.items()}


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train one epoch with image segmentation data only (no video)."""
    model.train()
    totals = {"loss": 0, "loss_1": 0, "loss_4": 0}
    num_batches = 0

    for images, masks, prompts in tqdm(loader, desc="Train", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        pred_mask, guided_features, text_embedding, _ = model(images, prompts)

        loss, loss_dict = criterion(
            pred_mask=pred_mask,
            target_mask=masks,
            visual_features=guided_features["f4"],
            text_embedding=text_embedding,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        totals["loss"] += loss.item()
        for k in ["loss_1", "loss_4"]:
            if k in loss_dict:
                totals[k] += loss_dict[k].item()
        num_batches += 1

    return {k: v / max(num_batches, 1) for k, v in totals.items()}


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    totals = {"loss": 0, "loss_1": 0, "loss_4": 0}
    num_batches = 0

    all_preds = []
    all_targets = []

    for images, masks, prompts in tqdm(loader, desc="Val", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        pred_mask, guided_features, text_embedding, _ = model(images, prompts)

        loss, loss_dict = criterion(
            pred_mask=pred_mask,
            target_mask=masks,
            visual_features=guided_features["f4"],
            text_embedding=text_embedding,
        )

        totals["loss"] += loss.item()
        for k in ["loss_1", "loss_4"]:
            if k in loss_dict:
                totals[k] += loss_dict[k].item()
        num_batches += 1

        pred_sigmoid = torch.sigmoid(pred_mask).cpu().numpy()
        all_preds.append(pred_sigmoid)
        all_targets.append(masks.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    metrics = compute_metrics(all_preds, all_targets)

    for k, v in totals.items():
        metrics[k] = v / max(num_batches, 1)

    return metrics


@torch.no_grad()
def validate_video(model, loader, criterion, device):
    model.eval()
    totals = {"loss": 0, "loss_1": 0, "loss_2": 0, "loss_3": 0, "loss_4": 0}
    num_batches = 0

    all_preds = []
    all_targets = []

    for vid_imgs1, vid_imgs2, vid_masks1, vid_masks2, vid_prompts in tqdm(
        loader, desc="VidVal", leave=False
    ):
        vid_imgs1 = vid_imgs1.to(device)
        vid_imgs2 = vid_imgs2.to(device)
        vid_masks1 = vid_masks1.to(device)

        pred_mask, guided_features, text_embedding, corr_outputs = model(
            vid_imgs1, vid_prompts, images2=vid_imgs2
        )

        loss, loss_dict = criterion(
            pred_mask=pred_mask,
            target_mask=vid_masks1,
            visual_features=guided_features["f4"],
            text_embedding=text_embedding,
            f_corr=corr_outputs["f_corr"],
            features2=corr_outputs["features2"],
            f_enhanced=corr_outputs["f_enhanced"],
            features1=corr_outputs["features1"],
        )

        totals["loss"] += loss.item()
        for k in ["loss_1", "loss_2", "loss_3", "loss_4"]:
            if k in loss_dict:
                totals[k] += loss_dict[k].item()
        num_batches += 1

        pred_sigmoid = torch.sigmoid(pred_mask).cpu().numpy()
        all_preds.append(pred_sigmoid)
        all_targets.append(vid_masks1.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    metrics = compute_metrics(all_preds, all_targets)

    for k, v in totals.items():
        metrics[k] = v / max(num_batches, 1)

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
    parser.add_argument("--no-video", action="store_true",
                        help="Train without video data (image segmentation only)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    use_video = not args.no_video and "video" in cfg

    if use_video:
        (seg_train_loader, video_train_loader, val_loader, test_loader,
         video_val_loader, video_test_loader) = get_mixed_dataloaders(cfg)
    else:
        seg_train_loader, val_loader, test_loader = get_dataloaders(cfg)
        video_train_loader = None
        video_val_loader = None
        video_test_loader = None

    img_test_count = len(test_loader.dataset) if test_loader else 0
    print(f"\nImage seg  - Train: {len(seg_train_loader.dataset)}, "
          f"Val: {len(val_loader.dataset)}, Test: {img_test_count}")
    if use_video:
        vid_test_count = len(video_test_loader.dataset) if video_test_loader else 0
        print(f"Video seg  - Train: {len(video_train_loader.dataset)} pairs, "
              f"Val: {len(video_val_loader.dataset)} pairs, "
              f"Test: {vid_test_count} pairs")

    model = PolypSegmentationModel(cfg).to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params:,} / Total: {total_params:,}")

    loss_cfg = cfg["loss"]
    criterion = CombinedLoss(
        visual_dim=cfg["model"]["encoder_dim"],
        text_dim=cfg["model"]["text_dim"],
        lambda_1=loss_cfg.get("lambda_1", 1.0),
        lambda_2=loss_cfg.get("lambda_2", 1.0),
        lambda_3=loss_cfg.get("lambda_3", 0.5),
        lambda_4=loss_cfg.get("lambda_4", 1.0),
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
            criterion.load_state_dict(
                checkpoint["criterion_state_dict"], strict=False
            )
        start_epoch = checkpoint["epoch"] + 1
        best_dice = checkpoint.get("best_dice", 0.0)
        print(f"Resumed from epoch {checkpoint['epoch']} "
              f"(best Dice: {best_dice:.4f})")

    patience = cfg["training"].get("early_stopping_patience", 0)
    epochs_without_improvement = 0
    metrics_path = os.path.join(save_dir, "metrics.jsonl")

    for epoch in range(start_epoch, cfg["training"]["epochs"] + 1):
        start = time.time()

        if use_video:
            train_metrics = train_one_epoch_mixed(
                model, seg_train_loader, video_train_loader,
                criterion, optimizer, device
            )
        else:
            train_metrics = train_one_epoch(
                model, seg_train_loader, criterion, optimizer, device
            )

        val_metrics = validate(model, val_loader, criterion, device)

        vid_val_metrics = None
        if video_val_loader is not None:
            vid_val_metrics = validate_video(
                model, video_val_loader, criterion, device
            )

        scheduler.step()

        elapsed = time.time() - start
        lr = optimizer.param_groups[0]["lr"]

        loss_parts = []
        for k in ["loss_1", "loss_2", "loss_3", "loss_4"]:
            if k in train_metrics and train_metrics[k] > 0:
                loss_parts.append(f"L{k[-1]}: {train_metrics[k]:.4f}")
        loss_str = ", ".join(loss_parts)

        print(
            f"Epoch {epoch:03d}/{cfg['training']['epochs']} "
            f"({elapsed:.1f}s) | "
            f"LR: {lr:.2e} | "
            f"Train Loss: {train_metrics['loss']:.4f} ({loss_str})"
        )
        print(
            f"  Img Val  | "
            f"Loss: {val_metrics['loss']:.4f} | "
            f"Dice: {val_metrics['dice']:.4f} | "
            f"IoU: {val_metrics['iou']:.4f} | "
            f"F1: {val_metrics['f1']:.4f} | "
            f"HD: {val_metrics['hausdorff']:.2f}"
        )
        if vid_val_metrics is not None:
            print(
                f"  Vid Val  | "
                f"Loss: {vid_val_metrics['loss']:.4f} | "
                f"Dice: {vid_val_metrics['dice']:.4f} | "
                f"IoU: {vid_val_metrics['iou']:.4f} | "
                f"F1: {vid_val_metrics['f1']:.4f} | "
                f"HD: {vid_val_metrics['hausdorff']:.2f}"
            )

        epoch_record = {
            "epoch": epoch,
            "lr": lr,
            "elapsed_s": round(elapsed, 1),
            "train": {k: float(v) for k, v in train_metrics.items()},
            "img_val": {k: float(v) for k, v in val_metrics.items()},
        }
        if vid_val_metrics is not None:
            epoch_record["vid_val"] = {
                k: float(v) for k, v in vid_val_metrics.items()
            }
        with open(metrics_path, "a") as f:
            f.write(json.dumps(epoch_record) + "\n")

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

    has_test = (test_loader is not None and len(test_loader.dataset) > 0)
    has_vid_test = (video_test_loader is not None and len(video_test_loader.dataset) > 0)

    if has_test or has_vid_test:
        print("\nLoading best model for test evaluation...")
        best_ckpt = torch.load(os.path.join(save_dir, "best_model.pth"),
                               map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt["model_state_dict"])

    if has_test:
        test_metrics = validate(model, test_loader, criterion, device)

        print("=" * 50)
        print(f"IMAGE TEST RESULTS ({len(test_loader.dataset)} images)")
        print("=" * 50)
        print(f"  Dice:              {test_metrics['dice']:.4f}")
        print(f"  IoU:               {test_metrics['iou']:.4f}")
        print(f"  F1:                {test_metrics['f1']:.4f}")
        print(f"  Hausdorff Dist:    {test_metrics['hausdorff']:.2f}")
        print(f"  Loss:              {test_metrics['loss']:.4f}")
        print("=" * 50)

    if has_vid_test:
        vid_test_metrics = validate_video(
            model, video_test_loader, criterion, device
        )

        print("=" * 50)
        print(f"VIDEO TEST RESULTS ({len(video_test_loader.dataset)} pairs)")
        print("=" * 50)
        print(f"  Dice:              {vid_test_metrics['dice']:.4f}")
        print(f"  IoU:               {vid_test_metrics['iou']:.4f}")
        print(f"  F1:                {vid_test_metrics['f1']:.4f}")
        print(f"  Hausdorff Dist:    {vid_test_metrics['hausdorff']:.2f}")
        print(f"  Loss:              {vid_test_metrics['loss']:.4f}")
        print("=" * 50)


if __name__ == "__main__":
    main()
