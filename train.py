"""
train.py — Training loop with early stopping for both pipelines.

Usage (original gated-fusion pipeline):
    python train.py
"""

import copy
import time

import torch
import torch.nn as nn
from tqdm import tqdm

from config import cfg
from dataset import load_dataframe, get_splits, build_loaders
from models import FusionModel, build_resnet50_encoder, build_audio_encoder


# ── Generic training engine ────────────────────────────────────────────────────

def train_model(
    name: str,
    model: nn.Module,
    train_loader,
    val_loader,
    forward_fn=None,
    epochs: int = cfg.EPOCHS,
    patience: int = cfg.PATIENCE,
    lr: float = cfg.LR,
    trainable_params=None,
):
    """
    Generic training loop with early stopping and LR scheduling.

    Args:
        forward_fn : callable(model, images, specs) → logits.
                     Defaults to the standard multimodal forward pass.
        trainable_params : list of params to optimise (default: all with grad).

    Returns:
        model, train_losses, val_losses, train_accs, val_accs
    """
    print(f"\n{'='*60}")
    print(f"  Training: {name}")
    print(f"{'='*60}")

    criterion = nn.CrossEntropyLoss()
    params = trainable_params or [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=2, factor=0.5, verbose=False)

    best_val_loss = float("inf")
    patience_ctr  = 0
    best_state    = None
    train_losses, val_losses   = [], []
    train_accs,   val_accs     = [], []
    t_start = time.time()

    for epoch in range(epochs):
        # ── Train ──────────────────────────────────────────────────────────────
        model.train()
        run_loss, correct, total = 0.0, 0, 0

        for images, specs, labels in tqdm(train_loader,
                                          desc=f"Epoch {epoch+1}/{epochs} train",
                                          leave=False):
            images = images.to(cfg.DEVICE)
            specs  = specs.to(cfg.DEVICE)
            labels = labels.to(cfg.DEVICE)

            optimizer.zero_grad()
            logits = forward_fn(model, images, specs) if forward_fn else model(images, specs)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            run_loss += loss.item()
            correct  += (logits.argmax(1) == labels).sum().item()
            total    += labels.size(0)

        ep_loss = run_loss / len(train_loader)
        ep_acc  = correct / total
        train_losses.append(ep_loss)
        train_accs.append(ep_acc)

        # ── Validate ────────────────────────────────────────────────────────────
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0

        with torch.no_grad():
            for images, specs, labels in val_loader:
                images = images.to(cfg.DEVICE)
                specs  = specs.to(cfg.DEVICE)
                labels = labels.to(cfg.DEVICE)
                logits = forward_fn(model, images, specs) if forward_fn else model(images, specs)
                v_loss    += criterion(logits, labels).item()
                v_correct += (logits.argmax(1) == labels).sum().item()
                v_total   += labels.size(0)

        v_ep_loss = v_loss / len(val_loader)
        v_ep_acc  = v_correct / v_total
        val_losses.append(v_ep_loss)
        val_accs.append(v_ep_acc)
        scheduler.step(v_ep_loss)

        print(
            f"  Epoch {epoch+1:02d}/{epochs}  "
            f"train_loss={ep_loss:.4f}  train_acc={ep_acc:.4f}  "
            f"val_loss={v_ep_loss:.4f}  val_acc={v_ep_acc:.4f}"
        )

        # ── Early stopping ──────────────────────────────────────────────────────
        if v_ep_loss < best_val_loss:
            best_val_loss = v_ep_loss
            patience_ctr  = 0
            best_state    = copy.deepcopy(model.state_dict())
            torch.save(best_state, cfg.BEST_CKPT_PATH)
            print("  Best model updated.")
        else:
            patience_ctr += 1
            print(f"  No improvement for {patience_ctr} epoch(s).")
            if patience_ctr >= patience:
                print("  Early stopping triggered.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    elapsed = time.time() - t_start
    print(f"  Done in {elapsed:.0f}s  (best val loss={best_val_loss:.4f})")
    return model, train_losses, val_losses, train_accs, val_accs


# ── Gated-fusion training pipeline ────────────────────────────────────────────

def run_gated_fusion_training():
    """
    Full training run for the gated ResNet50+AudioCNN fusion model.
    Returns trained models and loss history.
    """
    df = load_dataframe()
    train_df, val_df = get_splits(df)
    train_loader, val_loader, _, _ = build_loaders(train_df, val_df)

    resnet      = build_resnet50_encoder().to(cfg.DEVICE)
    audio_model = build_audio_encoder().to(cfg.DEVICE)
    fusion      = FusionModel().to(cfg.DEVICE)

    criterion  = nn.CrossEntropyLoss()
    optimizer  = torch.optim.Adam(
        list(resnet.layer4.parameters()) +
        list(audio_model.parameters()) +
        list(fusion.parameters()),
        lr=cfg.LR,
    )

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    patience_ctr  = 0

    for epoch in range(cfg.EPOCHS):
        # Train
        fusion.train()
        running_loss = 0.0

        for images, specs, labels in tqdm(train_loader):
            images = images.to(cfg.DEVICE)
            specs  = specs.to(cfg.DEVICE)
            labels = labels.to(cfg.DEVICE)
            optimizer.zero_grad()
            img_feat = resnet(images)
            aud_feat = audio_model(specs)
            outputs  = fusion(img_feat, aud_feat)
            loss     = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        print(f"\nEpoch {epoch+1}/{cfg.EPOCHS}  Train Loss: {epoch_loss:.4f}")

        # Validate
        fusion.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, specs, labels in val_loader:
                images = images.to(cfg.DEVICE)
                specs  = specs.to(cfg.DEVICE)
                labels = labels.to(cfg.DEVICE)
                outputs  = fusion(resnet(images), audio_model(specs))
                val_loss += criterion(outputs, labels).item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        print(f"Validation Loss: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_ctr  = 0
            torch.save(fusion.state_dict(), cfg.BEST_CKPT_PATH)
            print("Best model updated.")
        else:
            patience_ctr += 1
            print(f"No improvement for {patience_ctr} epoch(s).")
            if patience_ctr >= cfg.PATIENCE:
                print("Early stopping triggered.")
                break

    # Save final fusion model
    torch.save(fusion.state_dict(), cfg.FUSION_WEIGHTS_PATH)
    torch.save(fusion, cfg.FUSION_FULLMODEL_PATH)
    print(f"Fusion model saved to {cfg.FUSION_WEIGHTS_PATH}")

    return resnet, audio_model, fusion, train_losses, val_losses


if __name__ == "__main__":
    resnet, audio_model, fusion, train_losses, val_losses = run_gated_fusion_training()
    print(f"Final train loss: {train_losses[-1]:.4f}")
    print(f"Final val   loss: {val_losses[-1]:.4f}")
