"""
Train the Character-level CNN for language detection on CUDA.

Usage:
    python train_cnn.py --epochs 30 --batch-size 128 --lr 0.001 --patience 5

Outputs:
    - char_cnn_model.pt       (model checkpoint)
    - runs/                   (TensorBoard logs)
"""
import argparse
import time
import random
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report

from app.model import CharCNN, text_to_indices, VOCAB_SIZE, NUM_CLASSES, MAX_LEN, LABELS

# ══════════════════════════════════════════════════════════════
# Reproducibility
# ══════════════════════════════════════════════════════════════
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ══════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════
class LanguageDataset(Dataset):
    """Character-level dataset for language detection."""

    def __init__(self, texts, labels, max_len=MAX_LEN):
        self.texts = texts
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        indices = text_to_indices(self.texts[idx], self.max_len)
        return (
            torch.tensor(indices, dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


# ══════════════════════════════════════════════════════════════
# Early Stopping
# ══════════════════════════════════════════════════════════════
class EarlyStopping:
    """Stop training when validation metric stops improving."""

    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_score = score
            self.counter = 0


# ══════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════
def train_epoch(model, loader, criterion, optimizer, scaler, device):
    """Train for one epoch with mixed precision."""
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
            logits = model(batch_x)
            loss = criterion(logits, batch_y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * batch_x.size(0)
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    return avg_loss, f1


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
            logits = model(batch_x)
            loss = criterion(logits, batch_y)

        total_loss += loss.item() * batch_x.size(0)
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    return avg_loss, f1, all_preds, all_labels


def main(args):
    # ── Device ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{'='*60}")
    print(f"TRAINING CharCNN — Language Detection")
    print(f"{'='*60}")
    print(f"Device:       {device}")
    if device.type == "cuda":
        print(f"GPU:          {torch.cuda.get_device_name(0)}")
        print(f"VRAM:         {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"CUDA version: {torch.version.cuda}")
    print(f"PyTorch:      {torch.__version__}")
    print(f"{'='*60}")

    # ── Load data ──
    data = pd.read_csv("Language Detection.csv")
    data = data.drop_duplicates(subset=["Text"], keep="first").reset_index(drop=True)
    print(f"Dataset: {len(data)} samples, {data['Language'].nunique()} classes")

    le = LabelEncoder()
    labels = le.fit_transform(data["Language"])
    texts = data["Text"].tolist()

    # ── Stratified split ──
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.20, random_state=SEED, stratify=labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.125, random_state=SEED, stratify=y_train
    )
    # 0.125 of 0.80 = 0.10 → final split: 70/10/20

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # ── DataLoaders ──
    train_ds = LanguageDataset(X_train, y_train)
    val_ds = LanguageDataset(X_val, y_val)
    test_ds = LanguageDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)

    # ── Model ──
    model = CharCNN(
        vocab_size=VOCAB_SIZE,
        embed_dim=args.embed_dim,
        num_classes=NUM_CLASSES,
        dropout=args.dropout,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    # ── Optimizer & Scheduler ──
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
    early_stopping = EarlyStopping(patience=args.patience)

    # ── TensorBoard ──
    writer = SummaryWriter("runs/char_cnn")
    print(f"\nTensorBoard logs: runs/char_cnn")
    print(f"  → tensorboard --logdir runs/\n")

    # ── Training loop ──
    best_val_f1 = 0
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_f1 = train_epoch(model, train_loader, criterion,
                                           optimizer, scaler, device)
        val_loss, val_f1, _, _ = evaluate(model, val_loader, criterion, device)

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        # Log to TensorBoard
        writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("F1", {"train": train_f1, "val": val_f1}, epoch)
        writer.add_scalar("LR", lr, epoch)

        if device.type == "cuda":
            gpu_mem = torch.cuda.max_memory_allocated() / 1024**2
            writer.add_scalar("GPU_Memory_MB", gpu_mem, epoch)

        print(f"Epoch {epoch:02d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} F1: {train_f1:.4f} | "
              f"Val Loss: {val_loss:.4f} F1: {val_f1:.4f} | "
              f"LR: {lr:.6f} | {elapsed:.1f}s"
              + (f" | GPU: {gpu_mem:.0f}MB" if device.type == "cuda" else ""))

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                "model_state_dict": model.state_dict(),
                "vocab_size": VOCAB_SIZE,
                "embed_dim": args.embed_dim,
                "num_classes": NUM_CLASSES,
                "dropout": args.dropout,
                "epoch": epoch,
                "val_f1": val_f1,
                "val_loss": val_loss,
            }, "char_cnn_model.pt")
            print(f"  ★ Best model saved (val F1={val_f1:.4f})")

        scheduler.step(val_f1)
        early_stopping(val_f1)
        if early_stopping.should_stop:
            print(f"\n✓ Early stopping at epoch {epoch} (patience={args.patience})")
            break

    total_time = time.time() - t_start
    writer.close()

    # ── Final evaluation on test set ──
    print(f"\n{'='*60}")
    print(f"FINAL TEST EVALUATION")
    print(f"{'='*60}")

    # Reload best checkpoint
    checkpoint = torch.load("char_cnn_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_f1, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device
    )

    print(f"Test Loss:     {test_loss:.4f}")
    print(f"Test F1:       {test_f1:.4f}")
    print(f"Best Val F1:   {best_val_f1:.4f}")
    print(f"Total time:    {total_time:.1f}s")
    print(f"Checkpoint:    char_cnn_model.pt")

    print(f"\n--- Per-Class Report ---")
    print(classification_report(test_labels, test_preds, target_names=LABELS))

    if device.type == "cuda":
        print(f"\nGPU peak memory: {torch.cuda.max_memory_allocated() / 1024**2:.0f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CharCNN for Language Detection")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()
    main(args)
