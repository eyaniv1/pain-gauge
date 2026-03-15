"""
Pain Detection Model Fine-tuning Script

Fine-tunes a pre-trained model on labeled pain data (UNBC-McMaster format
or clinician corrections). Supports configurable hyperparameters, saves
best checkpoint, and exports ONNX version.

Usage:
  python train.py --data-dir training/data/unbc --base-model models/active/model.onnx
  python train.py --data-dir training/data/unbc --epochs 20 --lr 0.0001
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# PyTorch is optional — only needed for actual training
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class PainDataset:
    """Dataset that loads preprocessed images + pain labels from a split directory."""

    def __init__(self, split_dir: str, target_size: tuple[int, int] = (224, 224)):
        self.split_dir = Path(split_dir)
        self.target_size = target_size
        self.samples = []
        self._load_labels()

    def _load_labels(self):
        csv_path = self.split_dir / "labels.csv"
        if not csv_path.exists():
            return

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row["pain_score"]:
                    continue
                self.samples.append({
                    "filename": row["filename"],
                    "pain_score": float(row["pain_score"]),
                    "aus": row.get("aus", ""),
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = self.split_dir / "images" / sample["filename"]

        img = cv2.imread(str(img_path))
        if img is None:
            # Return zeros if image is missing
            img = np.zeros((*self.target_size, 3), dtype=np.uint8)

        img = cv2.resize(img, self.target_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Normalize
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std

        # HWC → CHW
        img = np.transpose(img, (2, 0, 1))

        pain_score = sample["pain_score"] / 10.0  # Normalize to 0-1 for training

        if TORCH_AVAILABLE:
            return torch.tensor(img, dtype=torch.float32), torch.tensor(pain_score, dtype=torch.float32)
        return img, pain_score


class SimplePainModel(nn.Module if TORCH_AVAILABLE else object):
    """Simple CNN for pain regression. Used as default when no base model provided."""

    def __init__(self, n_classes: int = 1):
        if not TORCH_AVAILABLE:
            return
        super().__init__()
        # Simple feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),
            nn.Sigmoid(),  # Output in 0-1 range
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def load_base_model(model_path: str | None) -> nn.Module:
    """Load a base model for fine-tuning, or create a new one."""
    if model_path and Path(model_path).exists():
        ext = Path(model_path).suffix
        if ext in (".pt", ".pth"):
            model = torch.load(model_path, map_location="cpu", weights_only=False)
            if hasattr(model, "eval"):
                model.eval()
            print(f"Loaded base model from: {model_path}")
            return model
        else:
            print(f"Cannot fine-tune from {ext} format directly. Creating new model.")

    print("Creating new SimplePainModel")
    return SimplePainModel(n_classes=1)


def freeze_early_layers(model: nn.Module, freeze_ratio: float = 0.7):
    """Freeze early layers of the model for fine-tuning."""
    params = list(model.parameters())
    n_freeze = int(len(params) * freeze_ratio)

    for i, param in enumerate(params):
        param.requires_grad = i >= n_freeze

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,} total, {trainable:,} trainable ({100*trainable/total:.0f}%)")


def train_epoch(model, loader, criterion, optimizer, device) -> dict:
    """Train for one epoch. Returns metrics dict."""
    model.train()
    total_loss = 0
    n_batches = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return {"loss": total_loss / max(n_batches, 1)}


def evaluate(model, loader, criterion, device) -> dict:
    """Evaluate model on a dataset. Returns metrics dict."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    n_batches = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            all_preds.extend(outputs.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
            n_batches += 1

    preds = np.array(all_preds) * 10  # Back to 0-10 scale
    labels = np.array(all_labels) * 10

    mae = float(np.mean(np.abs(preds - labels)))
    rmse = float(np.sqrt(np.mean((preds - labels) ** 2)))

    return {
        "loss": total_loss / max(n_batches, 1),
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
    }


def export_onnx(model, output_path: str, input_size: tuple = (1, 3, 224, 224)):
    """Export model to ONNX format."""
    model.eval()
    dummy_input = torch.randn(*input_size)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=13,
    )
    print(f"Exported ONNX model to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune pain detection model")
    parser.add_argument("--data-dir", required=True, help="Path to preprocessed dataset")
    parser.add_argument("--base-model", default=None, help="Path to base model (.pt)")
    parser.add_argument("--output-dir", default="models/archive", help="Output directory for trained model")
    parser.add_argument("--version", default="0.2.0", help="Model version string")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--freeze-ratio", type=float, default=0.7, help="Fraction of early layers to freeze")
    parser.add_argument("--target-size", type=int, default=224, help="Image input size")
    args = parser.parse_args()

    if not TORCH_AVAILABLE:
        print("ERROR: PyTorch is required for training. Install with: pip install torch")
        sys.exit(1)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_size = (args.target_size, args.target_size)

    # Load datasets
    print("Loading datasets...")
    train_ds = PainDataset(str(data_dir / "train"), target_size)
    val_ds = PainDataset(str(data_dir / "val"), target_size)

    if len(train_ds) == 0:
        print(f"ERROR: No training samples found in {data_dir / 'train'}")
        sys.exit(1)

    print(f"  Train: {len(train_ds)} samples")
    print(f"  Val:   {len(val_ds)} samples")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size) if len(val_ds) > 0 else None

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_base_model(args.base_model)
    freeze_early_layers(model, args.freeze_ratio)
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    # Training loop
    best_val_loss = float("inf")
    best_epoch = 0
    start_time = time.time()

    print(f"\nTraining for {args.epochs} epochs (lr={args.lr}, batch_size={args.batch_size})")
    print("-" * 60)

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)

        log = f"Epoch {epoch:3d}/{args.epochs} | Train Loss: {train_metrics['loss']:.4f}"

        if val_loader:
            val_metrics = evaluate(model, val_loader, criterion, device)
            log += f" | Val Loss: {val_metrics['loss']:.4f} | MAE: {val_metrics['mae']:.3f} | RMSE: {val_metrics['rmse']:.3f}"

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_epoch = epoch
                # Save best checkpoint
                best_path = output_dir / f"model-v{args.version}-best.pt"
                torch.save(model, str(best_path))
                log += " *best*"

        print(log)

    elapsed = time.time() - start_time
    print("-" * 60)
    print(f"Training complete in {elapsed:.1f}s")
    print(f"Best epoch: {best_epoch} (val_loss: {best_val_loss:.4f})")

    # Save final model
    final_pt_path = output_dir / f"model-v{args.version}.pt"
    torch.save(model, str(final_pt_path))
    print(f"Saved PyTorch model: {final_pt_path}")

    # Export ONNX
    final_onnx_path = output_dir / f"model-v{args.version}.onnx"
    export_onnx(model, str(final_onnx_path), input_size=(1, 3, *target_size))

    # Save metadata
    meta = {
        "version": args.version,
        "format": "onnx",
        "description": f"Fine-tuned pain detection model v{args.version}",
        "training": {
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "best_epoch": best_epoch,
            "best_val_loss": round(best_val_loss, 4),
            "elapsed_s": round(elapsed, 1),
        },
    }

    meta_path = output_dir / f"model-v{args.version}-meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata: {meta_path}")

    # Evaluation report
    if val_loader:
        print("\n--- Evaluation Report ---")
        val_metrics = evaluate(model, val_loader, criterion, device)
        print(f"  Validation MAE:  {val_metrics['mae']:.3f} (0-10 scale)")
        print(f"  Validation RMSE: {val_metrics['rmse']:.3f} (0-10 scale)")

    print(f"\nTo activate: copy {final_onnx_path} to models/active/")


if __name__ == "__main__":
    main()
