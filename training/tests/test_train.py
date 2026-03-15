"""Tests for the training script components."""

import csv
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from train import PainDataset, TORCH_AVAILABLE

if TORCH_AVAILABLE:
    from train import SimplePainModel, freeze_early_layers


@pytest.fixture
def mock_split(tmp_path):
    """Create a mock train split directory with images and labels."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    csv_path = tmp_path / "labels.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "subject_id", "sequence_id", "pain_score", "aus"])
        for i in range(10):
            filename = f"subj_seq_{i:04d}.jpg"
            img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            cv2.imwrite(str(images_dir / filename), img)
            writer.writerow([filename, "subj", "seq", f"{i * 1.0}", f"AU4={0.1*i}"])

    return tmp_path


class TestPainDataset:
    def test_loads_samples(self, mock_split):
        ds = PainDataset(str(mock_split))
        assert len(ds) == 10

    def test_returns_image_and_label(self, mock_split):
        ds = PainDataset(str(mock_split))
        img, label = ds[0]
        if TORCH_AVAILABLE:
            assert img.shape == (3, 224, 224)  # CHW
            assert 0 <= float(label) <= 1  # Normalized
        else:
            assert img.shape == (3, 224, 224)

    def test_empty_dir(self, tmp_path):
        ds = PainDataset(str(tmp_path))
        assert len(ds) == 0

    def test_missing_images_handled(self, tmp_path):
        """Dataset still works if images are missing."""
        csv_path = tmp_path / "labels.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "subject_id", "sequence_id", "pain_score", "aus"])
            writer.writerow(["missing.jpg", "s", "q", "5.0", ""])

        ds = PainDataset(str(tmp_path))
        assert len(ds) == 1
        img, label = ds[0]  # Should return zeros, not crash


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestSimplePainModel:
    def test_forward_pass(self):
        import torch
        model = SimplePainModel(n_classes=1)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out.shape == (2, 1)
        assert (out >= 0).all() and (out <= 1).all()  # Sigmoid output

    def test_forward_different_sizes(self):
        import torch
        model = SimplePainModel(n_classes=1)
        for size in [64, 128, 224]:
            x = torch.randn(1, 3, size, size)
            out = model(x)
            assert out.shape == (1, 1)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestFreezeEarlyLayers:
    def test_freezes_parameters(self):
        model = SimplePainModel(n_classes=1)
        total_before = sum(p.requires_grad for p in model.parameters())
        freeze_early_layers(model, freeze_ratio=0.7)
        trainable = sum(p.requires_grad for p in model.parameters())
        assert trainable < total_before
        assert trainable > 0
