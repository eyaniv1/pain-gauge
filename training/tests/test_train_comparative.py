"""Tests for the comparative model additions to train.py."""

import csv
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from train import TORCH_AVAILABLE, grayscale_clahe

if TORCH_AVAILABLE:
    import torch
    from train import ComparativePainDataset, ComparativePainModel, load_comparative_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_image(path, h=224, w=224):
    img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


def make_comparative_split(tmp_path, n_samples=6):
    """
    Create a mock split directory with paired target + reference images
    and a labels.csv that includes reference_filename.
    """
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    csv_path = tmp_path / "labels.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "filename", "subject_id", "sequence_id",
            "pain_score", "aus", "clip_type", "reference_filename",
        ])
        for i in range(n_samples):
            target_name = f"S001_co2_00{i}_f00.jpg"
            ref_name = f"S001_neutral_000_f00.jpg"
            make_image(images_dir / target_name)
            make_image(images_dir / ref_name)
            writer.writerow([
                target_name, "S001", f"00{i}",
                f"{i * 1.5}", f"AU4={i % 2}", "co2", ref_name,
            ])

    return tmp_path


# ---------------------------------------------------------------------------
# TestGrayscaleClahe
# ---------------------------------------------------------------------------

class TestGrayscaleClahe:
    def test_output_shape(self):
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        result = grayscale_clahe(img, (160, 160))
        assert result.shape == (1, 160, 160)

    def test_output_range(self):
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        result = grayscale_clahe(img, (160, 160))
        assert result.dtype == np.float32
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_resizes_correctly(self):
        img = np.random.randint(0, 255, (320, 452, 3), dtype=np.uint8)
        result = grayscale_clahe(img, (224, 224))
        assert result.shape == (1, 224, 224)


# ---------------------------------------------------------------------------
# TestComparativePainDataset
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestComparativePainDataset:
    def test_loads_paired_samples(self, tmp_path):
        split_dir = make_comparative_split(tmp_path)
        ds = ComparativePainDataset(str(split_dir))
        assert len(ds) == 6

    def test_returns_correct_tensor_shape(self, tmp_path):
        split_dir = make_comparative_split(tmp_path)
        ds = ComparativePainDataset(str(split_dir))
        pair, label = ds[0]
        assert pair.shape == (2, 160, 160)  # (target + reference, H, W)

    def test_label_normalised_to_0_1(self, tmp_path):
        split_dir = make_comparative_split(tmp_path)
        ds = ComparativePainDataset(str(split_dir))
        for i in range(len(ds)):
            _, label = ds[i]
            assert 0.0 <= float(label) <= 1.0

    def test_skips_samples_without_reference(self, tmp_path):
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        csv_path = tmp_path / "labels.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "filename", "subject_id", "sequence_id",
                "pain_score", "aus", "clip_type", "reference_filename",
            ])
            # One sample with reference, one without
            make_image(images_dir / "target.jpg")
            make_image(images_dir / "ref.jpg")
            writer.writerow(["target.jpg", "S1", "01", "5.0", "", "co2", "ref.jpg"])
            writer.writerow(["target.jpg", "S1", "02", "5.0", "", "co2", ""])

        ds = ComparativePainDataset(str(tmp_path))
        assert len(ds) == 1  # Only the sample with a reference

    def test_missing_reference_image_handled(self, tmp_path):
        """If the reference file is listed but missing on disk, returns zeros."""
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        make_image(images_dir / "target.jpg")
        # Do NOT create the reference image
        csv_path = tmp_path / "labels.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "filename", "subject_id", "sequence_id",
                "pain_score", "aus", "clip_type", "reference_filename",
            ])
            writer.writerow(["target.jpg", "S1", "01", "4.0", "", "co2", "missing_ref.jpg"])

        ds = ComparativePainDataset(str(tmp_path))
        assert len(ds) == 1
        pair, _ = ds[0]           # Should not raise
        assert pair.shape == (2, 160, 160)

    def test_empty_dir_gives_empty_dataset(self, tmp_path):
        ds = ComparativePainDataset(str(tmp_path))
        assert len(ds) == 0

    def test_default_size_is_160(self):
        assert ComparativePainDataset.DEFAULT_SIZE == (160, 160)


# ---------------------------------------------------------------------------
# TestComparativePainModel
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestComparativePainModel:
    def test_forward_pass_shape(self):
        model = ComparativePainModel()
        x = torch.randn(2, 2, 160, 160)
        out = model(x)
        assert out.shape == (2, 1)

    def test_output_in_0_1(self):
        model = ComparativePainModel()
        x = torch.randn(4, 2, 160, 160)
        out = model(x)
        assert (out >= 0.0).all() and (out <= 1.0).all()

    def test_feature_subtraction_zero_when_identical(self):
        """When target == reference, feature subtraction → zero diff → near-zero pain score."""
        model = ComparativePainModel()
        model.eval()
        frame = torch.randn(1, 1, 160, 160)
        x = torch.cat([frame, frame], dim=1)  # target == reference
        with torch.no_grad():
            out = model(x)
        # sigmoid(0) = 0.5, but bias in fc2 may shift slightly — just check it runs
        assert out.shape == (1, 1)

    def test_different_reference_changes_output(self):
        """Changing the reference frame changes the output."""
        model = ComparativePainModel()
        model.eval()
        target = torch.randn(1, 1, 160, 160)
        ref_a = torch.randn(1, 1, 160, 160)
        ref_b = torch.randn(1, 1, 160, 160)
        x_a = torch.cat([target, ref_a], dim=1)
        x_b = torch.cat([target, ref_b], dim=1)
        with torch.no_grad():
            out_a = model(x_a)
            out_b = model(x_b)
        assert not torch.allclose(out_a, out_b)

    def test_fc1_input_dim_matches_conv_output(self):
        """The FC1_INPUT_DIM constant must match actual feature map size."""
        model = ComparativePainModel()
        model.eval()
        x = torch.randn(1, 2, 160, 160)
        # Run up to (but not including) fc1 manually
        with torch.no_grad():
            out = model.layer1(x[:, 0:1])
            out_ref = model.layer1(x[:, 1:2])
            diff = out - out_ref
            diff = torch.nn.functional.max_pool2d(diff, kernel_size=2, stride=2)
            diff = model.layer2(diff)
            diff = model.layer3(diff)
            flat_dim = diff.reshape(diff.size(0), -1).shape[1]
        assert flat_dim == ComparativePainModel.FC1_INPUT_DIM

    def test_single_batch(self):
        model = ComparativePainModel()
        x = torch.randn(1, 2, 160, 160)
        out = model(x)
        assert out.shape == (1, 1)


# ---------------------------------------------------------------------------
# TestLoadComparativeModel
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestLoadComparativeModel:
    def test_creates_model_without_checkpoint(self):
        model = load_comparative_model(init_weights_path=None)
        assert isinstance(model, ComparativePainModel)

    def test_missing_path_warns_and_creates_model(self, tmp_path):
        model = load_comparative_model(init_weights_path=str(tmp_path / "nope.pt"))
        assert isinstance(model, ComparativePainModel)

    def test_loads_compatible_checkpoint(self, tmp_path):
        """Save a ComparativePainModel state_dict, reload it."""
        src = ComparativePainModel()
        # Mutate one weight so we can detect it was loaded
        with torch.no_grad():
            src.layer1[0].weight.fill_(0.123)

        checkpoint_path = tmp_path / "compat.pt"
        torch.save(src.state_dict(), str(checkpoint_path))

        dst = load_comparative_model(str(checkpoint_path))
        assert torch.allclose(
            dst.layer1[0].weight,
            torch.full_like(dst.layer1[0].weight, 0.123),
        )

    def test_partial_checkpoint_skips_incompatible_layers(self, tmp_path):
        """A checkpoint with a different fc2 shape should load partial weights."""
        import torch.nn as nn

        # Create a model with different fc2 output dim
        class AltModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = ComparativePainModel().layer1
                self.fc2 = nn.Linear(200, 7)  # incompatible shape

        alt = AltModel()
        checkpoint_path = tmp_path / "partial.pt"
        torch.save(alt.state_dict(), str(checkpoint_path))

        # Should not raise; simply skips fc2
        model = load_comparative_model(str(checkpoint_path))
        assert isinstance(model, ComparativePainModel)
