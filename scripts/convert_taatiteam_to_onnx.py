"""
Convert TaatiTeam pain detection model to ONNX.

The original model takes 2-channel input (target + reference frame).
This wrapper uses a learned zero reference, making it a single-image model.
The output is the PSPI pain score (0-16 scale, we'll normalize to 0-10 in our pipeline).
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np


# ── Reproduce the model architecture from TaatiTeam ──────────

class ConvNetOrdinalLateFusion(nn.Module):
    def __init__(self, num_outputs=1, dropout=0, fc2_size=200):
        super(ConvNetOrdinalLateFusion, self).__init__()
        self.dropout = dropout
        self.dout1 = nn.Dropout2d(dropout)
        self.input_bn = nn.BatchNorm2d(2, affine=False)
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout))
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=5, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout))
        self.fc1 = nn.Linear(4608, fc2_size)
        self.fc2 = nn.Linear(fc2_size, num_outputs)

    def forward(self, x, return_features=False):
        out = self.layer1(x[:, 0:1, ...])
        out_ref = self.layer1(x[:, 1:, ...])
        out = out - out_ref
        out = nn.functional.max_pool2d(out, kernel_size=2, stride=2)
        out = self.dout1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        features = self.fc1(out.reshape(out.size(0), -1))
        features = nn.functional.relu(features)
        pred = self.fc2(features)
        if return_features:
            return pred, features
        else:
            return pred


class SingleImagePainModel(nn.Module):
    """Wraps the comparative model to accept single-channel input.
    Uses a zero reference frame (no subject-specific calibration)."""

    def __init__(self, base_model):
        super(SingleImagePainModel, self).__init__()
        self.base_model = base_model

    def forward(self, x):
        # x is (batch, 1, H, W) — single grayscale image
        # Create zero reference frame
        ref = torch.zeros_like(x)
        # Concatenate: (batch, 2, H, W)
        combined = torch.cat([x, ref], dim=1)
        # Get prediction — last 3 outputs are PSPI for [Dementia, Healthy, UNBC]
        pred = self.base_model(combined)
        # Take UNBC PSPI prediction (index -1), clamp to 0-16
        pspi = torch.clamp(pred[:, -1:], 0, 16)
        # Normalize to 0-10 scale
        score = pspi * (10.0 / 16.0)
        return score


def convert(checkpoint_path, output_path, num_outputs=40):
    """Load TaatiTeam checkpoint and export as single-image ONNX model."""
    print(f"Loading checkpoint: {checkpoint_path}")

    # Load original model
    base_model = ConvNetOrdinalLateFusion(num_outputs=num_outputs)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    base_model.load_state_dict(state_dict)
    base_model.eval()

    # Wrap for single-image input
    wrapper = SingleImagePainModel(base_model)
    wrapper.eval()

    # Create dummy input (batch=1, channels=1, 160x160 grayscale)
    dummy_input = torch.randn(1, 1, 160, 160)

    # Test forward pass
    with torch.no_grad():
        test_output = wrapper(dummy_input)
        print(f"Test output shape: {test_output.shape}, value: {test_output.item():.2f}")

    # Export to ONNX using legacy exporter (more compatible)
    print(f"Exporting to: {output_path}")
    torch.onnx.export(
        wrapper,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["pain_score"],
        dynamic_axes={"input": {0: "batch"}, "pain_score": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )

    # Verify ONNX
    import onnxruntime as ort
    session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]
    print(f"ONNX input: {input_info.name} {input_info.shape}")
    print(f"ONNX output: {output_info.name} {output_info.shape}")

    result = session.run(None, {"input": dummy_input.numpy()})
    print(f"ONNX test output: {result[0].flatten()[0]:.2f}")
    print("Conversion complete!")


if __name__ == "__main__":
    # Default paths
    repo_dir = os.path.join(os.path.dirname(__file__), "..", "..", "pain_detection_demo")
    checkpoint = os.path.join(repo_dir, "checkpoints", "50342566", "50343918_3", "model_epoch4.pt")
    output = os.path.join(os.path.dirname(__file__), "..", "models", "active", "pain-unbc-v1.onnx")

    if len(sys.argv) > 1:
        checkpoint = sys.argv[1]
    if len(sys.argv) > 2:
        output = sys.argv[2]

    os.makedirs(os.path.dirname(output), exist_ok=True)
    convert(checkpoint, output)
