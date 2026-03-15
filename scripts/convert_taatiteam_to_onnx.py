"""
Convert TaatiTeam pain detection model to ONNX.

Exports the base comparative model with 2-channel input (target + reference frame).
The inference service handles reference frame management at runtime:
- When calibrated: uses the patient's calibration face as reference
- When not calibrated: uses a zero reference (fallback)

Output: 40 regression values. Last 3 are PSPI for [Dementia, Healthy, UNBC].
We use the UNBC PSPI (index -1) in the inference pipeline.
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

    def forward(self, x):
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
        return pred


def convert(checkpoint_path, output_path, num_outputs=40):
    """Load TaatiTeam checkpoint and export as 2-channel comparative ONNX model."""
    print(f"Loading checkpoint: {checkpoint_path}")

    # Load original model
    model = ConvNetOrdinalLateFusion(num_outputs=num_outputs)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    # Create dummy input: (batch=1, channels=2, 160x160)
    # Channel 0 = target face, Channel 1 = reference face
    dummy_input = torch.randn(1, 2, 160, 160)

    # Test forward pass
    with torch.no_grad():
        test_output = model(dummy_input)
        print(f"Test output shape: {test_output.shape}")
        # Last 3 outputs are PSPI for [Dementia, Healthy, UNBC]
        pspi_unbc = test_output[0, -1].item()
        print(f"UNBC PSPI (raw): {pspi_unbc:.2f}")

    # Export to ONNX using legacy exporter (more compatible)
    print(f"Exporting to: {output_path}")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["predictions"],
        dynamic_axes={"input": {0: "batch"}, "predictions": {0: "batch"}},
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

    # Test with zero reference (equivalent to uncalibrated mode)
    test_target = np.random.randn(1, 1, 160, 160).astype(np.float32)
    test_ref = np.zeros((1, 1, 160, 160), dtype=np.float32)
    test_combined = np.concatenate([test_target, test_ref], axis=1)
    result = session.run(None, {"input": test_combined})
    pspi = np.clip(result[0][0, -1], 0, 16)
    score = pspi * (10.0 / 16.0)
    print(f"ONNX test (zero ref): PSPI={pspi:.2f}, score={score:.2f}/10")

    # Test with same image as both target and reference (should give ~0 pain)
    test_same = np.concatenate([test_target, test_target], axis=1)
    result_same = session.run(None, {"input": test_same})
    pspi_same = np.clip(result_same[0][0, -1], 0, 16)
    score_same = pspi_same * (10.0 / 16.0)
    print(f"ONNX test (self ref): PSPI={pspi_same:.2f}, score={score_same:.2f}/10")

    print("Conversion complete!")


if __name__ == "__main__":
    # Default paths
    repo_dir = os.path.join(os.path.dirname(__file__), "..", "..", "pain_detection_demo")
    checkpoint = os.path.join(repo_dir, "checkpoints", "50342566", "50343918_3", "model_epoch4.pt")
    output = os.path.join(os.path.dirname(__file__), "..", "models", "active", "pain-unbc-v2.onnx")

    if len(sys.argv) > 1:
        checkpoint = sys.argv[1]
    if len(sys.argv) > 2:
        output = sys.argv[2]

    os.makedirs(os.path.dirname(output), exist_ok=True)
    convert(checkpoint, output)
