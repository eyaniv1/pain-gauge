"""Create a tiny ONNX model for testing purposes."""

import numpy as np


def create_dummy_onnx_model(output_path: str, n_classes: int = 8):
    """Create a minimal ONNX model that accepts 1x3x64x64 input and outputs n_classes logits."""
    try:
        import onnx
        from onnx import helper, TensorProto

        # Input: batch x channels x height x width
        X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 64, 64])
        Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, n_classes])

        # Simple: flatten then matmul with random weights
        flatten_node = helper.make_node("Flatten", inputs=["input"], outputs=["flat"], axis=1)

        # Weight matrix: (3*64*64) x n_classes
        w_data = np.random.randn(3 * 64 * 64, n_classes).astype(np.float32) * 0.01
        W = helper.make_tensor("W", TensorProto.FLOAT, [3 * 64 * 64, n_classes], w_data.flatten().tolist())

        matmul_node = helper.make_node("MatMul", inputs=["flat", "W"], outputs=["output"])

        graph = helper.make_graph(
            [flatten_node, matmul_node],
            "test_model",
            [X],
            [Y],
            initializer=[W],
        )

        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        onnx.save(model, output_path)
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "test_model.onnx"
    if create_dummy_onnx_model(out):
        print(f"Created dummy model at {out}")
    else:
        print("onnx package not installed, cannot create test model")
