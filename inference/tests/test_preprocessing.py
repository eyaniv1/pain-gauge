"""Tests for the image preprocessing pipeline."""

import base64
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from preprocessing import (
    PreprocessResult,
    crop_and_resize,
    decode_image,
    normalize,
    preprocess,
)


def _make_test_image(width=320, height=240) -> np.ndarray:
    """Create a simple test image (solid color with a circle for 'face')."""
    img = np.full((height, width, 3), 180, dtype=np.uint8)
    # Draw a skin-colored circle in center as rough face proxy
    cv2.circle(img, (width // 2, height // 2), 60, (130, 160, 200), -1)
    return img


def _encode_image(img: np.ndarray, fmt: str = ".jpg") -> str:
    """Encode a numpy image to base64 string."""
    _, buf = cv2.imencode(fmt, img)
    return base64.b64encode(buf.tobytes()).decode("utf-8")


class TestDecodeImage:
    def test_decode_valid_jpeg(self):
        img = _make_test_image()
        b64 = _encode_image(img, ".jpg")
        result = decode_image(b64)
        assert result is not None
        assert result.shape[0] > 0 and result.shape[1] > 0

    def test_decode_valid_png(self):
        img = _make_test_image()
        b64 = _encode_image(img, ".png")
        result = decode_image(b64)
        assert result is not None

    def test_decode_with_data_uri_prefix(self):
        img = _make_test_image()
        b64 = "data:image/jpeg;base64," + _encode_image(img)
        result = decode_image(b64)
        assert result is not None

    def test_decode_invalid_data(self):
        result = decode_image("not-valid-base64!!!")
        assert result is None

    def test_decode_raw_bytes(self):
        img = _make_test_image()
        _, buf = cv2.imencode(".jpg", img)
        result = decode_image(buf.tobytes())
        assert result is not None


class TestCropAndResize:
    def test_resize_full_image_when_no_bbox(self):
        img = _make_test_image(320, 240)
        result = crop_and_resize(img, None, target_size=(64, 64))
        assert result.shape == (64, 64, 3)

    def test_crop_with_bbox(self):
        img = _make_test_image(320, 240)
        bbox = (100, 60, 120, 120)  # x, y, w, h
        result = crop_and_resize(img, bbox, target_size=(64, 64))
        assert result.shape == (64, 64, 3)

    def test_bbox_at_edge_no_crash(self):
        img = _make_test_image(320, 240)
        bbox = (0, 0, 50, 50)
        result = crop_and_resize(img, bbox, target_size=(64, 64))
        assert result.shape == (64, 64, 3)

    def test_bbox_exceeding_bounds(self):
        img = _make_test_image(320, 240)
        bbox = (280, 200, 100, 100)  # extends beyond image
        result = crop_and_resize(img, bbox, target_size=(64, 64))
        assert result.shape == (64, 64, 3)


class TestNormalize:
    def test_output_shape(self):
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        result = normalize(img)
        assert result.shape == (1, 3, 224, 224)
        assert result.dtype == np.float32

    def test_output_range(self):
        # White image
        img = np.full((64, 64, 3), 255, dtype=np.uint8)
        result = normalize(img)
        # After normalization, values should be roughly in [-2, 3] range
        assert result.min() > -3
        assert result.max() < 4

    def test_different_sizes(self):
        img = np.zeros((128, 96, 3), dtype=np.uint8)
        result = normalize(img)
        assert result.shape == (1, 3, 128, 96)


class TestPreprocess:
    def test_preprocess_returns_none_for_invalid_data(self):
        result = preprocess("garbage-data", require_face=True)
        assert result is None

    def test_preprocess_no_face_required(self):
        img = _make_test_image()
        b64 = _encode_image(img)
        result = preprocess(b64, target_size=(64, 64), require_face=False)
        assert result is not None
        assert isinstance(result, PreprocessResult)
        assert result.image_array.shape == (1, 3, 64, 64)

    def test_preprocess_output_dtype(self):
        img = _make_test_image()
        b64 = _encode_image(img)
        result = preprocess(b64, target_size=(64, 64), require_face=False)
        assert result.image_array.dtype == np.float32
