"""
Image Preprocessing Pipeline

Decodes incoming images, detects faces, crops, resizes, and normalizes
for model inference. Single-pass pipeline optimized for speed.
"""

import base64
import os
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PreprocessResult:
    """Result of preprocessing an image."""
    image_array: np.ndarray  # Model-ready array: (1, C, H, W) float32
    face_found: bool
    face_bbox: tuple[int, int, int, int] | None = None  # x, y, w, h in original image


# Lazy-loaded face detector
_face_detector = None

# Default model path (relative to project root)
_MODEL_PATH = os.getenv(
    "MEDIAPIPE_FACE_MODEL",
    os.path.join(os.path.dirname(__file__), "..", "models", "mediapipe", "blaze_face_short_range.tflite"),
)


def _get_face_detector():
    """Lazy-load MediaPipe face detection (Tasks API) to avoid startup cost."""
    global _face_detector
    if _face_detector is None:
        import mediapipe as mp
        base_options = mp.tasks.BaseOptions(model_asset_path=_MODEL_PATH)
        options = mp.tasks.vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=0.5,
        )
        _face_detector = mp.tasks.vision.FaceDetector.create_from_options(options)
    return _face_detector


def decode_image(image_data: str | bytes) -> np.ndarray | None:
    """
    Decode a base64-encoded or raw bytes image into a BGR numpy array.

    Args:
        image_data: Base64 string (with or without data URI prefix) or raw bytes.

    Returns:
        BGR image as numpy array, or None if decoding fails.
    """
    try:
        if isinstance(image_data, str):
            # Strip data URI prefix if present
            if "," in image_data:
                image_data = image_data.split(",", 1)[1]
            raw_bytes = base64.b64decode(image_data)
        else:
            raw_bytes = image_data

        arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def detect_face(image_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
    """
    Detect the largest face in an image using MediaPipe Tasks API.

    Args:
        image_bgr: BGR image as numpy array.

    Returns:
        (x, y, w, h) bounding box of the largest face, or None if no face found.
    """
    try:
        import mediapipe as mp
        detector = _get_face_detector()

        # MediaPipe Tasks API expects RGB via mp.Image
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = detector.detect(mp_image)

        if not result.detections:
            return None

        # Find largest detection by area
        best = None
        best_area = 0
        for detection in result.detections:
            bb = detection.bounding_box
            area = bb.width * bb.height
            if area > best_area:
                best_area = area
                best = (bb.origin_x, bb.origin_y, bb.width, bb.height)

        return best
    except Exception:
        return None


def crop_and_resize(
    image_bgr: np.ndarray,
    bbox: tuple[int, int, int, int] | None,
    target_size: tuple[int, int] = (224, 224),
    padding_ratio: float = 0.2,
) -> np.ndarray:
    """
    Crop face region with padding and resize to target dimensions.

    If no bounding box is provided, resizes the full image.

    Args:
        image_bgr: BGR image.
        bbox: (x, y, w, h) face bounding box, or None for full image.
        target_size: (width, height) output dimensions.
        padding_ratio: Extra padding around the face box (0.2 = 20%).

    Returns:
        Resized BGR image of target_size.
    """
    h, w = image_bgr.shape[:2]

    if bbox is not None:
        bx, by, bw, bh = bbox
        # Add padding
        pad_w = int(bw * padding_ratio)
        pad_h = int(bh * padding_ratio)
        x1 = max(0, bx - pad_w)
        y1 = max(0, by - pad_h)
        x2 = min(w, bx + bw + pad_w)
        y2 = min(h, by + bh + pad_h)
        cropped = image_bgr[y1:y2, x1:x2]
    else:
        cropped = image_bgr

    resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_LINEAR)
    return resized


def normalize(image_bgr: np.ndarray) -> np.ndarray:
    """
    Normalize image for model input (ImageNet-style RGB).

    Converts BGR to RGB, scales to [0, 1], applies ImageNet normalization,
    and transposes to (1, C, H, W) format.

    Args:
        image_bgr: BGR image, uint8, shape (H, W, 3).

    Returns:
        Float32 array, shape (1, 3, H, W), normalized.
    """
    # BGR -> RGB
    img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Scale to [0, 1]
    img = img.astype(np.float32) / 255.0

    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std

    # HWC -> CHW, add batch dim
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    return img


# Lazy-loaded CLAHE instance
_clahe = None


def _get_clahe():
    global _clahe
    if _clahe is None:
        _clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return _clahe


def normalize_grayscale(image_bgr: np.ndarray) -> np.ndarray:
    """
    Normalize image for grayscale model input (TaatiTeam-style).

    Converts to grayscale, applies CLAHE histogram equalization,
    scales to [0, 1], returns (1, 1, H, W) format.

    Args:
        image_bgr: BGR image, uint8, shape (H, W, 3).

    Returns:
        Float32 array, shape (1, 1, H, W), normalized to [0, 1].
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = _get_clahe().apply(gray)
    img = gray.astype(np.float32) / 255.0
    return img.reshape(1, 1, img.shape[0], img.shape[1])


def preprocess(
    image_data: str | bytes,
    target_size: tuple[int, int] = (224, 224),
    require_face: bool = True,
    grayscale: bool = False,
) -> PreprocessResult | None:
    """
    Full preprocessing pipeline: decode → detect face → crop → resize → normalize.

    Args:
        image_data: Base64-encoded image string or raw bytes.
        target_size: Model input dimensions (width, height).
        require_face: If True, returns None when no face is detected.
        grayscale: If True, output grayscale with CLAHE (1,1,H,W) instead of RGB (1,3,H,W).

    Returns:
        PreprocessResult with model-ready array, or None on failure.
    """
    # Decode
    image_bgr = decode_image(image_data)
    if image_bgr is None:
        return None

    # Detect face
    bbox = detect_face(image_bgr)
    face_found = bbox is not None

    if require_face and not face_found:
        return None

    # Crop and resize
    resized = crop_and_resize(image_bgr, bbox, target_size)

    # Normalize
    if grayscale:
        image_array = normalize_grayscale(resized)
    else:
        image_array = normalize(resized)

    return PreprocessResult(
        image_array=image_array,
        face_found=face_found,
        face_bbox=bbox,
    )
