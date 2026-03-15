"""
AU-Weighted Pain Scoring

Applies configurable Action Unit zone weights to blend with model predictions.
Weights derived from MDPI research on facial pain detection (92.72% accuracy).

AU zones: AU4 (brow lowerer), AU6 (cheek raiser), AU7 (lid tightener),
AU9 (nose wrinkler), AU10 (upper lip raiser), AU43 (eye closure).
"""

import json
import os
from pathlib import Path


# Default weights from MDPI paper
_DEFAULT_WEIGHTS = {
    "AU4": 0.35,
    "AU6": 0.10,
    "AU7": 0.10,
    "AU9": 0.15,
    "AU10": 0.15,
    "AU43": 0.15,
}

_DEFAULT_BLEND_ALPHA = 0.6  # 60% model, 40% AU-weighted

_config_cache: dict | None = None


def _load_config() -> dict:
    """Load AU weights from JSON config file, with defaults as fallback."""
    global _config_cache
    if _config_cache is not None:
        return _config_cache

    config_path = os.getenv(
        "AU_WEIGHTS_CONFIG",
        os.path.join(os.path.dirname(__file__), "au_weights.json"),
    )

    try:
        with open(config_path) as f:
            data = json.load(f)
        _config_cache = {
            "weights": data.get("weights", _DEFAULT_WEIGHTS),
            "blend_alpha": data.get("blend_alpha", _DEFAULT_BLEND_ALPHA),
        }
    except (FileNotFoundError, json.JSONDecodeError):
        _config_cache = {
            "weights": _DEFAULT_WEIGHTS.copy(),
            "blend_alpha": _DEFAULT_BLEND_ALPHA,
        }

    return _config_cache


def reload_config():
    """Force reload of AU weights config (e.g., after file update)."""
    global _config_cache
    _config_cache = None
    return _load_config()


def compute_au_score(au_data: dict[str, float]) -> float:
    """
    Compute a weighted AU-based pain score from AU intensities.

    Args:
        au_data: Dict of AU name → intensity (0.0 to 1.0).
                 e.g., {"AU4": 0.8, "AU6": 0.3, "AU9": 0.5}

    Returns:
        Pain score on 0-10 scale based on weighted AU intensities.
    """
    config = _load_config()
    weights = config["weights"]

    weighted_sum = 0.0
    total_weight = 0.0

    for au_name, weight in weights.items():
        intensity = au_data.get(au_name, 0.0)
        # Clamp intensity to [0, 1]
        intensity = max(0.0, min(1.0, float(intensity)))
        weighted_sum += intensity * weight
        total_weight += weight

    if total_weight == 0:
        return 0.0

    # Normalize to 0-10 scale
    normalized = (weighted_sum / total_weight) * 10.0
    return round(min(10.0, max(0.0, normalized)), 2)


def blend_scores(model_score: float, au_data: dict[str, float] | None) -> dict:
    """
    Blend model prediction with AU-weighted score.

    Args:
        model_score: Pain score from model (0-10).
        au_data: Optional AU intensity data. If None, returns model score unchanged.

    Returns:
        Dict with final_score, model_score, au_score (if applicable),
        blend_alpha, and au_weights_applied.
    """
    if au_data is None or not au_data:
        return {
            "final_score": model_score,
            "model_score": model_score,
            "au_score": None,
            "blend_alpha": None,
            "au_weights_applied": None,
        }

    config = _load_config()
    alpha = config["blend_alpha"]
    au_score = compute_au_score(au_data)

    final_score = round(alpha * model_score + (1 - alpha) * au_score, 2)
    final_score = min(10.0, max(0.0, final_score))

    return {
        "final_score": final_score,
        "model_score": model_score,
        "au_score": au_score,
        "blend_alpha": alpha,
        "au_weights_applied": config["weights"],
    }
