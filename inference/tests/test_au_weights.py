"""Tests for AU-weighted pain scoring."""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from au_weights import blend_scores, compute_au_score, reload_config


class TestComputeAUScore:
    def test_all_zeros(self):
        au_data = {"AU4": 0.0, "AU6": 0.0, "AU7": 0.0, "AU9": 0.0, "AU10": 0.0, "AU43": 0.0}
        score = compute_au_score(au_data)
        assert score == 0.0

    def test_all_max(self):
        au_data = {"AU4": 1.0, "AU6": 1.0, "AU7": 1.0, "AU9": 1.0, "AU10": 1.0, "AU43": 1.0}
        score = compute_au_score(au_data)
        assert score == 10.0

    def test_partial_aus(self):
        au_data = {"AU4": 0.8, "AU9": 0.5}
        score = compute_au_score(au_data)
        assert 0 < score < 10

    def test_empty_au_data(self):
        score = compute_au_score({})
        assert score == 0.0

    def test_unknown_aus_ignored(self):
        au_data = {"AU99": 1.0, "FAKE": 0.5}
        score = compute_au_score(au_data)
        assert score == 0.0

    def test_clamps_intensity(self):
        au_data = {"AU4": 2.0, "AU6": -0.5}
        score = compute_au_score(au_data)
        # AU4 clamped to 1.0, AU6 clamped to 0.0
        assert score > 0

    def test_score_in_range(self):
        au_data = {"AU4": 0.5, "AU6": 0.3, "AU7": 0.7, "AU9": 0.2, "AU10": 0.4, "AU43": 0.6}
        score = compute_au_score(au_data)
        assert 0 <= score <= 10


class TestBlendScores:
    def test_no_au_data_returns_model_score(self):
        result = blend_scores(5.0, None)
        assert result["final_score"] == 5.0
        assert result["au_score"] is None
        assert result["au_weights_applied"] is None

    def test_empty_au_data_returns_model_score(self):
        result = blend_scores(5.0, {})
        assert result["final_score"] == 5.0

    def test_blend_with_au_data(self):
        au_data = {"AU4": 1.0, "AU6": 1.0, "AU7": 1.0, "AU9": 1.0, "AU10": 1.0, "AU43": 1.0}
        result = blend_scores(5.0, au_data)
        # With default alpha=0.6: 0.6*5 + 0.4*10 = 3 + 4 = 7
        assert result["final_score"] == 7.0
        assert result["model_score"] == 5.0
        assert result["au_score"] == 10.0
        assert result["blend_alpha"] == 0.6
        assert result["au_weights_applied"] is not None

    def test_blend_preserves_range(self):
        au_data = {"AU4": 0.5}
        result = blend_scores(8.0, au_data)
        assert 0 <= result["final_score"] <= 10

    def test_au_weights_applied_has_expected_keys(self):
        au_data = {"AU4": 0.5}
        result = blend_scores(5.0, au_data)
        weights = result["au_weights_applied"]
        assert "AU4" in weights
        assert "AU6" in weights
        assert "AU43" in weights


class TestConfigReload:
    def test_reload_returns_config(self):
        config = reload_config()
        assert "weights" in config
        assert "blend_alpha" in config

    def test_custom_config_file(self, tmp_path, monkeypatch):
        custom = {"weights": {"AU4": 0.5, "AU9": 0.5}, "blend_alpha": 0.8}
        config_path = tmp_path / "custom_au.json"
        with open(config_path, "w") as f:
            json.dump(custom, f)

        monkeypatch.setenv("AU_WEIGHTS_CONFIG", str(config_path))
        config = reload_config()
        assert config["weights"]["AU4"] == 0.5
        assert config["blend_alpha"] == 0.8

        # Cleanup: reload default
        monkeypatch.delenv("AU_WEIGHTS_CONFIG")
        reload_config()
