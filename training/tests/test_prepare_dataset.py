"""Tests for the UNBC-McMaster dataset loader."""

import csv
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from prepare_dataset import (
    collect_samples,
    parse_facs_file,
    parse_pain_score,
    preprocess_image,
    split_by_subject,
    write_split,
)


@pytest.fixture
def mock_dataset(tmp_path):
    """Create a mock UNBC-McMaster directory structure."""
    # Create 3 subjects with 2 sequences each
    for subj in ("042", "043", "047"):
        for seq in ("seq1", "seq2"):
            img_dir = tmp_path / "Images" / subj / seq
            img_dir.mkdir(parents=True)
            facs_dir = tmp_path / "Frame_Labels" / "FACS" / subj / seq
            facs_dir.mkdir(parents=True)
            pain_dir = tmp_path / "Frame_Labels" / "OPR" / subj / seq
            pain_dir.mkdir(parents=True)

            # Create 5 frames per sequence
            for i in range(5):
                frame_name = f"{subj}_{seq}_{i:04d}"

                # Image
                img = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
                cv2.imwrite(str(img_dir / f"{frame_name}.png"), img)

                # FACS labels
                with open(facs_dir / f"{frame_name}_facs.txt", "w") as f:
                    f.write(f"4 {0.2 * i}\n")
                    f.write(f"6 {0.1 * i}\n")

                # Pain score (0-5 OPI scale)
                with open(pain_dir / f"{frame_name}_opr.txt", "w") as f:
                    f.write(f"{min(i, 5)}\n")

    return tmp_path


class TestParseFacsFile:
    def test_parse_valid(self, tmp_path):
        facs_file = tmp_path / "test_facs.txt"
        facs_file.write_text("4 0.8\n6 0.3\n9 0.5\n")
        result = parse_facs_file(facs_file)
        assert result["AU4"] == 0.8
        assert result["AU6"] == 0.3
        assert result["AU9"] == 0.5

    def test_parse_missing_file(self, tmp_path):
        result = parse_facs_file(tmp_path / "nonexistent.txt")
        assert result == {}


class TestParsePainScore:
    def test_parse_valid(self, tmp_path):
        score_file = tmp_path / "score.txt"
        score_file.write_text("3.5\n")
        assert parse_pain_score(score_file) == 3.5

    def test_parse_missing(self, tmp_path):
        assert parse_pain_score(tmp_path / "nope.txt") is None


class TestCollectSamples:
    def test_collects_all_samples(self, mock_dataset):
        samples = collect_samples(mock_dataset)
        # 3 subjects × 2 sequences × 5 frames = 30
        assert len(samples) == 30

    def test_samples_have_required_fields(self, mock_dataset):
        samples = collect_samples(mock_dataset)
        s = samples[0]
        assert "image_path" in s
        assert "subject_id" in s
        assert "pain_score" in s
        assert "aus" in s

    def test_samples_have_pain_scores(self, mock_dataset):
        samples = collect_samples(mock_dataset)
        with_pain = [s for s in samples if s["pain_score"] is not None]
        assert len(with_pain) == 30

    def test_samples_have_aus(self, mock_dataset):
        samples = collect_samples(mock_dataset)
        with_aus = [s for s in samples if s["aus"]]
        assert len(with_aus) > 0

    def test_empty_dir_returns_empty(self, tmp_path):
        samples = collect_samples(tmp_path)
        assert samples == []


class TestSplitBySubject:
    def test_split_sizes(self, mock_dataset):
        samples = collect_samples(mock_dataset)
        train, val, test = split_by_subject(samples)
        assert len(train) + len(val) + len(test) == len(samples)

    def test_no_subject_overlap(self, mock_dataset):
        samples = collect_samples(mock_dataset)
        train, val, test = split_by_subject(samples)

        train_subj = set(s["subject_id"] for s in train)
        val_subj = set(s["subject_id"] for s in val)
        test_subj = set(s["subject_id"] for s in test)

        assert train_subj.isdisjoint(val_subj)
        assert train_subj.isdisjoint(test_subj)
        assert val_subj.isdisjoint(test_subj)


class TestPreprocessImage:
    def test_resize(self, mock_dataset):
        # Find first image
        samples = collect_samples(mock_dataset)
        img = preprocess_image(samples[0]["image_path"], (224, 224))
        assert img is not None
        assert img.shape == (224, 224, 3)

    def test_invalid_path(self):
        assert preprocess_image("/nonexistent/image.png") is None


class TestWriteSplit:
    def test_writes_images_and_csv(self, mock_dataset, tmp_path):
        samples = collect_samples(mock_dataset)
        output_dir = tmp_path / "output"
        count = write_split(samples[:5], "train", output_dir)

        assert count == 5
        assert (output_dir / "train" / "labels.csv").exists()
        assert (output_dir / "train" / "images").is_dir()

        # Check CSV
        with open(output_dir / "train" / "labels.csv") as f:
            reader = csv.reader(f)
            rows = list(reader)
        assert rows[0] == ["filename", "subject_id", "sequence_id", "pain_score", "aus"]
        assert len(rows) == 6  # header + 5 data rows
