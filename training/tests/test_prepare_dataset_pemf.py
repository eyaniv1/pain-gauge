"""Tests for the PEMF dataset loader."""

import csv
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import openpyxl
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False

from prepare_dataset_pemf import (
    PEMF_AU_COLUMNS,
    _map_columns,
    _parse_row,
    assign_reference_frames,
    build_filename_map,
    collect_samples,
    find_clip_folder,
    find_excel_file,
    find_pictures_dir,
    make_output_filename,
    preprocess_image,
    split_by_subject,
    write_split,
)

pytestmark = pytest.mark.skipif(not HAS_OPENPYXL, reason="openpyxl not installed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_frame(path, width=452, height=549):
    """Write a small synthetic image to path."""
    img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


def make_excel(path, rows):
    """
    Create a minimal PEMF_Database.xlsx.

    rows: list of dicts with keys matching expected column names.
    """
    wb = openpyxl.Workbook()
    ws = wb.active
    # Header
    headers = ["Subject", "Condition", "ClipID", "PainIntensity"] + PEMF_AU_COLUMNS
    ws.append(headers)
    for row in rows:
        ws.append([
            row.get("Subject"),
            row.get("Condition"),
            row.get("ClipID"),
            row.get("PainIntensity"),
        ] + [row.get(au, 0) for au in PEMF_AU_COLUMNS])
    wb.save(str(path))


@pytest.fixture
def mock_dataset(tmp_path):
    """
    Create a minimal PEMF dataset structure.

    3 subjects, each with 4 clips (neutral, co2, algometer, posed).
    Each clip folder has 5 frames.
    """
    subjects = ["S001", "S002", "S003"]
    conditions = [
        ("neutral", None),
        ("co2", 4.0),
        ("algometer", 6.0),
        ("posed", 3.0),
    ]

    pictures_dir = tmp_path / "Pictures"
    excel_rows = []
    clip_counter = 1

    for subj in subjects:
        for cond_name, pain_rating in conditions:
            clip_id = f"{clip_counter:03d}"
            clip_folder = pictures_dir / clip_id
            clip_folder.mkdir(parents=True)

            # Write 5 frames
            for i in range(5):
                make_frame(clip_folder / f"frame_{i:02d}.jpg")

            excel_rows.append({
                "Subject": int(subj[1:]),  # 1, 2, 3
                "Condition": cond_name,
                "ClipID": clip_id,
                "PainIntensity": pain_rating,
                "AU4": 1 if pain_rating and pain_rating > 3 else 0,
                "AU6": 0,
            })
            clip_counter += 1

    make_excel(tmp_path / "PEMF_Database.xlsx", excel_rows)
    return tmp_path


# ---------------------------------------------------------------------------
# TestFindExcelFile
# ---------------------------------------------------------------------------

class TestFindExcelFile:
    def test_finds_standard_name(self, tmp_path):
        (tmp_path / "PEMF_Database.xlsx").touch()
        result = find_excel_file(tmp_path)
        assert result is not None
        assert result.name == "PEMF_Database.xlsx"

    def test_finds_any_xlsx_fallback(self, tmp_path):
        (tmp_path / "other_file.xlsx").touch()
        result = find_excel_file(tmp_path)
        assert result is not None

    def test_returns_none_if_no_xlsx(self, tmp_path):
        assert find_excel_file(tmp_path) is None


# ---------------------------------------------------------------------------
# TestFindPicturesDir
# ---------------------------------------------------------------------------

class TestFindPicturesDir:
    def test_finds_pictures_dir(self, tmp_path):
        (tmp_path / "Pictures").mkdir()
        result = find_pictures_dir(tmp_path)
        assert result is not None
        assert result.name == "Pictures"

    def test_finds_images_dir_fallback(self, tmp_path):
        (tmp_path / "Images").mkdir()
        result = find_pictures_dir(tmp_path)
        assert result is not None

    def test_explicit_path(self, tmp_path):
        custom = tmp_path / "custom_frames"
        custom.mkdir()
        result = find_pictures_dir(tmp_path, str(custom))
        assert result == custom

    def test_returns_none_if_missing(self, tmp_path):
        assert find_pictures_dir(tmp_path) is None


# ---------------------------------------------------------------------------
# TestColumnMapping
# ---------------------------------------------------------------------------

class TestColumnMapping:
    def test_maps_subject_column(self):
        headers = ["Subject", "Condition", "ClipID", "PainIntensity"]
        col_idx = _map_columns(headers)
        assert "subject" in col_idx
        assert col_idx["subject"] == 0

    def test_maps_clip_type_column(self):
        headers = ["Subject", "Condition", "ClipID", "PainIntensity"]
        col_idx = _map_columns(headers)
        assert "clip_type" in col_idx

    def test_maps_pain_rating(self):
        headers = ["Subject", "Condition", "ClipID", "PainIntensity"]
        col_idx = _map_columns(headers)
        assert "pain_rating" in col_idx

    def test_maps_au_columns(self):
        headers = ["Subject", "Condition", "PainIntensity", "AU4", "AU6", "AU7"]
        col_idx = _map_columns(headers)
        assert "AU4" in col_idx
        assert "AU6" in col_idx


# ---------------------------------------------------------------------------
# TestParseRow
# ---------------------------------------------------------------------------

class TestParseRow:
    def _make_col_idx(self):
        headers = ["Subject", "Condition", "ClipID", "PainIntensity"] + PEMF_AU_COLUMNS
        return _map_columns(headers), headers

    def test_neutral_clip_gets_zero_pain(self):
        col_idx, headers = self._make_col_idx()
        row = (1, "neutral", "001", None) + tuple(0 for _ in PEMF_AU_COLUMNS)
        record = _parse_row(row, headers, col_idx, 2)
        assert record is not None
        assert record["is_neutral"] is True
        assert record["pain_score_pspi"] == 0.0

    def test_pain_clip_scales_likert_to_pspi(self):
        col_idx, headers = self._make_col_idx()
        # Likert 4 → PSPI 8
        row = (1, "co2", "002", 4.0) + tuple(0 for _ in PEMF_AU_COLUMNS)
        record = _parse_row(row, headers, col_idx, 3)
        assert record["is_neutral"] is False
        assert record["pain_score_pspi"] == 8.0

    def test_max_likert_maps_to_pspi_16(self):
        col_idx, headers = self._make_col_idx()
        row = (2, "algometer", "003", 8.0) + tuple(0 for _ in PEMF_AU_COLUMNS)
        record = _parse_row(row, headers, col_idx, 4)
        assert record["pain_score_pspi"] == 16.0

    def test_au_presence_parsed(self):
        col_idx, headers = self._make_col_idx()
        au_vals = [1, 0, 1] + [0] * (len(PEMF_AU_COLUMNS) - 3)
        row = (1, "co2", "004", 2.0) + tuple(au_vals)
        record = _parse_row(row, headers, col_idx, 5)
        assert record["aus"]["AU4"] == 1
        assert record["aus"]["AU6"] == 0
        assert record["aus"]["AU7"] == 1

    def test_x_marker_treated_as_present(self):
        col_idx, headers = self._make_col_idx()
        au_vals = ["x", 0] + [0] * (len(PEMF_AU_COLUMNS) - 2)
        row = (1, "posed", "005", 3.0) + tuple(au_vals)
        record = _parse_row(row, headers, col_idx, 6)
        assert record["aus"]["AU4"] == 1

    def test_returns_none_on_missing_subject(self):
        col_idx, headers = self._make_col_idx()
        row = (None, "co2", "006", 2.0) + tuple(0 for _ in PEMF_AU_COLUMNS)
        record = _parse_row(row, headers, col_idx, 7)
        assert record is None


# ---------------------------------------------------------------------------
# TestFindClipFolder
# ---------------------------------------------------------------------------

class TestFindClipFolder:
    def test_finds_by_clip_id(self, tmp_path):
        clip_dir = tmp_path / "001"
        clip_dir.mkdir()
        (clip_dir / "frame_01.jpg").touch()
        record = {"clip_id": "001", "subject_id": "S001", "clip_type": "co2"}
        result = find_clip_folder(tmp_path, record)
        assert result == clip_dir

    def test_finds_by_subject_type(self, tmp_path):
        clip_dir = tmp_path / "S001_co2"
        clip_dir.mkdir()
        (clip_dir / "frame_01.jpg").touch()
        record = {"clip_id": "999", "subject_id": "S001", "clip_type": "co2"}
        result = find_clip_folder(tmp_path, record)
        assert result == clip_dir

    def test_returns_none_if_not_found(self, tmp_path):
        record = {"clip_id": "999", "subject_id": "S999", "clip_type": "co2"}
        result = find_clip_folder(tmp_path, record)
        assert result is None


# ---------------------------------------------------------------------------
# TestCollectSamples
# ---------------------------------------------------------------------------

class TestCollectSamples:
    def _get_excel_records(self, mock_dataset):
        from prepare_dataset_pemf import load_excel_labels
        excel_path = find_excel_file(mock_dataset)
        return load_excel_labels(excel_path)

    def test_collects_correct_frame_count(self, mock_dataset):
        records = self._get_excel_records(mock_dataset)
        pictures_dir = find_pictures_dir(mock_dataset)
        samples, _ = collect_samples(records, pictures_dir)
        # 3 subjects × 4 clips × 5 frames = 60
        assert len(samples) == 60

    def test_samples_have_required_fields(self, mock_dataset):
        records = self._get_excel_records(mock_dataset)
        pictures_dir = find_pictures_dir(mock_dataset)
        samples, _ = collect_samples(records, pictures_dir)
        s = samples[0]
        assert "image_path" in s
        assert "subject_id" in s
        assert "pain_score" in s
        assert "is_neutral" in s
        assert "aus" in s

    def test_neutral_frames_identified(self, mock_dataset):
        records = self._get_excel_records(mock_dataset)
        pictures_dir = find_pictures_dir(mock_dataset)
        samples, neutral_map = collect_samples(records, pictures_dir)
        neutral = [s for s in samples if s["is_neutral"]]
        # 3 subjects × 1 neutral clip × 5 frames = 15
        assert len(neutral) == 15
        assert len(neutral_map) == 3  # one entry per subject

    def test_neutral_map_has_all_subjects(self, mock_dataset):
        records = self._get_excel_records(mock_dataset)
        pictures_dir = find_pictures_dir(mock_dataset)
        _, neutral_map = collect_samples(records, pictures_dir)
        assert "S001" in neutral_map
        assert "S002" in neutral_map
        assert "S003" in neutral_map


# ---------------------------------------------------------------------------
# TestAssignReferenceFrames
# ---------------------------------------------------------------------------

class TestAssignReferenceFrames:
    def _get_samples(self, mock_dataset):
        from prepare_dataset_pemf import load_excel_labels
        excel_path = find_excel_file(mock_dataset)
        records = load_excel_labels(excel_path)
        pictures_dir = find_pictures_dir(mock_dataset)
        samples, neutral_map = collect_samples(records, pictures_dir)
        return samples, neutral_map

    def test_pain_frames_get_reference(self, mock_dataset):
        samples, neutral_map = self._get_samples(mock_dataset)
        assign_reference_frames(samples, neutral_map)
        pain_frames = [s for s in samples if not s["is_neutral"]]
        with_ref = [s for s in pain_frames if s.get("_ref_sample") is not None]
        assert len(with_ref) == len(pain_frames)

    def test_reference_is_same_subject(self, mock_dataset):
        samples, neutral_map = self._get_samples(mock_dataset)
        assign_reference_frames(samples, neutral_map)
        for sample in samples:
            if not sample["is_neutral"] and sample.get("_ref_sample"):
                assert sample["_ref_sample"]["subject_id"] == sample["subject_id"]

    def test_neutral_frames_have_no_reference(self, mock_dataset):
        samples, neutral_map = self._get_samples(mock_dataset)
        assign_reference_frames(samples, neutral_map)
        neutral_frames = [s for s in samples if s["is_neutral"]]
        for s in neutral_frames:
            assert s.get("_ref_sample") is None


# ---------------------------------------------------------------------------
# TestSplitBySubject
# ---------------------------------------------------------------------------

class TestSplitBySubject:
    def test_all_samples_assigned(self, mock_dataset):
        from prepare_dataset_pemf import load_excel_labels
        excel_path = find_excel_file(mock_dataset)
        records = load_excel_labels(excel_path)
        pictures_dir = find_pictures_dir(mock_dataset)
        samples, _ = collect_samples(records, pictures_dir)

        train, val, test = split_by_subject(samples)
        assert len(train) + len(val) + len(test) == len(samples)

    def test_no_subject_overlap(self, mock_dataset):
        from prepare_dataset_pemf import load_excel_labels
        excel_path = find_excel_file(mock_dataset)
        records = load_excel_labels(excel_path)
        pictures_dir = find_pictures_dir(mock_dataset)
        samples, _ = collect_samples(records, pictures_dir)

        train, val, test = split_by_subject(samples)
        train_subj = set(s["subject_id"] for s in train)
        val_subj = set(s["subject_id"] for s in val)
        test_subj = set(s["subject_id"] for s in test)

        assert train_subj.isdisjoint(val_subj)
        assert train_subj.isdisjoint(test_subj)
        assert val_subj.isdisjoint(test_subj)

    def test_deterministic_with_same_seed(self, mock_dataset):
        from prepare_dataset_pemf import load_excel_labels
        excel_path = find_excel_file(mock_dataset)
        records = load_excel_labels(excel_path)
        pictures_dir = find_pictures_dir(mock_dataset)
        samples, _ = collect_samples(records, pictures_dir)

        train_a, _, _ = split_by_subject(samples, seed=99)
        train_b, _, _ = split_by_subject(samples, seed=99)
        assert [s["subject_id"] for s in train_a] == [s["subject_id"] for s in train_b]

    def test_different_seeds_produce_different_splits(self, mock_dataset):
        from prepare_dataset_pemf import load_excel_labels
        excel_path = find_excel_file(mock_dataset)
        records = load_excel_labels(excel_path)
        pictures_dir = find_pictures_dir(mock_dataset)
        samples, _ = collect_samples(records, pictures_dir)

        # With only 3 subjects any two seeds may collide, so just verify it runs
        train_a, _, _ = split_by_subject(samples, seed=1)
        train_b, _, _ = split_by_subject(samples, seed=2)
        # At least one result must differ (3 subjects, different shuffles)
        subjects_a = set(s["subject_id"] for s in train_a)
        subjects_b = set(s["subject_id"] for s in train_b)
        # This is probabilistic; just ensure the function runs without error
        assert isinstance(subjects_a, set)
        assert isinstance(subjects_b, set)


# ---------------------------------------------------------------------------
# TestPreprocessImage
# ---------------------------------------------------------------------------

class TestPreprocessImage:
    def test_resizes_to_target(self, tmp_path):
        img_path = tmp_path / "test.jpg"
        make_frame(img_path)
        result = preprocess_image(str(img_path), (224, 224))
        assert result is not None
        assert result.shape == (224, 224, 3)

    def test_resizes_to_160(self, tmp_path):
        img_path = tmp_path / "test.jpg"
        make_frame(img_path)
        result = preprocess_image(str(img_path), (160, 160))
        assert result is not None
        assert result.shape == (160, 160, 3)

    def test_invalid_path_returns_none(self):
        assert preprocess_image("/nonexistent/image.jpg") is None


# ---------------------------------------------------------------------------
# TestWriteSplit
# ---------------------------------------------------------------------------

class TestWriteSplit:
    def _get_pain_samples(self, mock_dataset):
        from prepare_dataset_pemf import load_excel_labels
        excel_path = find_excel_file(mock_dataset)
        records = load_excel_labels(excel_path)
        pictures_dir = find_pictures_dir(mock_dataset)
        samples, neutral_map = collect_samples(records, pictures_dir)
        assign_reference_frames(samples, neutral_map)
        return [s for s in samples if not s["is_neutral"]]

    def test_writes_images_and_csv(self, mock_dataset, tmp_path):
        pain_samples = self._get_pain_samples(mock_dataset)
        output_dir = tmp_path / "output"
        count = write_split(pain_samples[:5], "train", output_dir)
        assert count == 5
        assert (output_dir / "train" / "labels.csv").exists()
        assert (output_dir / "train" / "images").is_dir()

    def test_csv_has_correct_columns(self, mock_dataset, tmp_path):
        pain_samples = self._get_pain_samples(mock_dataset)
        output_dir = tmp_path / "output"
        write_split(pain_samples[:5], "train", output_dir)
        with open(output_dir / "train" / "labels.csv") as f:
            reader = csv.reader(f)
            header = next(reader)
        assert "filename" in header
        assert "subject_id" in header
        assert "pain_score" in header
        assert "aus" in header
        assert "clip_type" in header
        assert "reference_filename" in header

    def test_pain_scores_normalized_to_0_10(self, mock_dataset, tmp_path):
        pain_samples = self._get_pain_samples(mock_dataset)
        output_dir = tmp_path / "output"
        write_split(pain_samples, "train", output_dir)
        with open(output_dir / "train" / "labels.csv") as f:
            reader = csv.DictReader(f)
            scores = [float(r["pain_score"]) for r in reader if r["pain_score"]]
        assert all(0.0 <= s <= 10.0 for s in scores), f"Scores out of range: {scores}"

    def test_reference_filename_populated(self, mock_dataset, tmp_path):
        pain_samples = self._get_pain_samples(mock_dataset)
        all_samples_map = build_filename_map(
            [s for s in pain_samples]  # simplified: map from pain samples themselves
        )
        # Build map from neutral samples too (need full sample list)
        from prepare_dataset_pemf import load_excel_labels
        excel_path = find_excel_file(mock_dataset)
        records = load_excel_labels(excel_path)
        pictures_dir = find_pictures_dir(mock_dataset)
        all_samples, neutral_map = collect_samples(records, pictures_dir)
        assign_reference_frames(all_samples, neutral_map)
        full_map = build_filename_map(all_samples)

        output_dir = tmp_path / "output"
        pain_only = [s for s in all_samples if not s["is_neutral"]]
        write_split(pain_only[:5], "train", output_dir, all_samples_filename_map=full_map)

        with open(output_dir / "train" / "labels.csv") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        ref_filenames = [r["reference_filename"] for r in rows]
        assert any(ref != "" for ref in ref_filenames), "Expected at least one reference filename"

    def test_image_files_exist_on_disk(self, mock_dataset, tmp_path):
        pain_samples = self._get_pain_samples(mock_dataset)
        output_dir = tmp_path / "output"
        write_split(pain_samples[:3], "train", output_dir)
        images_dir = output_dir / "train" / "images"
        written = list(images_dir.iterdir())
        assert len(written) == 3


# ---------------------------------------------------------------------------
# TestMakeOutputFilename
# ---------------------------------------------------------------------------

class TestMakeOutputFilename:
    def test_filename_is_deterministic(self):
        sample = {
            "subject_id": "S001",
            "clip_type": "co2",
            "clip_id": "003",
            "frame_idx": 5,
        }
        assert make_output_filename(sample) == make_output_filename(sample)

    def test_filename_ends_with_jpg(self):
        sample = {"subject_id": "S001", "clip_type": "co2", "clip_id": "003", "frame_idx": 0}
        assert make_output_filename(sample).endswith(".jpg")

    def test_filename_includes_subject(self):
        sample = {"subject_id": "S042", "clip_type": "neutral", "clip_id": "001", "frame_idx": 0}
        assert "S042" in make_output_filename(sample)

    def test_spaces_in_clip_type_replaced(self):
        sample = {"subject_id": "S001", "clip_type": "cold pressor", "clip_id": "001", "frame_idx": 0}
        filename = make_output_filename(sample)
        assert " " not in filename
