"""Tests for export_corrections module."""

import csv
import os
import sqlite3
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from export_corrections import export_corrections


@pytest.fixture
def sample_db(tmp_path):
    """Create a temporary database with test data."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            timestamp_ms REAL NOT NULL,
            score REAL NOT NULL,
            raw_score REAL,
            pspi REAL,
            sensor_data TEXT,
            frame_filename TEXT,
            ai_score REAL,
            ai_confidence REAL,
            model_version TEXT,
            corrected_score REAL,
            corrected_by TEXT,
            corrected_at TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)

    # Insert corrected samples
    conn.execute("""
        INSERT INTO samples (session_id, timestamp_ms, score, pspi, ai_score, ai_confidence,
            model_version, corrected_score, corrected_by, corrected_at, frame_filename)
        VALUES ('sess-1', 1000, 3.5, 3.2, 4.0, 0.85, 'v1.0', 3.0, 'dr_smith', '2025-01-01T10:00:00', 'sess-1_1000.jpg')
    """)
    conn.execute("""
        INSERT INTO samples (session_id, timestamp_ms, score, pspi, ai_score, ai_confidence,
            model_version, corrected_score, corrected_by, corrected_at, frame_filename)
        VALUES ('sess-1', 2000, 5.0, 4.8, 6.0, 0.72, 'v1.0', 5.5, 'dr_smith', '2025-01-01T10:01:00', 'sess-1_2000.jpg')
    """)

    # Insert uncorrected sample (should not be exported)
    conn.execute("""
        INSERT INTO samples (session_id, timestamp_ms, score, pspi, ai_score, ai_confidence, model_version)
        VALUES ('sess-1', 3000, 2.0, 1.8, 2.5, 0.90, 'v1.0')
    """)

    conn.commit()
    conn.close()

    # Create frames directory with dummy images
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    (frames_dir / "sess-1_1000.jpg").write_bytes(b"fake-jpeg-1")
    (frames_dir / "sess-1_2000.jpg").write_bytes(b"fake-jpeg-2")

    return str(db_path)


class TestExportCorrections:
    def test_exports_corrected_samples(self, sample_db, tmp_path):
        output_dir = str(tmp_path / "output")
        count = export_corrections(sample_db, output_dir)
        assert count == 2

        csv_path = os.path.join(output_dir, "corrections.csv")
        assert os.path.exists(csv_path)

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
        assert len(reader) == 2
        assert reader[0]["corrected_score"] == "3.0"
        assert reader[1]["corrected_score"] == "5.5"

    def test_csv_has_all_columns(self, sample_db, tmp_path):
        output_dir = str(tmp_path / "output")
        export_corrections(sample_db, output_dir)

        csv_path = os.path.join(output_dir, "corrections.csv")
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            row = next(reader)

        expected_cols = [
            "id", "session_id", "timestamp_ms", "pspi_score",
            "ai_score", "ai_confidence", "corrected_score",
            "corrected_by", "corrected_at", "frame_filename",
        ]
        for col in expected_cols:
            assert col in row, f"Missing column: {col}"

    def test_copies_frames_when_requested(self, sample_db, tmp_path):
        output_dir = str(tmp_path / "output")
        count = export_corrections(sample_db, output_dir, include_frames=True)
        assert count == 2

        frames_dst = os.path.join(output_dir, "frames")
        assert os.path.exists(frames_dst)
        assert os.path.exists(os.path.join(frames_dst, "sess-1_1000.jpg"))
        assert os.path.exists(os.path.join(frames_dst, "sess-1_2000.jpg"))

    def test_no_frames_without_flag(self, sample_db, tmp_path):
        output_dir = str(tmp_path / "output")
        export_corrections(sample_db, output_dir, include_frames=False)
        frames_dst = os.path.join(output_dir, "frames")
        assert not os.path.exists(frames_dst)

    def test_empty_database(self, tmp_path):
        db_path = str(tmp_path / "empty.db")
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE samples (
                id INTEGER PRIMARY KEY, session_id TEXT, timestamp_ms REAL,
                score REAL, raw_score REAL, pspi REAL, frame_filename TEXT,
                ai_score REAL, ai_confidence REAL, model_version TEXT,
                corrected_score REAL, corrected_by TEXT, corrected_at TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.commit()
        conn.close()

        output_dir = str(tmp_path / "output")
        count = export_corrections(db_path, output_dir)
        assert count == 0

    def test_missing_database(self, tmp_path):
        output_dir = str(tmp_path / "output")
        count = export_corrections("/nonexistent/path.db", output_dir)
        assert count == 0

    def test_preserves_ai_metadata(self, sample_db, tmp_path):
        output_dir = str(tmp_path / "output")
        export_corrections(sample_db, output_dir)

        csv_path = os.path.join(output_dir, "corrections.csv")
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        assert rows[0]["ai_score"] == "4.0"
        assert rows[0]["ai_confidence"] == "0.85"
        assert rows[0]["model_version"] == "v1.0"
        assert rows[0]["corrected_by"] == "dr_smith"
