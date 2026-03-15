"""
Export Corrected Samples for Model Retraining

Queries the Pain Gauge SQLite database for samples with clinician corrections,
exports a CSV of labels and optionally copies associated frame images.

Usage:
    python export_corrections.py [--db PATH] [--output DIR] [--frames]
"""

import argparse
import csv
import os
import shutil
import sqlite3
import sys


DEFAULT_DB = os.path.join(os.path.dirname(__file__), "..", "server", "paingauge.db")
DEFAULT_OUTPUT = os.path.join(os.path.dirname(__file__), "data", "corrections")


def export_corrections(db_path, output_dir, include_frames=False):
    """Export corrected samples from the database.

    Args:
        db_path: Path to the SQLite database.
        output_dir: Directory to write CSV and optional frame images.
        include_frames: If True, copy frame images to output_dir/frames/.

    Returns:
        Number of corrected samples exported.
    """
    if not os.path.exists(db_path):
        print(f"Error: Database not found at {db_path}", file=sys.stderr)
        return 0

    os.makedirs(output_dir, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    query = """
        SELECT
            s.id,
            s.session_id,
            s.timestamp_ms,
            s.score AS pspi_score,
            s.raw_score,
            s.pspi,
            s.ai_score,
            s.ai_confidence,
            s.model_version,
            s.corrected_score,
            s.corrected_by,
            s.corrected_at,
            s.frame_filename
        FROM samples s
        WHERE s.corrected_score IS NOT NULL
        ORDER BY s.corrected_at
    """

    rows = conn.execute(query).fetchall()

    if not rows:
        print("No corrected samples found.")
        conn.close()
        return 0

    # Write CSV
    csv_path = os.path.join(output_dir, "corrections.csv")
    fieldnames = [
        "id", "session_id", "timestamp_ms",
        "pspi_score", "raw_score", "pspi",
        "ai_score", "ai_confidence", "model_version",
        "corrected_score", "corrected_by", "corrected_at",
        "frame_filename",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))

    print(f"Exported {len(rows)} corrected samples to {csv_path}")

    # Optionally copy frames
    if include_frames:
        frames_src = os.path.join(os.path.dirname(db_path), "frames")
        frames_dst = os.path.join(output_dir, "frames")
        os.makedirs(frames_dst, exist_ok=True)

        copied = 0
        for row in rows:
            fname = row["frame_filename"]
            if not fname:
                continue
            src = os.path.join(frames_src, fname)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(frames_dst, fname))
                copied += 1

        print(f"Copied {copied} frame images to {frames_dst}")

    conn.close()
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description="Export corrected samples for retraining")
    parser.add_argument("--db", default=DEFAULT_DB, help="Path to paingauge.db")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output directory")
    parser.add_argument("--frames", action="store_true", help="Also copy frame images")
    args = parser.parse_args()

    db_path = os.path.abspath(args.db)
    output_dir = os.path.abspath(args.output)

    count = export_corrections(db_path, output_dir, include_frames=args.frames)
    if count > 0:
        print(f"\nReady for retraining. Use corrections.csv as ground-truth labels.")
    else:
        print("\nNo corrections to export. Have clinicians review and correct AI scores first.")


if __name__ == "__main__":
    main()
