"""
PEMF (Pain E-Motion Faces Database) Dataset Loader

Open-access dataset: https://osf.io/3hgca/?view_only=12b04cd8164d4a6784c04b8c83bf95fb
Paper: doi.org/10.3758/s13428-022-01992-4

272 clips from 68 subjects × 4 clip types:
  - Neutral expression (suffix N)
  - Laser-induced spontaneous pain (suffix L)
  - Algometer-induced spontaneous pain (suffix A)
  - Posed pain (suffix P)

Each clip has 20 extracted still frames. Labels are in PEMF_Database.xlsx:
  - Columns: Clip (e.g. S001A), Kind, Intensity (mean(SD) in European decimal format),
             AU4..AU45 (0-6 inter-rater agreement intensity)
  - Pain intensity: 9-point Likert scale (0-8), normalized to PSPI-equivalent (×2 = 0-16)
  - AU values: 0-6 (number of raters who coded AU present)
  - Neutral clips: pain_score = 0

For comparative model training, the corresponding neutral clip from each subject is recorded
as a reference in labels.csv (reference_filename column).

Expected structure after downloading and extracting from OSF:
  {dataset_root}/
    PEMF_Database.xlsx
    Pictures/
      {clip_folder}/          (one folder per clip, named by clip ID or subject+type)
        frame_01.jpg
        ...
        frame_20.jpg

Usage:
  python prepare_dataset_pemf.py --dataset-root /path/to/PEMF --output-dir training/data/pemf
  python prepare_dataset_pemf.py --dataset-root /path/to/PEMF --output-dir training/data/pemf --pictures-dir /path/to/Pictures
"""

import argparse
import csv
import os
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

try:
    import openpyxl
except ImportError:
    print("ERROR: openpyxl is required. Install with: pip install openpyxl")
    sys.exit(1)


# PEMF AU columns (0-6 intensity scale in the real dataset, 6 = max rater agreement)
PEMF_AU_COLUMNS = ["AU4", "AU6", "AU7", "AU9", "AU10", "AU12", "AU20", "AU25", "AU26", "AU27", "AU43", "AU45"]

# Clip type strings to look for in Excel (case-insensitive)
NEUTRAL_KEYWORDS = ("neutral", "base", "baseline", "rest")
PAIN_KEYWORDS = ("laser", "algometer", "alg", "posed", "pain")

# Suffix → type mapping for composite clip IDs like "S001A"
CLIP_SUFFIX_MAP = {"a": "algometer", "l": "laser", "n": "neutral", "p": "posed"}


def find_excel_file(dataset_root: Path) -> Path | None:
    """Find PEMF_Database.xlsx (or any .xlsx) in dataset root."""
    for name in ("PEMF_Database.xlsx", "PEMF_database.xlsx", "pemf_database.xlsx"):
        candidate = dataset_root / name
        if candidate.exists():
            return candidate
    # Fallback: any xlsx in root
    xlsx_files = list(dataset_root.glob("*.xlsx"))
    if xlsx_files:
        return xlsx_files[0]
    return None


def find_pictures_dir(dataset_root: Path, pictures_dir: str | None = None) -> Path | None:
    """Find the directory containing per-clip image folders."""
    if pictures_dir:
        p = Path(pictures_dir)
        if p.is_dir():
            return p
        print(f"WARNING: --pictures-dir '{pictures_dir}' does not exist, searching automatically.")

    for name in ("Pictures", "pictures", "Images", "images", "Frames", "frames", "Stills", "stills"):
        candidate = dataset_root / name
        if candidate.is_dir():
            return candidate
    return None


def load_excel_labels(excel_path: Path) -> list[dict]:
    """
    Parse PEMF_Database.xlsx into a list of clip records.

    Each record has:
        clip_id, subject_id, clip_type, is_neutral, pain_score_raw (0-8 Likert),
        pain_score_pspi (0-16, raw × 2), aus (dict AU→0/1)
    """
    wb = openpyxl.load_workbook(excel_path, data_only=True)
    ws = wb.active

    # Read header row
    headers = []
    for cell in ws[1]:
        val = cell.value
        headers.append(str(val).strip() if val is not None else "")

    print(f"Excel columns found: {headers}")

    # Map column names flexibly
    col_idx = _map_columns(headers)
    print(f"Column mapping: {col_idx}")

    records = []
    for row_idx, row in enumerate(ws.iter_rows(min_row=2, values_only=True), start=2):
        if all(v is None for v in row):
            continue  # Skip empty rows

        record = _parse_row(row, headers, col_idx, row_idx)
        if record is not None:
            records.append(record)

    print(f"Loaded {len(records)} clip records from Excel")
    return records


def parse_european_mean(value) -> float | None:
    """
    Parse a European-formatted mean(SD) string into a float.

    Examples:
        "5,20 (2,516)" → 5.20
        "3,82 (2)"     → 3.82
        "0,28 (0,86)"  → 0.28
        3.5             → 3.5   (already numeric)
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if not s:
        return None
    # Strip parenthesised SD portion
    if "(" in s:
        s = s[:s.index("(")].strip()
    # European comma → dot
    s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def _map_columns(headers: list[str]) -> dict:
    """Map semantic column names to indices."""
    h_lower = [h.lower() for h in headers]
    col_idx = {}

    # Subject ID (may be absent in real PEMF — extracted from Clip column later)
    for kw in ("subject", "participant", "subj"):
        for i, h in enumerate(h_lower):
            if kw in h and "subject" not in col_idx:
                col_idx["subject"] = i
                break

    # Clip/condition type — "kind" first (real PEMF), then generic keywords
    for kw in ("kind", "condition", "type", "stimulus", "expression", "category"):
        for i, h in enumerate(h_lower):
            if kw in h and "clip_type" not in col_idx:
                col_idx["clip_type"] = i
                break

    # Clip ID / label — avoid overlapping with clip_type or subject columns
    occupied = {col_idx.get("subject"), col_idx.get("clip_type")} - {None}
    for kw in ("clip", "video", "label", "name"):
        for i, h in enumerate(h_lower):
            if kw in h and "clip_id" not in col_idx and i not in occupied:
                col_idx["clip_id"] = i
                break

    # Pain intensity rating
    for kw in ("pain", "intensity", "rating", "score", "likert"):
        for i, h in enumerate(h_lower):
            if kw in h and "pain_rating" not in col_idx:
                col_idx["pain_rating"] = i
                break

    # AU columns (look for exact AU4, AU6 etc. or "au 4", "au4", "4")
    for au in PEMF_AU_COLUMNS:
        au_lower = au.lower()
        for i, h in enumerate(h_lower):
            # Match AU4, au4, "au 4", "4 (au4)", etc.
            if h == au_lower or h == au_lower.replace("au", "au ") or h.startswith(au_lower):
                col_idx[au] = i
                break
            # Also try bare number: "4" for AU4
            bare = au.replace("AU", "")
            if h == bare:
                col_idx[au] = i
                break

    return col_idx


def _extract_subject_from_clip_id(clip_id_str: str) -> tuple[str, str]:
    """
    Extract subject ID and type suffix from a composite clip ID.

    PEMF uses codes like 'S001A', 'S002L', 'S068P':
      subject = S001, suffix = A (Algometer)

    Returns (subject_id, suffix_letter_or_empty).
    """
    s = clip_id_str.strip()
    if not s:
        return s, ""
    # Pattern: S###<suffix> or just a number
    if s[0].upper() == "S" and len(s) >= 4:
        # Find where digits end
        i = 1
        while i < len(s) and s[i].isdigit():
            i += 1
        subject = s[:i]
        suffix = s[i:].strip()
        return subject, suffix
    return s, ""


def _parse_row(row: tuple, headers: list[str], col_idx: dict, row_idx: int) -> dict | None:
    """Parse a single Excel row into a clip record."""
    def get(key, default=None):
        idx = col_idx.get(key)
        if idx is None or idx >= len(row):
            return default
        return row[idx]

    # --- Subject ID ---
    # If a dedicated subject column exists, use it; otherwise extract from clip_id
    clip_id_raw = get("clip_id", row_idx)
    clip_id = str(clip_id_raw).strip() if clip_id_raw is not None else str(row_idx)

    subject_raw = get("subject")
    if subject_raw is not None:
        s = str(subject_raw).strip()
        subject_id = f"S{int(float(s)):03d}" if s.replace(".", "").isdigit() else s
    else:
        # Real PEMF: extract from composite clip ID (e.g. S001A → S001)
        subject_id, clip_suffix = _extract_subject_from_clip_id(clip_id)
        if not subject_id:
            return None  # Cannot determine subject

    # --- Clip type ---
    clip_type_raw = get("clip_type", "")
    clip_type = str(clip_type_raw).strip().lower() if clip_type_raw else ""

    # If clip_type is empty, infer from composite clip ID suffix
    if not clip_type and subject_raw is None:
        _, suffix = _extract_subject_from_clip_id(clip_id)
        clip_type = CLIP_SUFFIX_MAP.get(suffix.lower(), suffix.lower())

    is_neutral = any(kw in clip_type for kw in NEUTRAL_KEYWORDS)

    # --- Pain intensity ---
    pain_raw = get("pain_rating")
    if is_neutral:
        pain_score_raw = 0.0
    elif pain_raw is not None:
        # Handle European-formatted mean(SD) strings like "5,20 (2,516)"
        pain_score_raw = parse_european_mean(pain_raw)
    else:
        pain_score_raw = None

    # PSPI-equivalent: Likert (0-8) × 2 = 0-16
    pain_score_pspi = round(pain_score_raw * 2.0, 2) if pain_score_raw is not None else (0.0 if is_neutral else None)

    # --- AU intensities (0-6 in real PEMF, or binary 0/1, or "x" markers) ---
    aus = {}
    for au in PEMF_AU_COLUMNS:
        val = get(au)
        if val is not None:
            try:
                aus[au] = int(float(str(val))) if str(val).strip() not in ("", "x", "X") else 1
            except (ValueError, TypeError):
                aus[au] = 1 if str(val).strip().lower() in ("x", "yes", "1") else 0
        else:
            aus[au] = 0

    return {
        "clip_id": clip_id,
        "subject_id": subject_id,
        "clip_type": clip_type,
        "is_neutral": is_neutral,
        "pain_score_raw": pain_score_raw,
        "pain_score_pspi": pain_score_pspi,
        "aus": aus,
    }


def find_clip_folder(pictures_dir: Path, clip_record: dict) -> Path | None:
    """
    Try to find the image folder for a clip record.

    Tries several naming conventions in order:
      1. clip_id directly (e.g. "S001A" for real PEMF)
      2. subject_type combo (e.g. "S001_algometer")
      3. Numeric zero-padded (e.g. "001")
      4. Fuzzy match on subject + type keywords
    """
    clip_id = clip_record["clip_id"]
    subject_id = clip_record["subject_id"]
    clip_type = clip_record["clip_type"]

    candidates = [
        clip_id,
        f"{clip_id:>03}" if clip_id.isdigit() else None,
        f"{subject_id}_{clip_type}",
        f"{subject_id}_{clip_id}",
        f"{subject_id}-{clip_type}",
        f"clip_{clip_id}",
        f"{subject_id}",
    ]

    for name in candidates:
        if name is None:
            continue
        p = pictures_dir / name
        if p.is_dir() and any(p.iterdir()):
            return p

    # Fuzzy: any subdir containing the clip_id (case-insensitive)
    clip_id_lower = clip_id.lower()
    for subdir in sorted(pictures_dir.iterdir()):
        if not subdir.is_dir():
            continue
        if subdir.name.lower() == clip_id_lower:
            return subdir

    # Fuzzy: subdir containing subject_id + type keywords
    for subdir in sorted(pictures_dir.iterdir()):
        if not subdir.is_dir():
            continue
        name_lower = subdir.name.lower()
        if subject_id.lower() in name_lower:
            if clip_type and any(kw in name_lower for kw in clip_type.split()):
                return subdir

    return None


def collect_samples(
    excel_records: list[dict],
    pictures_dir: Path,
) -> tuple[list[dict], dict[str, list[dict]]]:
    """
    Match Excel records to image folders and collect per-frame samples.

    Returns:
        samples: list of frame-level sample dicts
        neutral_frames_by_subject: {subject_id: [frame sample dicts for neutral clips]}
    """
    samples = []
    neutral_frames_by_subject: dict[str, list[dict]] = {}
    missing_clips = 0

    for record in excel_records:
        clip_folder = find_clip_folder(pictures_dir, record)

        if clip_folder is None:
            missing_clips += 1
            if missing_clips <= 5:
                print(f"  WARNING: Could not find image folder for clip_id='{record['clip_id']}' subject='{record['subject_id']}' type='{record['clip_type']}'")
            continue

        # Collect frame images in this clip folder
        frame_files = sorted(
            f for f in clip_folder.iterdir()
            if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
        )

        if not frame_files:
            missing_clips += 1
            continue

        for frame_idx, frame_file in enumerate(frame_files):
            sample = {
                "image_path": str(frame_file),
                "clip_id": record["clip_id"],
                "subject_id": record["subject_id"],
                "clip_type": record["clip_type"],
                "is_neutral": record["is_neutral"],
                "frame_idx": frame_idx,
                "pain_score": record["pain_score_pspi"],
                "aus": record["aus"],
                "reference_filename": "",  # filled in later
            }
            samples.append(sample)

            if record["is_neutral"]:
                neutral_frames_by_subject.setdefault(record["subject_id"], []).append(sample)

    if missing_clips > 5:
        print(f"  ... and {missing_clips - 5} more missing clip folders.")
    if missing_clips:
        print(f"  Total missing clip folders: {missing_clips}")

    return samples, neutral_frames_by_subject


def assign_reference_frames(samples: list[dict], neutral_frames_by_subject: dict[str, list[dict]]) -> None:
    """
    For each pain frame, assign the first neutral frame of the same subject as reference.
    This enables comparative model training (target + reference pair).
    Reference filename is left empty for neutral frames.
    """
    for sample in samples:
        if sample["is_neutral"]:
            continue
        subject_id = sample["subject_id"]
        neutral_frames = neutral_frames_by_subject.get(subject_id, [])
        if neutral_frames:
            # Use the middle frame of the neutral clip as reference (most stable)
            mid = len(neutral_frames) // 2
            ref = neutral_frames[mid]
            # Reference filename will be set during write_split once we know the output name
            sample["_ref_sample"] = ref
        else:
            sample["_ref_sample"] = None


def preprocess_image(image_path: str, target_size: tuple[int, int] = (224, 224)) -> np.ndarray | None:
    """Load and resize an image. Returns None on failure."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        return cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    except Exception:
        return None


def split_by_subject(
    samples: list[dict],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Split samples by subject to avoid data leakage.
    All frames from a subject go into the same split.
    """
    subjects = sorted(set(s["subject_id"] for s in samples))
    rng = random.Random(seed)
    rng.shuffle(subjects)

    n = len(subjects)
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))

    train_subjects = set(subjects[:n_train])
    val_subjects = set(subjects[n_train:n_train + n_val])
    test_subjects = set(subjects[n_train + n_val:])

    if not test_subjects and len(val_subjects) > 1:
        moved = val_subjects.pop()
        test_subjects.add(moved)

    train = [s for s in samples if s["subject_id"] in train_subjects]
    val = [s for s in samples if s["subject_id"] in val_subjects]
    test = [s for s in samples if s["subject_id"] in test_subjects]

    return train, val, test


def make_output_filename(sample: dict) -> str:
    """Build a deterministic output filename for a sample."""
    clip_type_safe = sample["clip_type"].replace(" ", "_").replace("/", "_")[:20]
    return f"{sample['subject_id']}_{clip_type_safe}_{sample['clip_id']}_f{sample['frame_idx']:02d}.jpg"


def write_split(
    samples: list[dict],
    split_name: str,
    output_dir: Path,
    target_size: tuple[int, int] = (224, 224),
    all_samples_filename_map: dict | None = None,
) -> int:
    """
    Preprocess images and write split to disk.

    CSV columns match prepare_dataset.py plus reference_filename for comparative training.
    Returns count of successfully written samples.
    """
    split_dir = output_dir / split_name / "images"
    split_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / split_name / "labels.csv"

    count = 0

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "filename", "subject_id", "sequence_id", "pain_score", "aus", "clip_type", "reference_filename"
        ])

        for sample in samples:
            img = preprocess_image(sample["image_path"], target_size)
            if img is None:
                continue

            filename = make_output_filename(sample)
            out_path = split_dir / filename
            cv2.imwrite(str(out_path), img)

            # Normalize pain score to 0-10 (PSPI 0-16 → ÷16 × 10)
            pain_score = sample["pain_score"]
            if pain_score is not None:
                if pain_score > 10:
                    pain_score = round((pain_score / 16.0) * 10, 2)  # PSPI scale
                elif 0 < pain_score <= 5:
                    pain_score = round((pain_score / 5.0) * 10, 2)   # OPI scale
                # Values already 0-10 (neutral=0, low Likert) pass through unchanged

            aus_str = ";".join(f"{k}={v}" for k, v in sorted(sample["aus"].items()))

            # Reference filename: look up from precomputed map
            ref_filename = ""
            ref_sample = sample.get("_ref_sample")
            if ref_sample is not None and all_samples_filename_map is not None:
                ref_key = id(ref_sample)
                ref_filename = all_samples_filename_map.get(ref_key, "")

            writer.writerow([
                filename,
                sample["subject_id"],
                sample["clip_id"],        # sequence_id = clip_id for PEMF
                pain_score if pain_score is not None else "",
                aus_str,
                sample["clip_type"],
                ref_filename,
            ])
            count += 1

    return count


def build_filename_map(samples: list[dict]) -> dict[int, str]:
    """Build a map from sample object id → output filename."""
    return {id(s): make_output_filename(s) for s in samples}


def print_statistics(samples: list[dict], split_name: str) -> None:
    """Print dataset statistics for a split."""
    n = len(samples)
    subjects = set(s["subject_id"] for s in samples)
    pain_samples = [s for s in samples if not s["is_neutral"]]
    neutral_samples = [s for s in samples if s["is_neutral"]]

    clip_types = Counter(s["clip_type"] for s in samples)

    print(f"\n  {split_name}:")
    print(f"    Total frames:   {n}")
    print(f"    Subjects:       {len(subjects)}")
    print(f"    Pain frames:    {len(pain_samples)}")
    print(f"    Neutral frames: {len(neutral_samples)}")
    print(f"    Clip types:     {dict(clip_types)}")

    scored = [s for s in samples if s["pain_score"] is not None]
    if scored:
        scores = [s["pain_score"] for s in scored]
        print(f"    Pain score (PSPI-equiv) range: {min(scores):.1f} - {max(scores):.1f}")
        print(f"    Pain score mean: {sum(scores)/len(scores):.2f}")

    with_ref = [s for s in samples if s.get("_ref_sample") is not None]
    if pain_samples:
        print(f"    With reference frame: {len(with_ref)}/{len(pain_samples)} pain frames")


def main():
    parser = argparse.ArgumentParser(description="Prepare PEMF dataset for pain model training")
    parser.add_argument("--dataset-root", required=True,
                        help="Path to extracted PEMF dataset root (containing PEMF_Database.xlsx)")
    parser.add_argument("--output-dir", default="training/data/pemf",
                        help="Output directory for train/val/test splits")
    parser.add_argument("--pictures-dir", default=None,
                        help="Path to Pictures directory (auto-detected if omitted)")
    parser.add_argument("--target-size", type=int, default=224,
                        help="Image resize target in pixels (square, default 224)")
    parser.add_argument("--train-ratio", type=float, default=0.70,
                        help="Fraction of subjects for training (default 0.70)")
    parser.add_argument("--val-ratio", type=float, default=0.15,
                        help="Fraction of subjects for validation (default 0.15)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for subject split (default 42)")
    parser.add_argument("--exclude-posed", action="store_true",
                        help="Exclude posed pain clips (train on spontaneous pain only)")
    parser.add_argument("--include-neutral", action="store_true",
                        help="Include neutral frames in output (labelled pain_score=0)")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    target_size = (args.target_size, args.target_size)

    if not dataset_root.exists():
        print(f"ERROR: Dataset root does not exist: {dataset_root}")
        sys.exit(1)

    # Find Excel labels file
    excel_path = find_excel_file(dataset_root)
    if excel_path is None:
        print(f"ERROR: Could not find PEMF_Database.xlsx in {dataset_root}")
        print("Download from: https://osf.io/3hgca/?view_only=12b04cd8164d4a6784c04b8c83bf95fb")
        sys.exit(1)
    print(f"Labels file: {excel_path}")

    # Find pictures directory
    pictures_dir = find_pictures_dir(dataset_root, args.pictures_dir)
    if pictures_dir is None:
        print(f"ERROR: Could not find Pictures directory in {dataset_root}")
        print("Expected: {dataset_root}/Pictures/ (from Pictures.zip in OSF download)")
        sys.exit(1)
    print(f"Pictures dir: {pictures_dir}")

    # Load Excel labels
    excel_records = load_excel_labels(excel_path)
    if not excel_records:
        print("ERROR: No records loaded from Excel. Check column names.")
        sys.exit(1)

    # Optionally filter out posed pain
    if args.exclude_posed:
        before = len(excel_records)
        excel_records = [r for r in excel_records if "posed" not in r["clip_type"].lower()]
        print(f"--exclude-posed: removed {before - len(excel_records)} posed clips")

    # Collect frame-level samples
    print(f"\nCollecting frames from: {pictures_dir}")
    samples, neutral_frames_by_subject = collect_samples(excel_records, pictures_dir)

    if not samples:
        print("ERROR: No samples collected. Check that Pictures directory contains clip subfolders.")
        print("Expected: {pictures_dir}/{clip_id}/frame_01.jpg ... frame_20.jpg")
        sys.exit(1)

    print(f"\nTotal frames collected: {len(samples)}")
    print(f"Subjects: {len(set(s['subject_id'] for s in samples))}")

    # Assign reference (neutral) frames for comparative model support
    assign_reference_frames(samples, neutral_frames_by_subject)
    ref_count = sum(1 for s in samples if not s["is_neutral"] and s.get("_ref_sample") is not None)
    print(f"Pain frames with reference: {ref_count}")

    # Filter neutral frames from output unless requested
    output_samples = samples if args.include_neutral else [s for s in samples if not s["is_neutral"]]
    print(f"Output samples: {len(output_samples)} ({'including' if args.include_neutral else 'excluding'} neutral frames)")

    # Subject-level split
    train, val, test = split_by_subject(output_samples, args.train_ratio, args.val_ratio, args.seed)

    print("\n--- Dataset Statistics ---")
    print_statistics(train, "Train")
    print_statistics(val, "Validation")
    print_statistics(test, "Test")

    # Build filename map for reference resolution
    all_filename_map = build_filename_map(samples)  # includes neutral frames

    # Write splits
    print(f"\nWriting preprocessed data to: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    n_train = write_split(train, "train", output_dir, target_size, all_filename_map)
    n_val = write_split(val, "val", output_dir, target_size, all_filename_map)
    n_test = write_split(test, "test", output_dir, target_size, all_filename_map)

    print(f"\nDone!")
    print(f"  Train: {n_train} images  ({len(set(s['subject_id'] for s in train))} subjects)")
    print(f"  Val:   {n_val} images  ({len(set(s['subject_id'] for s in val))} subjects)")
    print(f"  Test:  {n_test} images  ({len(set(s['subject_id'] for s in test))} subjects)")
    print(f"  Total: {n_train + n_val + n_test} images")
    print(f"\nOutput structure:")
    print(f"  {output_dir}/train/images/  + labels.csv")
    print(f"  {output_dir}/val/images/    + labels.csv")
    print(f"  {output_dir}/test/images/   + labels.csv")
    print(f"\nTrain on this data:")
    print(f"  python training/train.py --data-dir {output_dir} --output-dir models/archive")


if __name__ == "__main__":
    main()
