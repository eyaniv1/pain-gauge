"""
UNBC-McMaster Shoulder Pain Expression Dataset Loader

Reads the UNBC-McMaster dataset directory structure, extracts frames with
AU labels and OPI pain scores, preprocesses images, and outputs a standard
train/val/test split ready for model training.

Dataset structure expected:
  {dataset_root}/
    Images/
      {subject_id}/
        {sequence_id}/
          {frame}.png
    Frame_Labels/
      FACS/
        {subject_id}/
          {sequence_id}/
            {frame}_facs.txt
    Frame_Labels/
      OPR/  (or PSPI/)
        {subject_id}/
          {sequence_id}/
            {frame}_opr.txt  (or _pspi.txt)

Usage:
  python prepare_dataset.py --dataset-root /path/to/UNBC --output-dir training/data/unbc
"""

import argparse
import csv
import os
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


def find_image_dirs(dataset_root: Path) -> Path | None:
    """Find the Images directory in the dataset."""
    for name in ("Images", "images", "Image"):
        candidate = dataset_root / name
        if candidate.is_dir():
            return candidate
    return None


def find_label_dirs(dataset_root: Path) -> dict:
    """Find FACS and pain score label directories."""
    labels = {}
    frame_labels = dataset_root / "Frame_Labels"
    if not frame_labels.exists():
        frame_labels = dataset_root

    for facs_name in ("FACS", "facs"):
        candidate = frame_labels / facs_name
        if candidate.is_dir():
            labels["facs"] = candidate
            break

    for pain_name in ("OPR", "PSPI", "OPI", "opr", "pspi"):
        candidate = frame_labels / pain_name
        if candidate.is_dir():
            labels["pain"] = candidate
            break

    return labels


def parse_facs_file(facs_path: Path) -> dict[str, float]:
    """Parse a FACS label file into AU name → intensity dict."""
    aus = {}
    try:
        with open(facs_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    au_name = f"AU{int(float(parts[0]))}"
                    intensity = float(parts[1])
                    aus[au_name] = intensity
    except Exception:
        pass
    return aus


def parse_pain_score(pain_path: Path) -> float | None:
    """Parse a pain score file (OPI/PSPI). Returns score or None."""
    try:
        with open(pain_path) as f:
            text = f.read().strip()
            if text:
                return float(text)
    except Exception:
        pass
    return None


def collect_samples(dataset_root: Path) -> list[dict]:
    """
    Walk the dataset and collect all frame samples with labels.

    Returns list of dicts with keys:
        image_path, subject_id, sequence_id, frame_id, aus, pain_score
    """
    images_dir = find_image_dirs(dataset_root)
    if images_dir is None:
        print(f"ERROR: Could not find Images directory in {dataset_root}")
        return []

    label_dirs = find_label_dirs(dataset_root)
    print(f"Images dir: {images_dir}")
    print(f"Label dirs: {label_dirs}")

    samples = []

    # Walk subjects
    for subject_dir in sorted(images_dir.iterdir()):
        if not subject_dir.is_dir():
            continue
        subject_id = subject_dir.name

        # Walk sequences
        for seq_dir in sorted(subject_dir.iterdir()):
            if not seq_dir.is_dir():
                continue
            sequence_id = seq_dir.name

            # Walk frames
            for img_file in sorted(seq_dir.iterdir()):
                if img_file.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp"):
                    continue

                frame_id = img_file.stem

                # Load AU labels
                aus = {}
                if "facs" in label_dirs:
                    facs_path = label_dirs["facs"] / subject_id / sequence_id / f"{frame_id}_facs.txt"
                    if facs_path.exists():
                        aus = parse_facs_file(facs_path)

                # Load pain score
                pain_score = None
                if "pain" in label_dirs:
                    for suffix in ("_opr.txt", "_pspi.txt", "_opi.txt", ".txt"):
                        pain_path = label_dirs["pain"] / subject_id / sequence_id / f"{frame_id}{suffix}"
                        if pain_path.exists():
                            pain_score = parse_pain_score(pain_path)
                            break

                samples.append({
                    "image_path": str(img_file),
                    "subject_id": subject_id,
                    "sequence_id": sequence_id,
                    "frame_id": frame_id,
                    "aus": aus,
                    "pain_score": pain_score,
                })

    return samples


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
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Split samples by subject (not randomly) to avoid data leakage.
    All frames from a subject go into the same split.
    """
    subjects = sorted(set(s["subject_id"] for s in samples))
    random.shuffle(subjects)

    n = len(subjects)
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))

    train_subjects = set(subjects[:n_train])
    val_subjects = set(subjects[n_train:n_train + n_val])
    test_subjects = set(subjects[n_train + n_val:])

    # If test is empty, move last val subject to test
    if not test_subjects and len(val_subjects) > 1:
        moved = val_subjects.pop()
        test_subjects.add(moved)

    train = [s for s in samples if s["subject_id"] in train_subjects]
    val = [s for s in samples if s["subject_id"] in val_subjects]
    test = [s for s in samples if s["subject_id"] in test_subjects]

    return train, val, test


def write_split(
    samples: list[dict],
    split_name: str,
    output_dir: Path,
    target_size: tuple[int, int] = (224, 224),
) -> int:
    """Preprocess images and write split to disk. Returns count of successful samples."""
    split_dir = output_dir / split_name / "images"
    split_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / split_name / "labels.csv"
    count = 0

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "subject_id", "sequence_id", "pain_score", "aus"])

        for sample in samples:
            img = preprocess_image(sample["image_path"], target_size)
            if img is None:
                continue

            filename = f"{sample['subject_id']}_{sample['sequence_id']}_{sample['frame_id']}.jpg"
            out_path = split_dir / filename
            cv2.imwrite(str(out_path), img)

            # Normalize pain score to 0-10 (UNBC OPI is 0-5, PSPI is 0-16)
            pain_score = sample["pain_score"]
            if pain_score is not None:
                # Detect scale and normalize
                if pain_score > 10:
                    pain_score = round((pain_score / 16.0) * 10, 2)  # PSPI scale
                elif pain_score <= 5:
                    pain_score = round((pain_score / 5.0) * 10, 2)  # OPI scale

            aus_str = ";".join(f"{k}={v}" for k, v in sorted(sample["aus"].items()))

            writer.writerow([
                filename,
                sample["subject_id"],
                sample["sequence_id"],
                pain_score if pain_score is not None else "",
                aus_str,
            ])
            count += 1

    return count


def print_statistics(samples: list[dict], split_name: str):
    """Print dataset statistics for a split."""
    n = len(samples)
    subjects = set(s["subject_id"] for s in samples)
    with_pain = [s for s in samples if s["pain_score"] is not None]
    with_aus = [s for s in samples if s["aus"]]

    print(f"\n  {split_name}:")
    print(f"    Samples: {n}")
    print(f"    Subjects: {len(subjects)}")
    print(f"    With pain scores: {len(with_pain)}")
    print(f"    With AU labels: {len(with_aus)}")

    if with_pain:
        scores = [s["pain_score"] for s in with_pain]
        print(f"    Pain score range: {min(scores):.1f} - {max(scores):.1f}")
        print(f"    Pain score mean: {sum(scores)/len(scores):.2f}")

        # Distribution buckets
        buckets = Counter()
        for score in scores:
            if score <= 5:
                buckets["0-5 (OPI)"] += 1
            else:
                buckets["5-16 (PSPI)"] += 1
        for bucket, count in sorted(buckets.items()):
            print(f"      {bucket}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Prepare UNBC-McMaster dataset for training")
    parser.add_argument("--dataset-root", required=True, help="Path to UNBC-McMaster dataset root")
    parser.add_argument("--output-dir", default="training/data/unbc", help="Output directory")
    parser.add_argument("--target-size", type=int, default=224, help="Image resize target (square)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")
    args = parser.parse_args()

    random.seed(args.seed)
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    target_size = (args.target_size, args.target_size)

    if not dataset_root.exists():
        print(f"ERROR: Dataset root does not exist: {dataset_root}")
        sys.exit(1)

    print(f"Loading UNBC-McMaster dataset from: {dataset_root}")
    samples = collect_samples(dataset_root)

    if not samples:
        print("ERROR: No samples found. Check dataset directory structure.")
        print("Expected: {root}/Images/{subject}/{sequence}/{frame}.png")
        sys.exit(1)

    print(f"\nTotal samples found: {len(samples)}")
    print(f"Subjects: {len(set(s['subject_id'] for s in samples))}")

    # Split by subject
    train, val, test = split_by_subject(samples)

    print("\n--- Dataset Statistics ---")
    print_statistics(train, "Train")
    print_statistics(val, "Validation")
    print_statistics(test, "Test")

    # Write splits
    print(f"\nWriting preprocessed data to: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    n_train = write_split(train, "train", output_dir, target_size)
    n_val = write_split(val, "val", output_dir, target_size)
    n_test = write_split(test, "test", output_dir, target_size)

    print(f"\nDone!")
    print(f"  Train: {n_train} images")
    print(f"  Val:   {n_val} images")
    print(f"  Test:  {n_test} images")
    print(f"  Total: {n_train + n_val + n_test} images")


if __name__ == "__main__":
    main()
