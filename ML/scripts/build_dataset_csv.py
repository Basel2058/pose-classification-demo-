#!/usr/bin/env python3
"""Builds a pose landmark dataset from a generic folder of images.

This script processes all images in a specified source directory, extracts
pose landmarks using MediaPipe, and saves them to a specified CSV file.
It is a generic version of `build_dataset_by_class.py` that does not rely
on the predefined config paths.

Example:
    python build_dataset_csv.py --src /path/to/images --out /path/to/output.csv
"""

import argparse
import cv2
from pathlib import Path

# Local application/library specific imports
from pose_utils.pose_processor import PoseProcessor
from pose_utils.csv_io import open_writer, write_landmarks
from pose_utils.config import CONF_THRES, IMAGE_EXT

def sorted_imgs(folder: Path):
    """Returns a numerically sorted list of image paths in a folder.

    Filters for files with the specified image extension and a purely
    numeric stem (e.g., '123.jpg').

    Args:
        folder: The Path object of the directory to search.

    Returns:
        A list of Path objects sorted by their integer stem.
    """
    return sorted(
        [p for p in folder.glob(f"*{IMAGE_EXT}") if p.stem.isdigit()],
        key=lambda p: int(p.stem)
    )

def main():
    """Parses arguments, processes images, and writes landmarks to CSV."""
    # --- Argument Parsing ---
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True,
                    help="Path to the source directory of images.")
    ap.add_argument("--out", type=Path, required=True,
                    help="Path to the output CSV file.")
    args = ap.parse_args()

    # --- Landmark Extraction ---
    csv_f, writer = open_writer(args.out)
    with PoseProcessor() as pp:
        image_files = sorted_imgs(args.src)
        for idx, p in enumerate(image_files, 1):
            # Read image; `opened` is a flag for successful read.
            img = cv2.imread(str(p))
            opened = int(img is not None)

            # Process image only if it was successfully opened.
            lms = pp.landmarks(img) if opened else None

            # Write the extracted landmarks (or lack thereof) to the CSV.
            write_landmarks(writer, int(p.stem), lms, opened, CONF_THRES)

            # Log progress every 100 images.
            if idx % 100 == 0:
                print(f"[{idx}/{len(image_files)}] {p.name}")

    csv_f.close()
    print(f"[INFO] CSV saved → {args.out}")

if __name__ == "__main__":
    main()