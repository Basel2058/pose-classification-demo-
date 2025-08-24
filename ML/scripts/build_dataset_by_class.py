#!/usr/bin/env python3
"""Builds a pose landmark dataset from images for a specific class.

This script processes a directory of images corresponding to a predefined class
(e.g., "sitting" or "standing"), extracts pose landmarks using MediaPipe, and
saves them to a designated CSV file. The class information is pulled from the
central configuration file.

Example:
    python build_dataset_by_class.py --class_id 0 --show 12
"""

import argparse
import cv2
from pathlib import Path

# Local application/library specific imports
from pose_utils.config import DATA_BY_CLASS, CONF_THRES, IMAGE_EXT
from pose_utils.pose_processor import PoseProcessor
from pose_utils.csv_io import open_writer, write_landmarks
from pose_utils.viewer import show_image_skeleton

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
    ap.add_argument("--class_id", type=int, required=True, choices=[0, 1],
                    help="The class ID to process (e.g., 0 for sitting, 1 for standing).")
    ap.add_argument("--show", type=int,
                    help="Optional: Image number to preview with its skeleton after extraction.")
    args = ap.parse_args()

    # --- Setup Paths and Directories ---
    # Retrieve configuration (source folder, output CSV) for the specified class.
    cfg = DATA_BY_CLASS[args.class_id]
    src, out = cfg["folder"], cfg["csv"]
    # Ensure source and output directories exist.
    src.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)

    # --- Landmark Extraction ---
    csv_f, writer = open_writer(out)
    with PoseProcessor() as pp:
        image_files = sorted_imgs(src)
        for idx, p in enumerate(image_files, 1):
            img = cv2.imread(str(p))
            opened = int(img is not None)
            lms = pp.landmarks(img) if opened else None

            # Write the extracted landmarks to the CSV file.
            write_landmarks(writer, int(p.stem), lms, opened, CONF_THRES)

            # Log progress every 100 images.
            if idx % 100 == 0:
                print(f"[{idx}/{len(image_files)}] {p.name}")

    csv_f.close()
    print(f"[INFO] Landmark extraction complete. Data written to → {out}")

    # --- Optional Visualization ---
    if args.show is not None:
        print(f"[INFO] Displaying skeleton for image number: {args.show}")
        show_image_skeleton(args.show, src)

if __name__ == "__main__":
    main()