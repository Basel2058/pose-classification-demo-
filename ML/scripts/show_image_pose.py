#!/usr/bin/env python3
"""Displays the pose skeleton for a specific image in a folder.

This is a simple command-line utility to invoke the skeleton viewer
for a single image, identified by its number and parent folder.

Example:
    python show_image_pose.py --folder /path/to/images --num 123
"""

import argparse
from pathlib import Path

# Local application/library specific imports
from pose_utils.viewer import show_image_skeleton

def main():
    """Parses command-line arguments and calls the viewer function."""
    # --- Argument Parsing ---
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", type=Path, required=True,
                    help="The directory containing the image.")
    ap.add_argument("--num", type=int, required=True,
                    help="The number of the image file (without extension).")
    args = ap.parse_args()

    # --- Show Skeleton ---
    show_image_skeleton(args.num, args.folder)

if __name__ == "__main__":
    main()