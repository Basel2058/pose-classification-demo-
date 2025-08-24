"""
CSV I/O Operations

This module provides utility functions for handling CSV file operations,
specifically for writing pose landmark data. It standardizes the CSV
header and format for consistent data logging.
"""

import csv
from pathlib import Path

# Standard header row for the landmark CSV files.
# "image_num": The identifier for the source image/frame.
# "landmark_id": The index of the landmark (e.g., 0 for nose).
# "x", "y", "z": The 3D coordinates of the landmark.
# "visibility": The confidence score of the landmark detection.
# "opened": A binary flag indicating if the source was successfully processed.
CSV_HEADER = [
    "image_num", "landmark_id",
    "x", "y", "z", "visibility", "opened"
]

def open_writer(path: Path):
    """
    Opens a file for CSV writing and returns the file and writer objects.

    This function ensures the parent directory for the given path exists
    before attempting to create the file. It also writes the predefined
    CSV_HEADER to the new file automatically.

    Args:
        path (Path): The file path where the CSV will be created.

    Returns:
        tuple: A tuple containing the opened file object and the csv.writer instance.
    """
    # Ensure the directory structure exists.
    path.parent.mkdir(parents=True, exist_ok=True)
    # Open the file in write mode with no newline padding.
    f = open(path, "w", newline="")
    # Create a CSV writer and write the header row.
    w = csv.writer(f)
    w.writerow(CSV_HEADER)
    return f, w

def write_landmarks(writer, img_num: int, lms, opened: int, thr: float):
    """
    Writes all detected landmarks for a single image to the CSV file.

    For each landmark that meets the visibility threshold, a new row is
    written. If the image could not be processed (`opened` is 0) or no
    landmarks were found (`lms` is None), a single row is written with
    only the image number and the 'opened' status.

    Args:
        writer: The csv.writer object to use for writing.
        img_num (int): The identifier of the image/frame.
        lms: The pose landmarks object from MediaPipe. Can be None.
        opened (int): A flag indicating if the image was successfully opened (1) or not (0).
        thr (float): The visibility threshold to apply. Landmarks below this are ignored.
    """
    # If the source could not be opened or no landmarks were detected,
    # write a single row indicating the failure.
    if opened == 0 or lms is None:
        writer.writerow([img_num, "", "", "", "", "", opened])
        return

    # Iterate through each detected landmark.
    for idx, lm in enumerate(lms.landmark):
        # Only write landmarks that meet the visibility threshold.
        # The 'getattr' provides a default of 1.0 for visibility if not present.
        if getattr(lm, "visibility", 1.0) < thr:
            continue
        # Write the landmark data as a new row.
        writer.writerow([img_num, idx, lm.x, lm.y, lm.z, lm.visibility, opened])