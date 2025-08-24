"""
Image Skeleton Viewer

This module provides a function to visualize the pose skeleton detected
in a single image file. It is useful for debugging and inspecting the
output of the pose detection on individual data points.
"""

import cv2
import numpy as np
from pathlib import Path
import mediapipe as mp
from .config import IMAGE_EXT, CONF_THRES

def show_image_skeleton(img_num: int, folder: Path):
    """
    Reads an image, detects a pose, and displays the skeleton.

    This function is a self-contained utility for visualizing the pose from
    a single image file. It handles file reading, pose detection, drawing,
    and window management. The window remains open until the user presses 'q'.

    Args:
        img_num (int): The number (filename without extension) of the image to process.
        folder (Path): The directory containing the image file.

    Raises:
        FileNotFoundError: If the specified image file cannot be found.
    """
    # Construct the full path to the image.
    path = folder / f"{img_num}{IMAGE_EXT}"
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Image not found at: {path}")

    h, w = img.shape[:2]

    # Initialize MediaPipe Pose for a single, one-off detection.
    # Using a 'with' statement ensures resources are properly closed.
    with mp.solutions.pose.Pose(model_complexity=1) as pose:
        # Convert image to RGB and process to find landmarks.
        res = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # Filter landmarks by visibility and convert to pixel coordinates.
    pts = {}
    if res.pose_landmarks:
        for i, lm in enumerate(res.pose_landmarks.landmark):
            if getattr(lm, "visibility", 1.0) >= CONF_THRES:
                pts[i] = (int(lm.x * w), int(lm.y * h))

    # Create a black canvas to draw the skeleton on.
    canvas = np.zeros((h, w, 3), np.uint8)

    # Draw the skeleton connections.
    for a, b in mp.solutions.pose.POSE_CONNECTIONS:
        if a in pts and b in pts:
            cv2.line(canvas, pts[a], pts[b], (0, 255, 0), 2)

    # Draw the landmark points.
    for (x, y) in pts.values():
        cv2.circle(canvas, (x, y), 4, (0, 255, 0), -1)

    # Display the result in a window.
    cv2.imshow(f"{path.name} – q to quit", canvas)
    # Wait indefinitely for a key press and then close all OpenCV windows.
    cv2.waitKey(0)
    cv2.destroyAllWindows()