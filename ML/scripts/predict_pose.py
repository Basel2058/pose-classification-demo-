#!/usr/bin/env python3
"""Performs real-time pose classification on an image or YouTube video.

This script loads a pre-trained pose classifier and uses it to predict the
pose (e.g., "Sitting" or "Standing") from a local image file or a YouTube
video stream.

Example (Image):
    python predict_pose.py --image /path/to/your/image.jpg

Example (YouTube):
    python predict_pose.py --url https://www.youtube.com/watch?v=dQw4w9WgXcQ
"""

import argparse
import itertools
import cv2
import numpy as np
import ssl
import certifi
from pathlib import Path

# Local application/library specific imports
from pose_utils.model_zoo import load as load_model
from pose_utils.pose_processor import PoseProcessor
from pose_utils.youtube import fetch as fetch_yt
from pose_utils.config import CONF_THRES

# --- SSL Certificate Workaround ---
# Patches the default SSL context to use certifi's certificate bundle.
# This prevents potential SSL errors when fetching YouTube content.
ssl._create_default_https_context = (
    lambda *a, **kw: ssl.create_default_context(cafile=certifi.where()))

# --- Constants ---
LABELS = {0: "Sitting", 1: "Standing"}

def to_feature_vec(pts, w, h, n_lm=33):
    """Converts landmark points to a normalized feature vector.

    Args:
        pts: A dictionary of visible landmark points {id: (x, y)}.
        w: The width of the image frame.
        h: The height of the image frame.
        n_lm: The total number of landmarks in the model (default is 33 for MediaPipe Pose).

    Returns:
        A numpy array of shape (1, n_lm * 3) representing the normalized
        (x, y, z) coordinates for all possible landmarks.
    """
    vec = []
    for lid in range(n_lm):
        if lid in pts:
            x, y = pts[lid]
            # Normalize coordinates and assume z=0 for 2D inputs.
            vec.extend([x / w, y / h, 0])
        else:
            # If a landmark is not visible, pad with zeros.
            vec.extend([0, 0, 0])
    return np.asarray(vec).reshape(1, -1)

def frame_gen(args):
    """Yields frames from either a YouTube URL or a local image.

    Args:
        args: The command-line arguments object.

    Yields:
        Image frames (numpy arrays) from the specified source.
    """
    if args.url:
        # If a URL is provided, fetch the video and capture frames.
        path = fetch_yt(args.url, args.max_height)
        cap = cv2.VideoCapture(str(path))
        while True:
            ok, f = cap.read()
            if not ok:
                break
            yield f
        cap.release()
    else:
        # If an image is provided, read it and yield it indefinitely.
        img = cv2.imread(str(args.image))
        if img is None:
            raise FileNotFoundError(args.image)
        yield from itertools.repeat(img)

def main():
    """Main execution block for parsing arguments and running prediction."""
    # --- Argument Parsing ---
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--image", type=Path, help="Path to the input image file.")
    g.add_argument("--url", help="URL of the YouTube video to process.")
    ap.add_argument("--max_height", type=int, default=720,
                    help="Maximum height for YouTube video download.")
    args = ap.parse_args()

    # --- Model and Processor Initialization ---
    model = load_model()
    with PoseProcessor() as pp:
        # --- Main Prediction Loop ---
        for frame in frame_gen(args):
            # Detect visible landmarks in the current frame.
            pts = pp.visible_landmarks(frame, CONF_THRES)

            if pts:
                # If landmarks are found, create a feature vector.
                fv = to_feature_vec(pts, frame.shape[1], frame.shape[0])
                # Predict probabilities for each class.
                proba = model.predict_proba(fv)[0]
                pred = int(proba.argmax())
                # Format the text for display.
                txt = f"{LABELS[pred]} ({proba[pred]*100:.1f}%)"
            else:
                txt = "No pose"

            # --- Visualization ---
            # Create a semi-transparent background for the text.
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(frame, (10, 10), (10 + tw + 20, 10 + th + 20), (50, 200, 50), -1)
            # Draw the prediction text on the frame.
            cv2.putText(frame, txt, (20, 10 + th + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

            cv2.imshow("Pose Prediction – q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()