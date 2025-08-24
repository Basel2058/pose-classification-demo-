#!/usr/bin/env python3
"""
Interactive demo for pose detection and classification.

This script loads a trained pose classifier from disk and applies it to a
single image supplied by the user.  It uses MediaPipe (via the
`PoseProcessor` class) to detect body landmarks, converts the visible
landmarks into a feature vector, feeds this vector to the classifier
and displays the predicted class along with its probability.  The
detected skeleton is drawn on top of the original image for
visualisation.

Example usage:

    python demo.py --image path/to/photo.jpg

If the script is unable to detect a pose it will report "No pose".

Note: For the demo to produce meaningful results you should train a
classifier on real sitting and standing examples using
`train_pose_classifier.py` or your own custom dataset.  The
`train_pose_classifier_synthetic.py` script included in this project
generates a simple model based on synthetic data which is sufficient
for demonstration purposes but not production use.
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
import sys

# Ensure the project root is on the Python path so that `pose_utils`
# and other internal modules can be imported when this script is run
# directly.  Without this adjustment, Python may raise
# `ModuleNotFoundError` because it cannot locate the package.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pose_utils.pose_processor import PoseProcessor
from pose_utils.model_zoo import load as load_model
from pose_utils.config import CONF_THRES, MODEL_PATH

LABELS = {0: "Sitting", 1: "Standing"}


def to_feature_vec(pts: dict, w: int, h: int, n_lm: int = 33) -> np.ndarray:
    """Convert a dictionary of landmark points into a normalised feature vector.

    Each landmark is represented by its (x, y) pixel coordinates.  To make
    the classifier invariant to image resolution, the coordinates are
    normalised by the image width and height.  Depth (z) values are
    assumed to be zero since they are not provided when using the
    `visible_landmarks` method.

    Args:
        pts: Mapping from landmark index to (x, y) pixel coordinates.
        w: Image width.
        h: Image height.
        n_lm: Total number of landmarks (default 33).

    Returns:
        A 2‑D array of shape (1, n_lm * 3) ready for prediction.
    """
    vec = []
    for lid in range(n_lm):
        if lid in pts:
            x, y = pts[lid]
            vec.extend([x / w, y / h, 0.0])
        else:
            vec.extend([0.0, 0.0, 0.0])
    return np.asarray(vec).reshape(1, -1)


def overlay_skeleton(img: np.ndarray, pts: dict, connections) -> np.ndarray:
    """Draw a skeleton on a copy of the image given landmark points.

    Args:
        img: Original BGR image.
        pts: Dictionary of visible landmarks {id: (x, y)}.
        connections: List of landmark index pairs defining the skeleton graph.

    Returns:
        Annotated BGR image.
    """
    out = img.copy()
    # Draw bones
    for a, b in connections:
        if a in pts and b in pts:
            cv2.line(out, pts[a], pts[b], (0, 255, 0), 2)
    # Draw joints
    for x, y in pts.values():
        cv2.circle(out, (x, y), 4, (0, 255, 0), -1)
    return out


def main():
    parser = argparse.ArgumentParser(description="Pose detection and classification demo")
    parser.add_argument("--image", type=Path, required=True, help="Path to an input image")
    args = parser.parse_args()

    # Load classifier
    model = load_model(MODEL_PATH)

    # Read image
    img = cv2.imread(str(args.image))
    if img is None:
        raise FileNotFoundError(f"Could not open image: {args.image}")

    with PoseProcessor() as pp:
        pts = pp.visible_landmarks(img, CONF_THRES)
        if pts:
            # Convert to feature vector
            fv = to_feature_vec(pts, img.shape[1], img.shape[0])
            proba = model.predict_proba(fv)[0]
            pred = int(proba.argmax())
            label_text = f"{LABELS[pred]} ({proba[pred] * 100:.1f}%)"

            # Overlay skeleton
            annotated = overlay_skeleton(img, pts, PoseProcessor._CONN)
            # Draw label at top left
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(annotated, (10, 10), (10 + tw + 20, 10 + th + 20), (50, 200, 50), -1)
            cv2.putText(annotated, label_text, (20, 10 + th + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

            # Show the result in a window
            cv2.imshow("Pose Demo – press any key to exit", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("No pose detected in the image.")


if __name__ == "__main__":
    main()