#!/usr/bin/env python3
"""
Train a simple sitting/standing pose classifier using synthetic data.

This script demonstrates how to build a pose classifier from a single
class of examples by generating artificial examples for the opposing
class.  It loads the existing `landmarksSitting.csv` file bundled with
this project, constructs feature vectors from the landmark data, and
creates synthetic "standing" samples by scaling the vertical (y and z)
components of each landmark.  A logistic regression classifier is
trained on the combined dataset using a standard machine learning
pipeline (standardisation followed by classification) and evaluated
with stratified 5‑fold cross‑validation.

After training on the full dataset, the model is saved to disk via the
`pose_utils.model_zoo.save` function.  You can later load this model
with the `pose_utils.model_zoo.load` function for inference.  Note
that the synthetic standing data is only meant for demonstration
purposes; for a real project you should collect your own standing
images, extract landmarks with the build_dataset scripts, and then
train the classifier on real data.

Example:

    python train_pose_classifier_synthetic.py
"""

import math
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Ensure the project root is on the Python path so that
# `pose_utils` and other local modules can be imported when this
# script is executed directly.  Without this adjustment, running
# `python scripts/train_pose_classifier_synthetic.py` may result in
# `ModuleNotFoundError` because the `pose_utils` package won't be
# discovered.
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import model saving helper from this package
from pose_utils.model_zoo import save as save_model
from pose_utils.config import MODEL_PATH

N_LM = 33  # Number of pose landmarks produced by MediaPipe

def load_sitting(csv_path: Path) -> np.ndarray:
    """Load landmark CSV and convert to a feature matrix.

    Each image/frame in the CSV is converted into a 1D vector of length
    `N_LM * 3` representing the (x, y, z) coordinates for all 33
    landmarks.  Missing landmarks are filled with zeros.

    Args:
        csv_path: Path to the landmark CSV file.

    Returns:
        A 2‑D numpy array of shape (num_samples, N_LM*3).
    """
    df = pd.read_csv(csv_path)
    # Group by image number; each group corresponds to one sample
    samples = []
    for img_num, g in df.groupby("image_num"):
        # Map landmark id to coordinates
        pts = {int(lid): (x, y, z) for _, lid, x, y, z, vis, opened in g.itertuples(index=False)}
        row = []
        for lid in range(N_LM):
            x, y, z = pts.get(lid, (0.0, 0.0, 0.0))
            row.extend([x, y, z])
        samples.append(row)
    return np.asarray(samples)


def generate_standing(X_sit: np.ndarray, y_scale: float = 0.7, z_scale: float = 0.7) -> np.ndarray:
    """Generate synthetic standing samples by scaling vertical and depth coordinates.

    This function creates a copy of the sitting feature matrix and multiplies
    the y (vertical) and z (depth) coordinates by constant factors.  The
    scaling simulates a change in posture (e.g., a taller stance).

    Args:
        X_sit: The original sitting feature matrix of shape (n_samples, N_LM*3).
        y_scale: Multiplicative factor applied to all y coordinates.
        z_scale: Multiplicative factor applied to all z coordinates.

    Returns:
        A new array of the same shape as `X_sit` containing synthetic standing
        examples.
    """
    X_stand = X_sit.copy()
    # Every third element corresponds to x, y, z.  y is index 1, 4, 7,... and z is 2, 5, 8,...
    for i in range(N_LM):
        base = i * 3
        # Scale y coordinate
        X_stand[:, base + 1] *= y_scale
        # Scale z coordinate
        X_stand[:, base + 2] *= z_scale
    return X_stand


def main():
    # Path to the bundled sitting landmarks file
    csv_path = Path("pose_utils/Data/raw/v5/csv/landmarksSitting.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found at {csv_path}.  You may need to run the build_dataset_by_class script first.")

    # Load sitting data
    X_sit = load_sitting(csv_path)
    y_sit = np.zeros(len(X_sit), dtype=int)  # label 0 for sitting

    # Generate synthetic standing data from sitting data
    X_stand = generate_standing(X_sit, y_scale=0.7, z_scale=0.5)
    y_stand = np.ones(len(X_stand), dtype=int)  # label 1 for standing

    # Combine into single dataset
    X = np.vstack([X_sit, X_stand])
    y = np.hstack([y_sit, y_stand])
    print(f"[INFO] Constructed synthetic dataset with {X.shape[0]} samples and {X.shape[1]} features")

    # Create pipeline: standardise then logistic regression
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))

    # Evaluate with stratified 5‑fold CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    mean, std = scores.mean(), scores.std()
    print(f"[INFO] Cross‑validation accuracy: {mean:.3f} ± {std:.3f}")

    # Fit on full dataset
    clf.fit(X, y)
    print("[INFO] Training complete.  Saving model...")
    save_model(clf, MODEL_PATH)
    print(f"[✓] Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()