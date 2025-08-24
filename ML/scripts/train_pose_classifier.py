#!/usr/bin/env python3
"""Trains and evaluates multiple classifiers for pose detection.

This script loads pose landmark data from CSV files, trains several common
machine learning models (Logistic Regression, SVM, Random Forest, KNN),
evaluates them using stratified cross-validation, and saves the best-performing
model to a file.

Example:
    python train_pose_classifier.py
"""

import math
import ssl
import certifi
import numpy as np
import pandas as pd
from pathlib import Path

# Scikit-learn imports for model selection, pipelines, and classifiers.
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Local application/library specific imports
from pose_utils.config import DATA_BY_CLASS, MODEL_PATH
from pose_utils.model_zoo import save as save_model

# --- SSL Certificate Workaround ---
ssl._create_default_https_context = (
    lambda *a, **kw: ssl.create_default_context(cafile=certifi.where()))

# The number of landmarks produced by MediaPipe Pose.
N_LM = 33

def csv_to_xy(csv_path: Path, label: int):
    """Loads landmark data from a CSV and converts it to a feature matrix (X) and label vector (y).

    Args:
        csv_path: The path to the input CSV file.
        label: The integer class label to assign to all rows from this CSV.

    Returns:
        A tuple (X, y) containing the feature matrix and label vector as numpy arrays.
    """
    df = pd.read_csv(csv_path)
    # Group landmarks by image number.
    imgs = {}
    for r in df.itertuples(index=False):
        # Skip rows that are not associated with a landmark (e.g., placeholder rows).
        if math.isnan(r.landmark_id):
            continue
        imgs.setdefault(r.image_num, {})[int(r.landmark_id)] = (r.x, r.y, r.z)

    X, y = [], []
    # Convert each image's landmarks into a flat feature vector.
    for img_id, pts in imgs.items():
        row = []
        for lid in range(N_LM):
            # Get landmark coordinates or (0,0,0) if not present.
            row.extend(pts.get(lid, (0, 0, 0)))
        X.append(row)
        y.append(label)

    return np.asarray(X), np.asarray(y)


# --- Data Loading ---
# Load data for each class and assign the correct label.
Xs, ys = csv_to_xy(DATA_BY_CLASS[0]["csv"], 0)
Xt, yt = csv_to_xy(DATA_BY_CLASS[1]["csv"], 1)

# Combine the data from all classes into a single dataset.
X = np.vstack([Xs, Xt])
y = np.hstack([ys, yt])
print(f"[INFO] Dataset constructed with shape: {X.shape}")

# --- Model Definitions ---
# A dictionary of classifiers to be evaluated.
MODELS = {
    "LogReg": LogisticRegression(max_iter=2000),
    "SVM": SVC(kernel="rbf", probability=True),
    "RF": RandomForestClassifier(n_estimators=300, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

# --- Model Evaluation ---
# Use 5-fold stratified cross-validation to ensure class distribution is
# maintained in each fold, which is crucial for imbalanced datasets.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_name, best_score, best_pipe = None, 0, None

print("[INFO] Starting cross-validation for all models...")
for name, clf in MODELS.items():
    # Create a pipeline that first scales the data then runs the classifier.
    # StandardScaler is essential for distance-based algorithms like SVM and KNN.
    pipe = make_pipeline(StandardScaler(), clf)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
    mean, std = scores.mean(), scores.std()
    print(f"{name:<6} | Accuracy: {mean:.3f} ± {std:.3f}")

    # Keep track of the best performing model.
    if mean > best_score:
        best_name, best_score, best_pipe = name, mean, pipe

# --- Final Model Training and Saving ---
print(f"[INFO] Best model found: {best_name} (Accuracy={best_score:.3f})")
print("[INFO] Fitting the best model on the entire dataset...")
best_pipe.fit(X, y)
save_model(best_pipe, MODEL_PATH)