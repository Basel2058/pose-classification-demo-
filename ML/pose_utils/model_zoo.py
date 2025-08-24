"""
Model Persistence Utilities

This module, often referred to as a "model zoo," provides helper functions
for saving and loading machine learning models. It uses `joblib` for
efficient serialization of Python objects, which is particularly effective
for models from libraries like scikit-learn.
"""

import joblib
from pathlib import Path
from .config import MODEL_PATH

def save(model, path: Path = MODEL_PATH):
    """
    Serializes and saves a model to the specified path.

    Ensures the target directory exists before saving.

    Args:
        model: The machine learning model object to save.
        path (Path, optional): The file path to save the model to.
                               Defaults to MODEL_PATH from config.
    """
    # Ensure the parent directory exists.
    path.parent.mkdir(parents=True, exist_ok=True)
    # Dump the model object to the file using joblib.
    joblib.dump(model, path)
    print(f"[INFO] model saved → {path}")

def load(path: Path = MODEL_PATH):
    """
    Loads a model from the specified path.

    Args:
        path (Path, optional): The file path to load the model from.
                               Defaults to MODEL_PATH from config.

    Returns:
        The deserialized model object.

    Raises:
        FileNotFoundError: If the model file does not exist at the given path.
    """
    if not path.exists():
        raise FileNotFoundError(f"model file {path} not found")
    # Load the model from the file using joblib.
    return joblib.load(path)