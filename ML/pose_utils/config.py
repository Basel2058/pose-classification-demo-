"""
Configuration File

This file centralizes all static configuration variables for the project,
making it easier to manage and modify default settings. It includes
thresholds for model confidence, file extensions, dataset paths, and model
storage locations.
"""

from pathlib import Path

# ---------- Core Thresholds / Defaults ----------
# Defines the minimum visibility score for a landmark to be considered present.
# Landmarks detected with a confidence score below this value will be ignored.
CONF_THRES = 0.90

# Default image extension used when reading and writing image files.
IMAGE_EXT = ".jpg"

# Specifies the complexity of the MediaPipe Pose model.
# 0: Lite, 1: Full, 2: Heavy. Higher values increase accuracy but also
# increase latency and resource consumption.
MODEL_COMPLEX = 1

# ---------- Default Dataset Folders -------------
# This dictionary maps class labels to their respective data folders and
# CSV output files. It is used during data processing and model training
# to locate raw data and specify output destinations.

#
# Instead of hard‑coding absolute paths tied to a particular machine,
# all paths below are relative to the project root.  When running
# commands from the top‑level `ML` directory, these paths will
# resolve correctly.  Adjust the folder locations as necessary if you
# decide to organise your raw images differently.

DATA_BY_CLASS = {
    # Class 0: sitting
    0: {
        # Folder containing raw images for the 'sitting' class.  When
        # collecting your own data, place the images inside this
        # directory.  The build_dataset scripts will read from this
        # folder and write landmarks to the CSV file defined below.
        "folder": Path("Data/sitting"),
        # CSV file where processed landmarks for the 'sitting' class will
        # be stored.  By default this points at the existing
        # landmarksSitting.csv file included in this repository.
        "csv": Path("pose_utils/Data/raw/v5/csv/landmarksSitting.csv")
    },
    # Class 1: standing
    1: {
        # Folder containing raw images for the 'standing' class.  You
        # should collect your own standing images and place them in
        # this folder if you wish to train a real classifier.  For
        # demonstration purposes this folder may be empty.
        "folder": Path("Data/standing"),
        # CSV file where processed landmarks for the 'standing' class will
        # be stored.  If you run the build_dataset_by_class script with
        # --class_id 1, it will write to this location.
        "csv": Path("pose_utils/Data/raw/v5/csv/landmarksStanding.csv")
    }
}

# ---------- Default Model Path ------------------
# Defines the default file path for saving and loading the trained
# pose classification model.  Models will be saved inside the
# `models` directory relative to the project root.
MODEL_PATH = Path("models/pose_classifier.joblib")

#----------- Data Pre-processing ------------
# Specifies the input CSV file for pre-processing scripts.  When
# running the filter_visibility script directly, this path points at
# the existing sitting landmarks file by default.  Adjust as needed
# when processing other datasets.
PRE_PROCESS_INPUT = Path("pose_utils/Data/raw/v5/csv/landmarksSitting.csv")

# The default class id used by filter_visibility.py when no class_id
# argument is provided.  0 corresponds to the sitting class and 1 to
# standing.
CLASS_ID = 0