# Simple Pose Classifier

This project demonstrates how to build a **simple human pose classifier** using [MediaPipe Pose](https://developers.google.com/mediapipe/solutions/vision/pose) for landmark detection and a basic scikit-learn model for classification.  

It was created as a junior-level project to showcase core skills in:
- Working with datasets
- Feature extraction from raw landmarks
- Training and evaluating ML models
- Saving and loading trained models
- Running a demo script for prediction

---

## 🚀 Features
- Uses MediaPipe Pose to extract 33 body landmarks.
- Trains a Logistic Regression classifier to distinguish **sitting vs standing**.
- Synthetic standing examples included (replace with your own dataset if available).
- Includes a ready-to-run demo (`demo.py`) that loads an image, overlays skeleton landmarks, and predicts the pose.
- Clean and simple codebase with comments for easy understanding.

---

## 📂 Project Structure
```
ML/
├── scripts/
│   ├── train_pose_classifier_synthetic.py   # train model with synthetic standing data
│   ├── predict_pose.py                      # predict pose from image/YouTube
│   ├── demo.py                              # run demo on any image
│   └── build_dataset_csv.py                 # helper to create CSV from images
├── pose_utils/                              # reusable helpers
│   ├── pose_processor.py                    # MediaPipe wrapper
│   ├── csv_io.py                            # CSV utilities
│   ├── config.py                            # project config
│   └── model_zoo.py                         # save/load models
├── Data/                                    # dataset folder
└── models/pose_classifier.joblib            # trained model
```

---

## ⚡ Quickstart

### 1. Install requirements
```bash
pip install -r requirements.txt
```

Requirements include:
- mediapipe
- opencv-python
- numpy
- pandas
- scikit-learn
- joblib

### 2. Train the model
```bash
python scripts/train_pose_classifier_synthetic.py
```

### 3. Run the demo
```bash
python scripts/demo.py --image path/to/your_image.jpg
```

You’ll see the skeleton drawn on the image with a prediction: **Sitting (xx.x%)** or **Standing (xx.x%)**.

---

## 📌 Notes
- Replace the synthetic standing examples with your own dataset for real performance.
- This repo is **designed for junior-level portfolios** — the focus is on clarity and showing that you can take a project end-to-end.
- Accuracy with synthetic data is ~99% (demo only).

---

## 🙌 Acknowledgements
- [MediaPipe Pose](https://developers.google.com/mediapipe/solutions/vision/pose) for pose detection.
- Inspired by open pose datasets such as [Kick Detection & Pose Estimation](https://github.com/pachauriyash/Kick-Detection-and-pose-estimation).
