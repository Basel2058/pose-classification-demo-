"""
Pose Processing Module

This module provides the `PoseProcessor` class, a high-level wrapper around
the MediaPipe Pose solution. It simplifies the process of detecting pose
landmarks from images and provides helper methods for visualization.
"""

import cv2
import mediapipe as mp
import numpy as np
from .config import CONF_THRES, MODEL_COMPLEX

class PoseProcessor:
    """
    A thin wrapper over MediaPipe Pose for easy landmark detection and processing.

    This class handles the initialization of the MediaPipe Pose model and provides
    methods to extract landmarks from images, filter them by visibility, and
    draw the resulting skeleton. It also functions as a context manager to
    ensure proper resource cleanup.
    """

    # MediaPipe Pose solution and its connection graph for drawing skeletons.
    _MP_POSE = mp.solutions.pose
    _CONN = list(_MP_POSE.POSE_CONNECTIONS)

    def __init__(self, model_complexity: int = MODEL_COMPLEX):
        """
        Initializes the PoseProcessor with a MediaPipe Pose instance.

        Args:
            model_complexity (int, optional): The complexity of the pose model.
                                              Defaults to MODEL_COMPLEX from config.
        """
        self.pose = self._MP_POSE.Pose(model_complexity=model_complexity)

    # ---------- Landmarks ----------
    def landmarks(self, bgr_img):
        """
        Detects all pose landmarks in a BGR image.

        Args:
            bgr_img: The input image in BGR format (as read by OpenCV).

        Returns:
            The `pose_landmarks` object from the MediaPipe result, or None if
            no pose is detected.
        """
        # MediaPipe requires RGB images, so convert from BGR.
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        # Process the image to detect the pose.
        res = self.pose.process(rgb_img)
        return res.pose_landmarks

    def visible_landmarks(self, bgr_img, thr: float = CONF_THRES):
        """
        Detects pose landmarks and filters them by visibility.

        This method returns a dictionary of landmark points in image coordinates
        (pixels) for landmarks that meet the visibility threshold.

        Args:
            bgr_img: The input image in BGR format.
            thr (float, optional): The minimum visibility score.
                                   Defaults to CONF_THRES from config.

        Returns:
            dict: A dictionary mapping landmark index to its (x, y) pixel coordinates.
        """
        pts = {}
        lms = self.landmarks(bgr_img)
        if not lms:
            return pts

        h, w = bgr_img.shape[:2]
        for i, lm in enumerate(lms.landmark):
            # The 'getattr' ensures this check works even if visibility is not reported.
            if getattr(lm, "visibility", 1.0) >= thr:
                # Convert normalized coordinates (0.0-1.0) to pixel coordinates.
                pts[i] = (int(lm.x * w), int(lm.y * h))
        return pts

    # ---------- Drawing ----------
    @classmethod
    def draw_skeleton(cls, shape: tuple, pts: dict):
        """
        Draws a pose skeleton on a black canvas.

        Args:
            shape (tuple): The (height, width) of the canvas to create.
            pts (dict): A dictionary of landmark points as {index: (x, y)}.

        Returns:
            np.ndarray: A new BGR image (numpy array) with the skeleton drawn.
        """
        h, w = shape
        canvas = np.zeros((h, w, 3), np.uint8)
        # Draw lines connecting the landmarks.
        for a, b in cls._CONN:
            if a in pts and b in pts:
                cv2.line(canvas, pts[a], pts[b], (0, 255, 0), 2)
        # Draw circles for each landmark point.
        for x, y in pts.values():
            cv2.circle(canvas, (x, y), 4, (0, 255, 0), -1)
        return canvas

    # ---------- Context Manager ----------
    def __enter__(self):
        """Allows the class to be used in a 'with' statement."""
        return self

    def __exit__(self, exc_type, exc, tb):
        """Ensures the MediaPipe Pose resources are released."""
        self.pose.close()