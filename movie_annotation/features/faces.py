"""Face detection using OpenCV DNN face detector."""

import cv2
import numpy as np

# OpenCV DNN face detector model files (bundled with opencv-contrib or downloadable)
# We use the Caffe model that ships with OpenCV's data directory.
PROTOTXT_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
CAFFEMODEL_URL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

CONFIDENCE_THRESHOLD = 0.5


class FaceDetector:
    """Face detection using OpenCV's DNN SSD face detector."""

    def __init__(self, cache_dir: str = ".cache"):
        import os
        import urllib.request

        os.makedirs(cache_dir, exist_ok=True)
        prototxt_path = os.path.join(cache_dir, "deploy.prototxt")
        model_path = os.path.join(cache_dir, "res10_300x300_ssd_iter_140000.caffemodel")

        if not os.path.exists(prototxt_path):
            urllib.request.urlretrieve(PROTOTXT_URL, prototxt_path)
        if not os.path.exists(model_path):
            urllib.request.urlretrieve(CAFFEMODEL_URL, model_path)

        self.net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

    def extract(self, frame_bgr: np.ndarray) -> dict:
        """Extract face detection features from a single BGR frame.

        Returns dict with keys: n_faces, face_area_frac
        """
        h, w = frame_bgr.shape[:2]
        frame_area = h * w

        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame_bgr, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0),
        )
        self.net.setInput(blob)
        detections = self.net.forward()

        n_faces = 0
        total_face_area = 0.0

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < CONFIDENCE_THRESHOLD:
                continue

            n_faces += 1
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            # Clamp to frame boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            face_area = max(0, x2 - x1) * max(0, y2 - y1)
            total_face_area += face_area

        face_area_frac = float(total_face_area / frame_area) if frame_area > 0 else 0.0

        return {"n_faces": n_faces, "face_area_frac": face_area_frac}
