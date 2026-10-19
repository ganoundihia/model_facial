"""Face detection (YuNet) and alignment matching the FER2013 framing."""
import os

import cv2
import numpy as np

YUNET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "face_detection_yunet_2023mar.onnx")

# Where FER2013 faces sit in their 48x48 image (median of YuNet landmarks on 2,800 FER2013 train images),
# in fractions of the image side.
FER_EYE_CENTER = (0.493, 0.387)
FER_EYE_DIST = 0.352
FER_EYE_MOUTH = 0.362  # vertical distance eye center -> mouth center

DETECT_MAX_SIDE = 1280  # large photos are downscaled for detection (speed), landmarks rescaled back


class Face:
    def __init__(self, box, landmarks, score):
        self.box = box              # x, y, w, h in original image pixels
        self.landmarks = landmarks  # 5x2: right eye, left eye, nose, right mouth corner, left mouth corner
        self.score = score

    @property
    def area(self):
        return self.box[2] * self.box[3]


class FaceDetector:
    def __init__(self, score_threshold=0.7):
        self.net = cv2.FaceDetectorYN.create(YUNET_PATH, "", (320, 320), score_threshold, 0.3, 5000)

    def detect(self, bgr):
        """Return faces sorted from largest to smallest."""
        h, w = bgr.shape[:2]
        s = min(1.0, DETECT_MAX_SIDE / max(h, w))
        img = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else bgr
        self.net.setInputSize((img.shape[1], img.shape[0]))
        _, raw = self.net.detect(img)
        if raw is None:
            return []
        faces = [Face(r[0:4] / s, r[4:14].reshape(5, 2) / s, float(r[14])) for r in raw]
        return sorted(faces, key=lambda f: -f.area)


def align_face(img, face, size=144):
    """Rotate/scale/crop so the eyes land where they are in FER2013 images. Works on gray or color images."""
    re, le, _, rm, lm = face.landmarks
    eye_c = (re + le) / 2
    mouth_c = (rm + lm) / 2
    angle = np.degrees(np.arctan2(le[1] - re[1], le[0] - re[0]))
    # Face size from both the eye distance and the eye-mouth distance: robust to a head turned sideways
    eye_dist = np.linalg.norm(le - re)
    eye_mouth = np.linalg.norm(mouth_c - eye_c)
    side = max(eye_dist / FER_EYE_DIST, eye_mouth / FER_EYE_MOUTH)
    scale = size / side
    M = cv2.getRotationMatrix2D((float(eye_c[0]), float(eye_c[1])), angle, scale)
    M[0, 2] += FER_EYE_CENTER[0] * size - eye_c[0]
    M[1, 2] += FER_EYE_CENTER[1] * size - eye_c[1]
    return cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)
