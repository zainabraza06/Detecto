import numpy as np
import cv2

FRAME_SIZE = 112
RESIZE_TARGET = 128
KINETICS_MEAN = np.array([0.43216, 0.394666, 0.37645])
KINETICS_STD = np.array([0.22803, 0.22145, 0.216989])


def resize_preserve_aspect(frame, target_size=RESIZE_TARGET):
    h, w = frame.shape[:2]
    if h < w:
        new_h, new_w = target_size, int(w * target_size / h)
    else:
        new_h, new_w = int(h * target_size / w), target_size
    return cv2.resize(frame, (new_w, new_h))


def center_crop(frame, crop_h, crop_w):
    h, w = frame.shape[:2]
    top = max(0, (h - crop_h) // 2)
    left = max(0, (w - crop_w) // 2)
    return frame[top:top + crop_h, left:left + crop_w]


def preprocess_frame(frame, size=FRAME_SIZE):
    """Aspect-ratio-preserving resize + center crop -- avoids the square-squash
    distortion that the SCVD SOTA paper (SSIVD-Net) found hurts CCTV footage."""
    proc = resize_preserve_aspect(frame)
    proc = center_crop(proc, size, size)
    if proc.shape[0] != size or proc.shape[1] != size:
        padded = np.zeros((size, size, 3), dtype=np.uint8)
        ph, pw = min(size, proc.shape[0]), min(size, proc.shape[1])
        padded[:ph, :pw] = proc[:ph, :pw]
        proc = padded
    return proc


def normalize_clip(frames: np.ndarray) -> np.ndarray:
    clip = frames.astype(np.float32) / 255.0
    return (clip - KINETICS_MEAN) / KINETICS_STD
