import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.preprocessing import preprocess_frame, normalize_clip

NUM_FRAMES = 12


class DetectoVideoDataset(Dataset):
    """Loads SCVD-style clips with FIXED-length frame sampling -- this is
    what removes clip duration as an exploitable signal (see docs/METHODOLOGY.md,
    the duration-confound finding: 81.7% accuracy from duration alone)."""

    def __init__(self, df, num_frames: int = NUM_FRAMES, train: bool = True):
        self.df = df.reset_index(drop=True)
        self.num_frames = num_frames
        self.train = train

    def __len__(self):
        return len(self.df)

    def _sample_indices(self, total_frames):
        if total_frames <= 0:
            return [0] * self.num_frames
        base = np.linspace(0, total_frames - 1, self.num_frames)
        if self.train:
            jitter = np.random.uniform(-0.5, 0.5, size=self.num_frames)
            base = np.clip(base + jitter, 0, total_frames - 1)
        return base.astype(int)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        cap = cv2.VideoCapture(row['path'])
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = self._sample_indices(total)

        frames = []
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, frame = cap.read()
            if not ok:
                frame = np.zeros((112, 112, 3), dtype=np.uint8)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = preprocess_frame(frame)
            frames.append(frame)
        cap.release()

        clip = normalize_clip(np.stack(frames, axis=0))
        clip = torch.from_numpy(clip).permute(3, 0, 1, 2).float()
        labels = torch.tensor([row['violence_label'], row['weapon_label']], dtype=torch.float32)
        return clip, labels


def derive_binary_labels(df):
    """Original label: 0=Normal, 1=Violence, 2=Weaponized."""
    df = df.copy()
    df['violence_label'] = (df['label'] >= 1).astype(int)
    df['weapon_label'] = (df['label'] == 2).astype(int)
    return df
