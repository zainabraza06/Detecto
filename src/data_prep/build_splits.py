"""Builds a PERMANENTLY PINNED, deterministic train/val split and saves it
to disk. Never rebuild splits ad hoc from os.listdir() -- an earlier version
of this project did that and it caused silent train/val leakage (a model
scored 96% on data it had partially trained on). Always load from the saved
CSVs produced here."""

import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split

CLASSES = ['Normal', 'Violence', 'Weaponized']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
VIDEO_EXTS = ('.mp4', '.avi', '.mov', '.mkv')


def build(data_dir: str, out_dir: str, seed: int = 42):
    records = []
    for split in ['Train', 'Test']:
        for cls in CLASSES:
            cls_dir = os.path.join(data_dir, split, cls)
            if not os.path.isdir(cls_dir):
                continue
            for f in sorted(os.listdir(cls_dir)):
                if f.lower().endswith(VIDEO_EXTS):
                    records.append({'path': os.path.join(cls_dir, f), 'label': CLASS_TO_IDX[cls]})

    df_all = pd.DataFrame(records).sort_values('path').reset_index(drop=True)
    train_df, val_df = train_test_split(df_all, test_size=0.2, stratify=df_all['label'], random_state=seed)

    overlap = set(train_df['path']) & set(val_df['path'])
    assert len(overlap) == 0, f"LEAKAGE DETECTED: {len(overlap)} overlapping clips"

    os.makedirs(out_dir, exist_ok=True)
    df_all.to_csv(os.path.join(out_dir, 'detecto_full_dataset.csv'), index=False)
    train_df.to_csv(os.path.join(out_dir, 'detecto_train_split.csv'), index=False)
    val_df.to_csv(os.path.join(out_dir, 'detecto_val_split.csv'), index=False)
    print(f"Total: {len(df_all)}  Train: {len(train_df)}  Val: {len(val_df)}  Overlap: 0 (verified)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help='Path to SCVD_converted')
    parser.add_argument('--out_dir', default='data')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    build(args.data_dir, args.out_dir, args.seed)
