import argparse
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from src.model import MultiHeadR3D18
from src.dataset import DetectoVideoDataset, derive_binary_labels


def main(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    random.seed(cfg['training']['seed'])
    np.random.seed(cfg['training']['seed'])
    torch.manual_seed(cfg['training']['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    train_df = pd.read_csv(cfg['data']['train_split_csv'])
    val_df = pd.read_csv(cfg['data']['val_split_csv'])
    overlap = set(train_df['path']) & set(val_df['path'])
    assert len(overlap) == 0, f"LEAKAGE: {len(overlap)} overlapping clips -- stop and investigate"
    print(f"Train: {len(train_df)}  Val: {len(val_df)}  Overlap: 0 (verified)")

    train_df = derive_binary_labels(train_df)
    val_df = derive_binary_labels(val_df)

    train_ds = DetectoVideoDataset(train_df, train=True)
    val_ds = DetectoVideoDataset(val_df, train=False)
    train_loader = DataLoader(train_ds, batch_size=cfg['training']['batch_size'], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=cfg['training']['batch_size'], shuffle=False, num_workers=2)

    def pos_weight_for(col):
        pos = col.sum()
        neg = len(col) - pos
        return torch.tensor(neg / max(pos, 1), dtype=torch.float32).to(device)

    v_crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight_for(train_df['violence_label']))
    w_crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight_for(train_df['weapon_label']))

    model = MultiHeadR3D18(pretrained=True).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg['training']['lr'], momentum=cfg['training']['momentum'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    os.makedirs(cfg['output']['checkpoint_dir'], exist_ok=True)
    ckpt_path = os.path.join(cfg['output']['checkpoint_dir'], cfg['output']['checkpoint_name'])

    best_auc = 0.0
    for epoch in range(cfg['training']['epochs']):
        model.train()
        for clips, labels in train_loader:
            clips = clips.to(device)
            v_labels, w_labels = labels[:, 0].to(device), labels[:, 1].to(device)
            optimizer.zero_grad()
            v_logit, w_logit = model(clips)
            loss = v_crit(v_logit, v_labels) + w_crit(w_logit, w_labels)
            loss.backward()
            optimizer.step()

        model.eval()
        v_probs, w_probs, v_true, w_true = [], [], [], []
        with torch.no_grad():
            for clips, labels in val_loader:
                clips = clips.to(device)
                v_logit, w_logit = model(clips)
                v_probs.extend(torch.sigmoid(v_logit).cpu().numpy())
                w_probs.extend(torch.sigmoid(w_logit).cpu().numpy())
                v_true.extend(labels[:, 0].numpy())
                w_true.extend(labels[:, 1].numpy())

        v_auc = roc_auc_score(v_true, v_probs)
        w_auc = roc_auc_score(w_true, w_probs)
        avg_auc = (v_auc + w_auc) / 2
        scheduler.step(avg_auc)
        print(f"Epoch {epoch+1}/{cfg['training']['epochs']} | violence_AUC={v_auc:.4f} | weapon_AUC={w_auc:.4f} | avg={avg_auc:.4f}")

        if avg_auc > best_auc:
            best_auc = avg_auc
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> saved best ({ckpt_path})")

    print(f"\nBest combined AUC: {best_auc:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    args = parser.parse_args()
    main(args.config)
