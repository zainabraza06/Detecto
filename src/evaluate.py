import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, roc_auc_score

from src.model import MultiHeadR3D18
from src.dataset import DetectoVideoDataset, derive_binary_labels


def main(checkpoint: str, val_csv: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    val_df = derive_binary_labels(pd.read_csv(val_csv))
    val_ds = DetectoVideoDataset(val_df, train=False)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    model = MultiHeadR3D18(pretrained=False).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
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

    v_probs, w_probs = np.array(v_probs), np.array(w_probs)
    v_true, w_true = np.array(v_true), np.array(w_true)

    print("=== Violence head ===")
    print(classification_report(v_true, (v_probs >= 0.5).astype(int), target_names=['No Violence', 'Violence']))
    print(f"AUC: {roc_auc_score(v_true, v_probs):.4f}")

    print("=== Weapon head ===")
    print(classification_report(w_true, (w_probs >= 0.5).astype(int), target_names=['No Weapon', 'Weapon']))
    print(f"AUC: {roc_auc_score(w_true, w_probs):.4f}")

    # Head-independence diagnostic
    disagreement = (v_true == 1) & (w_true == 0)
    if disagreement.sum() > 1:
        corr = np.corrcoef(v_probs[disagreement], w_probs[disagreement])[0, 1]
        print(f"\nHead-independence correlation on disagreement cases (n={disagreement.sum()}): {corr:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--val_csv', default='data/detecto_val_split.csv')
    args = parser.parse_args()
    main(args.checkpoint, args.val_csv)
