"""Sliding-window live-stream-style demo. Shows Violence % and Weapon %
independently -- no hard-rule fusion (see docs/METHODOLOGY.md for why)."""

import argparse
import cv2
import numpy as np
import torch

from src.model import MultiHeadR3D18
from src.preprocessing import preprocess_frame, normalize_clip

NUM_FRAMES = 12
WINDOW_STRIDE = 4


def main(checkpoint: str, video_path: str, output_path: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultiHeadR3D18(pretrained=False).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    assert out.isOpened(), "VideoWriter failed to open"

    buffer, frame_idx = [], 0
    v_prob, w_prob = 0.0, 0.0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        buffer.append(preprocess_frame(frame_rgb))
        if len(buffer) > NUM_FRAMES:
            buffer.pop(0)

        if len(buffer) == NUM_FRAMES and frame_idx % WINDOW_STRIDE == 0:
            clip = normalize_clip(np.stack(buffer, axis=0))
            clip_t = torch.from_numpy(clip).permute(3, 0, 1, 2).float().unsqueeze(0).to(device)
            with torch.no_grad():
                v_logit, w_logit = model(clip_t)
                v_prob = torch.sigmoid(v_logit).item()
                w_prob = torch.sigmoid(w_logit).item()

        is_v, is_w = v_prob >= 0.5, w_prob >= 0.5
        color = (0, 0, 255) if (is_v and is_w) else (0, 140, 255) if (is_v or is_w) else (0, 180, 0)
        text = f"Violence {v_prob*100:.0f}% | Weapon {w_prob*100:.0f}%"
        cv2.rectangle(frame_bgr, (0, 0), (w, 50), color, -1)
        cv2.putText(frame_bgr, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        out.write(frame_bgr)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Demo saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--video', required=True)
    parser.add_argument('--output', default='demo_output.mp4')
    args = parser.parse_args()
    main(args.checkpoint, args.video, args.output)
