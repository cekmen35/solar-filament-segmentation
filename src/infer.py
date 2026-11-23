"""Predict on the test set and write submission.csv.  Usage: python -m src.infer"""
import os

import cv2
import numpy as np
import pandas as pd
import torch
from pycocotools import mask as mask_utils

from .config import Config
from .model import build_model
from .postprocess import postprocess


def load_model(cfg: Config, device="cuda"):
    model = build_model(cfg, pretrained=False).to(device)
    model.load_state_dict(torch.load(f"{cfg.work_dir}/best.pt", map_location=device))
    return model.eval()


@torch.no_grad()
def predict_prob(model, im, cfg: Config, device="cuda"):
    """Probability map for a (size, size) uint8 image, with horizontal-flip TTA."""
    x = torch.tensor((im / 255.0 - cfg.mean) / cfg.std, dtype=torch.float32)[None, None].to(device)
    with torch.autocast("cuda"):
        p = (torch.sigmoid(model(x)) + torch.sigmoid(model(x.flip(-1))).flip(-1)) / 2
    return p[0, 0].float().cpu().numpy()


def main(cfg: Config = Config()):
    model = load_model(cfg)
    test_dir = f"{cfg.root}/test/test_images"
    full = cfg.orig_size
    scale = full // cfg.size                        # parameters were tuned at `size`
    close_k = cfg.close_k * scale + 1 if cfg.close_k > 0 else 0
    rows, empty = [], []

    for fn in sorted(os.listdir(test_dir)):
        stem = os.path.splitext(fn)[0]
        im = cv2.imread(f"{test_dir}/{fn}", cv2.IMREAD_GRAYSCALE)
        im = cv2.resize(im, (cfg.size, cfg.size), interpolation=cv2.INTER_AREA)
        prob = cv2.resize(predict_prob(model, im, cfg), (full, full),
                          interpolation=cv2.INTER_LINEAR)
        segs = postprocess(prob, cfg.thr, cfg.min_area * scale ** 2, close_k)
        if not segs:
            empty.append(stem)
        for i, s in enumerate(segs, 1):
            rle = mask_utils.encode(np.asfortranarray(s.astype(np.uint8)))
            rows.append((f"{stem}_{i}", rle["counts"].decode("utf-8")))

    sub = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    sub.to_csv(f"{cfg.work_dir}/submission.csv", index=False)
    print(len(sub), "rows |", len(empty), "images with no prediction")


if __name__ == "__main__":
    main()
