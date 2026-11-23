"""Data loading: COCO-style polygon annotations -> image/mask pairs."""
import json
import os
import random
from collections import defaultdict

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

from .config import Config


def load_annotations(cfg: Config):
    """Return (items, by_img).

    items: image entries whose file exists in train_images, sorted by file name
           (file names start with the capture time, so this is chronological).
    by_img: image id -> list of annotation dicts.
    Note: the same photo may be annotated by several annotators; each annotation
    set is a separate entry, as suggested by the competition description.
    """
    ann_path = f"{cfg.root}/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    with open(ann_path) as f:
        ann = json.load(f)
    by_img = defaultdict(list)
    for a in ann["annotations"]:
        by_img[a["image_id"]].append(a)
    files = set(os.listdir(f"{cfg.root}/train/train_images"))
    items = sorted((e for e in ann["images"] if e["file_name"] in files),
                   key=lambda e: e["file_name"])
    return items, by_img


def polygon_instances(anns, size, orig_size=2048):
    """One boolean mask per filament polygon, at `size` x `size`."""
    scale = size / orig_size
    masks = []
    for a in anns:
        m = Image.new("L", (size, size), 0)
        d = ImageDraw.Draw(m)
        for poly in a["segmentation"]:
            if len(poly) >= 6:
                d.polygon([c * scale for c in poly], fill=1)
        masks.append(np.array(m, dtype=bool))
    return masks


def build_pair(cfg: Config, entry, anns):
    """Load one image and its merged binary mask, both resized to cfg.size."""
    im = cv2.imread(f"{cfg.root}/train/train_images/{entry['file_name']}",
                    cv2.IMREAD_GRAYSCALE)
    im = cv2.resize(im, (cfg.size, cfg.size), interpolation=cv2.INTER_AREA)
    m = Image.new("L", (cfg.orig_size, cfg.orig_size), 0)
    d = ImageDraw.Draw(m)
    for a in anns:
        for poly in a["segmentation"]:
            if len(poly) >= 6:
                d.polygon(poly, fill=255)
    m = cv2.resize(np.array(m), (cfg.size, cfg.size), interpolation=cv2.INTER_AREA)
    return im, m


def load_split(cfg: Config):
    """Chronological split: the last `val_frac` of images form the validation set.

    A chronological split is used because neighbouring captures look alike, so a
    random split would inflate validation scores.
    """
    items, by_img = load_annotations(cfg)
    data = [build_pair(cfg, e, by_img[e["id"]]) for e in items]
    n_val = int(len(data) * cfg.val_frac)
    return (data[:-n_val], data[-n_val:], items[:-n_val], items[-n_val:], by_img)


class SegDataset(Dataset):
    def __init__(self, data, cfg: Config, train: bool, reps: int = 2):
        self.data, self.cfg, self.train, self.reps = data, cfg, train, reps

    def __len__(self):
        return len(self.data) * (self.reps if self.train else 1)

    def __getitem__(self, i):
        cfg = self.cfg
        im, m = self.data[i % len(self.data)]
        if self.train:
            y, x = np.random.randint(0, cfg.size - cfg.crop + 1, 2)
            im, m = im[y:y + cfg.crop, x:x + cfg.crop], m[y:y + cfg.crop, x:x + cfg.crop]
            if random.random() < 0.5:
                im, m = im[:, ::-1], m[:, ::-1]
            if random.random() < 0.5:
                im, m = im[::-1], m[::-1]
            k = random.randint(0, 3)
            im, m = np.rot90(im, k), np.rot90(m, k)
        im = (np.ascontiguousarray(im) / 255.0 - cfg.mean) / cfg.std
        m = np.ascontiguousarray(m) / 255.0
        return (torch.tensor(im, dtype=torch.float32)[None],
                torch.tensor(m, dtype=torch.float32)[None])
