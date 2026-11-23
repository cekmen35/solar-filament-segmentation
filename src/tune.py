"""Grid-search post-processing parameters on the validation split.
Usage: python -m src.tune"""
import itertools

import numpy as np

from .config import Config
from .data import load_split, polygon_instances
from .infer import load_model, predict_prob
from .metrics import panoptic_quality
from .postprocess import postprocess


def main(cfg: Config = Config()):
    _, val_data, _, val_items, by_img = load_split(cfg)
    model = load_model(cfg)
    probs = [predict_prob(model, im, cfg).astype(np.float16) for im, _ in val_data]
    gts = [polygon_instances(by_img[e["id"]], cfg.size, cfg.orig_size) for e in val_items]

    results = []
    for thr, area, ck in itertools.product([0.3, 0.4, 0.5, 0.6], [20, 50, 100, 200], [0, 3, 5]):
        preds = [postprocess(p.astype(np.float32), thr, area, ck) for p in probs]
        results.append((panoptic_quality(gts, preds), thr, area, ck))
    for r in sorted(results, reverse=True)[:5]:
        print("PQ %.4f | thr %.1f | min_area %d | close_k %d" % r)


if __name__ == "__main__":
    main()
