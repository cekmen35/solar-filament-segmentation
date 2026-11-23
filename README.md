# Solar Filament Segmentation (MAGFiLO, Kaggle 2026)

Instance segmentation of solar filaments in full-disk H-alpha images
(2048×2048, grayscale) from the
[Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026).

## Approach

1. **Data**: COCO-style polygon annotations are rasterised into one binary
"filament" mask per image. Chirality classes (Left/Right/Unidentifiable/Ambiguous)
are merged. Only images and polygons are used; spine, bbox and class labels are not.
The same photo can be annotated by several annotators; each annotation set is
treated as a separate sample.
2. **Preprocessing**: images are downscaled 2048 → 1024 (area interpolation).
Training uses random 512×512 crops with flips and 90° rotations.
3. **Model**: U-Net with an ImageNet-pretrained ResNet34 encoder
(`segmentation\_models\_pytorch`), single input channel.
4. **Loss / optimiser**: BCE + Dice (filaments cover <1% of pixels), AdamW,
one-cycle LR, mixed precision, 25 epochs.
5. **Inference**: full 1024 image, horizontal-flip TTA, probability map upsampled to 2048.
6. **Instance extraction**: threshold → morphological closing → connected
components → remove components below a minimum area. Each component becomes one
submission row (RLE via `pycocotools`).

## Validation

\## Results



The validation set is the last 15% of images by capture time (neighbouring captures

look alike, so a random split would inflate scores).



| Evaluation | Score |

|---|---|

| Validation pixel IoU (threshold 0.5) | \~0.50 |

| Validation PQ-style score (own implementation, 1024 px) | \~0.41 |

| Kaggle test score, public (late submission) | 0.32 |

| Kaggle test score, private (late submission) | 0.31 |



The Kaggle scores come from a submission made after the competition deadline, so

they do not count toward the official leaderboard. The competition describes its

metric as Panoptic Quality.



My validation PQ is optimistic: post-processing parameters

(`thr=0.3, min\_area=100, close\_k=5`) were tuned on the same validation split, and my

metric implementation may differ in details from the official one. The drop from

\~0.41 to \~0.31 suggests limited generalisation to the test set; I did not

investigate the cause (for example differences between observatories or dates).

Labels are also noisy because annotators disagree, which caps achievable IoU.Usage

```bash
pip install -r requirements.txt
# edit paths in src/config.py if not running on Kaggle
python -m src.train    # trains, saves best.pt
python -m src.tune     # grid search for post-processing parameters
python -m src.infer    # writes submission.csv
```

## Layout

```
src/config.py       configuration
src/data.py         annotation loading, masks, Dataset
src/model.py        U-Net definition
src/train.py        training loop
src/postprocess.py  probability map -> instances
src/metrics.py      PQ-style metric
src/tune.py         post-processing search
src/infer.py        test prediction and submission file
```

## Limitations / next steps

* Annotator disagreement is not modelled; averaging masks of the same image into
soft labels is a likely improvement.
* Downscaling to 1024 may lose thin barbs; higher resolution untested.
* Touching filaments are not split; no instance-aware model (e.g. watershed or
Mask R-CNN) was tried.

