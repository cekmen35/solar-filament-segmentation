"""Central configuration. Edit values here or override fields in code."""
from dataclasses import dataclass


@dataclass
class Config:
    # Paths (defaults match the Kaggle notebook environment)
    root: str = "/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"
    work_dir: str = "/kaggle/working"

    # Data
    orig_size: int = 2048     # native image size
    size: int = 1024          # images are downscaled to this size
    crop: int = 512           # random training crop
    val_frac: float = 0.15    # last 15% (by capture time) used for validation
    mean: float = 0.449
    std: float = 0.226

    # Training
    encoder: str = "resnet34"
    epochs: int = 25
    batch_size: int = 8
    lr: float = 3e-4
    weight_decay: float = 1e-4

    # Post-processing (tuned on the validation split, values at `size` resolution)
    thr: float = 0.3
    min_area: int = 100
    close_k: int = 5
