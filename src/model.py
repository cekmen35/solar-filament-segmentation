"""Model definition: U-Net with an ImageNet-pretrained encoder."""
import segmentation_models_pytorch as smp

from .config import Config


def build_model(cfg: Config, pretrained: bool = True):
    return smp.Unet(
        cfg.encoder,
        encoder_weights="imagenet" if pretrained else None,
        in_channels=1,   # images are grayscale
        classes=1,
    )
