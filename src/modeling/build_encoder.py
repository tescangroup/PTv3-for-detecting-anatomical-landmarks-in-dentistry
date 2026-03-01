from src.modeling.ptv3 import ptv3_small, ptv3_medium, ptv3_large
from src.modeling.point_unet import SpUNetNoSkipBaseWrapper


REGISTRY = {
    "ptv3_small": ptv3_small,
    "ptv3_medium": ptv3_medium,
    "ptv3_large": ptv3_large,
    "conv": lambda input_channels, output_channels, patch_size: SpUNetNoSkipBaseWrapper(
        input_channels=input_channels, output_channels=output_channels)
}


def build_encoder(model_name: str, input_channels: int, output_channels: int, patch_size: int):
    return REGISTRY[model_name](input_channels=input_channels, output_channels=output_channels, patch_size=patch_size)
