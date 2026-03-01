from src.modeling.mlp_decoder import mlp_small, mlp_medium, mlp_large


REGISTRY = {
    "mlp_small": mlp_small,
    "mlp_medium": mlp_medium,
    "mlp_large": mlp_large,
}


def build_decoder(model_name: str, input_channels: int, output_channels: int, hidden_channels: int):
    return REGISTRY[model_name](input_channels=input_channels,
                                output_channels=output_channels,
                                hidden_channels=hidden_channels)
