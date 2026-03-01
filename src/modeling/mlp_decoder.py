import torch
from torch import nn
from typing import Dict, List, Tuple, Union


class MLPDecoder(nn.Module):
    def __init__(self,
                 input_channels: int,
                 output_channels: int = 256,
                 hidden_channels: int = 512,
                 hidden_layers: int = 2,
                 norm: bool = True,
                 dropout: bool = True
                 ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.hidden_channels = hidden_channels
        self.hidden_layers = hidden_layers
        self.norm = norm
        self.dropout = dropout

        self.mlp = nn.Sequential(*self.build_all_blocks())

    def build_all_blocks(self):
        modules = []

        previous_channels = self.input_channels
        for _ in range(self.hidden_layers):
            modules.extend(
                self.build_block(previous_channels, self.hidden_channels, self.norm, self.dropout)
            )
            previous_channels = self.hidden_channels

        modules.append(nn.Linear(previous_channels, self.output_channels))
        return modules

    def build_block(self, input_channels: int, output_channels: int, norm: bool, dropout: bool):
        modules = [nn.Linear(input_channels, output_channels)]

        if norm:
            modules.append(nn.BatchNorm1d(output_channels))

        modules.append(nn.ReLU())

        if dropout:
            modules.append(nn.Dropout())

        return modules

    def forward(self,
                batch_dict: Dict[str, Union[torch.Tensor, List[Tuple[Tuple[float, float, float], int]], List[str]]]
                ) -> Dict[str, Union[torch.Tensor, List[Tuple[Tuple[float, float, float], int]], List[str]]]:
        x = batch_dict["features"]
        B, N, _ = x.shape
        x = x.reshape(B * N, self.input_channels)
        batch_dict["dist_maps"] = self.mlp(x).view(B, N, self.output_channels)
        return batch_dict


def mlp_small(input_channels: int,
              output_channels: int = 256,
              hidden_channels: int = 512,
              device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    hidden_layers = 1
    norm = True
    dropout = True
    return MLPDecoder(
        input_channels=input_channels,
        output_channels=output_channels,
        hidden_channels=hidden_channels,
        hidden_layers=hidden_layers,
        norm=norm,
        dropout=dropout
    ).to(device)


def mlp_medium(input_channels: int,
               output_channels: int = 256,
               hidden_channels: int = 512,
               device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    hidden_layers = 3
    norm = True
    dropout = True
    return MLPDecoder(
        input_channels=input_channels,
        output_channels=output_channels,
        hidden_channels=hidden_channels,
        hidden_layers=hidden_layers,
        norm=norm,
        dropout=dropout
    ).to(device)


def mlp_large(input_channels: int,
              output_channels: int = 256,
              hidden_channels: int = 512,
              device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    hidden_layers = 5
    norm = True
    dropout = True
    return MLPDecoder(
        input_channels=input_channels,
        output_channels=output_channels,
        hidden_channels=hidden_channels,
        hidden_layers=hidden_layers,
        norm=norm,
        dropout=dropout
    ).to(device)
