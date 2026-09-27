"""Hash-grid encoding (tiny-cuda-nn) followed by a ReLU MLP.

The MLP / create_model pattern follows https://github.com/jqmcginnis/multi_contrast_inr.
"""
import torch
import torch.nn as nn
import tinycudann as tcnn


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim):
        super().__init__()
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, out_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.model(x)


def create_model(encoding_config, input_size, hidden_sizes, output_size):
    """Encoding (e.g. multi-resolution hash grid) + MLP; input coordinates in [0, 1]^input_size."""
    input_mapper = tcnn.Encoding(input_size, encoding_config, dtype=torch.float32)
    model = MLP(input_mapper.n_output_dims, hidden_sizes, output_size)
    return torch.nn.Sequential(input_mapper, model)
