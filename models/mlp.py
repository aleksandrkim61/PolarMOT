from typing import List, Any, Iterable

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim: int, fc_dims: Iterable[int], nonlinearity: nn.Module,
                 dropout_p: float = 0, use_batchnorm: bool = False, last_output_free: bool = False):
        super().__init__()
        assert isinstance(fc_dims, (list, tuple)
                          ), f"fc_dims must be a list or a tuple, but got {type(fc_dims)}"

        self.input_dim = input_dim
        self.fc_dims = fc_dims
        self.nonlinearity = nonlinearity
        # if dropout_p is None:
        #     dropout_p = 0
        # if use_batchnorm is None:
        # use_batchnorm = False
        self.dropout_p = dropout_p
        self.use_batchnorm = use_batchnorm

        layers: List[nn.Module] = []
        for layer_i, dim in enumerate(fc_dims):
            layers.append(nn.Linear(input_dim, dim))
            if last_output_free and layer_i == len(fc_dims) - 1:
                continue

            layers.append(nonlinearity)
            if dim != 1:
                if use_batchnorm:
                    layers.append(nn.BatchNorm1d(dim))
                if dropout_p > 0:
                    layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim
        self.fc_layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc_layers(x)

    @property
    def output_dim(self) -> int:
        return self.fc_dims[-1]
