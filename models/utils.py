from typing import Iterable, Tuple, Mapping, Dict

import torch
from torch_scatter import scatter
from torch_geometric.utils import softmax as edge_softmax

from models.mlp import MLP


def aggregate_features(features: torch.Tensor, agg_index: torch.Tensor, num_nodes: int, agg_mode: str, attention_model=None, edge_attr=None, **kwargs):
    if "attention" not in agg_mode:
        return scatter(src=features, index=agg_index, reduce=agg_mode, dim=0, dim_size=num_nodes)

    if "classifier" in agg_mode:
        assert edge_attr is not None
        coeffs = attention_model(edge_attr)
    else:
        coeffs = attention_model(features)
    assert coeffs.shape == (len(features), 1), f"features {features.shape}, coeffs {coeffs.shape}"
    # normalize coefficients for each node to sum to 1 (forward edges only)
    if "softmax" in agg_mode:
        coeffs = edge_softmax(coeffs, index=agg_index, num_nodes=num_nodes)
    elif "normalized" in agg_mode:
        coeffs = normalize_edge_coefficients(coeffs, index=agg_index, num_nodes=num_nodes)
    weighted_features = features * coeffs  # multiply by their attention coefficients and sum up
    return scatter(src=weighted_features, index=agg_index, reduce="sum", dim=0, dim_size=num_nodes)


def normalize_edge_coefficients(coeffs: torch.Tensor, index: torch.Tensor, num_nodes: int):
    coeffs_sum = scatter(coeffs, index=index, reduce="sum", dim=0, dim_size=num_nodes)
    coeffs_sum_selected = coeffs_sum.index_select(index=index, dim=0)
    return coeffs / (coeffs_sum_selected + 1e-16)


def dims_from_multipliers(output_dim: int, multipliers: Iterable[int]) -> Tuple[int, ...]:
    return tuple(int(output_dim * mult) for mult in multipliers)


def model_params_from_params(params: Mapping, model_prefix: str, param_names: Iterable[str]):
    return {param_name: params[f"{model_prefix}_{param_name}"] for param_name in param_names}
    # return {
    #     "input_dim":        params.get(f"{model_prefix}_input_dim"),
    #     "fc_dims":          params.get(f"{model_prefix}_fc_dims"),
    #     "nonlinearity":     params.get(f"{model_prefix}_nonlinearity"),
    #     "dropout_p":        params.get(f"{model_prefix}_dropout_p"),
    #     "use_batchnorm":    params.get(f"{model_prefix}_use_batchnorm"),
    # }
