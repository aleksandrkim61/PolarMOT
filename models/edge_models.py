from typing import List, Any

import torch
from torch import nn
from models.mlp import MLP


class BasicEdgeModel(nn.Module):
    """ Class used to peform an edge update during neural message passing """

    def __init__(self, edge_mlp):
        super(BasicEdgeModel, self).__init__()
        self.edge_mlp = edge_mlp

    def forward(self, x, edge_index, edge_attr):
        source_nodes, target_nodes = edge_index
        # assert len(source_nodes) == len(target_nodes) == len(
            # edge_attr), f"Different lengths {len(source_nodes)}, {len(target_nodes)}, {len(edge_attr)} "
        merged_features = torch.cat([x[source_nodes], x[target_nodes], edge_attr], dim=1)
        # print(f"merged_features, {merged_features.shape}")
        assert len(merged_features) == len(source_nodes), f"Merged input has wrong length {merged_features.shape} != {edge_attr.shape}"
        return self.edge_mlp(merged_features)
