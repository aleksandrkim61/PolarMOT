from typing import List, Optional

import torch
from torch import nn
import pytorch_lightning as pl
from models.mlp import MLP
from torch_geometric.data import Data

# TODO: rework these models to define only a single layer/pass and stack them in Sequential or something


class MessagePassingNetworkNonRecurrent(nn.Module):
    def __init__(self, edge_models: List[nn.Module], node_models: List[nn.Module], steps: int, use_same_frame: bool=False):
        """
        Args:
            edge_models: a list/tuple of callable Edge Update models
            node_models: a list/tuple of callable Node Update models
        """
        super().__init__()
        assert len(edge_models) == steps, f"steps={steps} not equal edge models {len(edge_models)}"
        assert len(node_models) == steps - 1, f"steps={steps} -1 not equal node models {len(node_models)}"
        self.edge_models = nn.ModuleList(edge_models)
        self.node_models = nn.ModuleList(node_models)
        self.steps = steps
        self.use_same_frame = use_same_frame

    def forward(self, x, edge_index, edge_attr, num_nodes: int):
        """
        Does a single node and edge feature vectors update.
        Args:
            x: node features matrix
            edge_index: tensor with shape [2, M], with M being the number of edges, indicating nonzero entries in the
            graph adjacency (i.e. edges)
            edge_attr: edge features matrix (ordered by edge_index)
        Returns: Updated Node and Edge Feature matrices
        """
        edge_embeddings = []
        for step, (edge_model, node_model) in enumerate(zip(self.edge_models, self.node_models.append(None))):
            # Edge Update
            edge_attr_mpn = edge_model(x, edge_index, edge_attr)
            edge_embeddings.append(edge_attr_mpn)

            if step == self.steps - 1:
                continue  # do not process nodes in the last step - only edge features are used for classification
            # Node Update
            x = node_model(x, edge_index, edge_attr_mpn)
        assert len(
            edge_embeddings) == self.steps, f"Collected {len(edge_embeddings)} edge embeddings for {self.steps} steps"
        return x, edge_embeddings


class MessagePassingNetworkRecurrent(nn.Module):
    def __init__(self, edge_model: nn.Module, node_model: nn.Module, steps: int,
                 use_same_frame: bool = False, same_frame_edge_model: Optional[nn.Module] = None):
        """
        Args:
            edge_model: an Edge Update model
            node_model: an Node Update model
        """
        super().__init__()
        self.edge_model = edge_model
        self.same_frame_edge_model = same_frame_edge_model
        self.node_model = node_model
        self.steps = steps
        self.use_same_frame = use_same_frame

    def forward(self, x, edge_index, edge_attr, num_nodes: int, same_frame_edge_index=None, same_frame_edge_attr=None):
        """
        Does a single node and edge feature vectors update.
        Args:
            x: node features matrix
            edge_index: tensor with shape [2, M], with M being the number of edges, indicating nonzero entries in the
            graph adjacency (i.e. edges)
            edge_attr: edge features matrix (ordered by edge_index)
        Returns: Updated Node and Edge Feature matrices
        """
        for step in range(self.steps):
            # Edge Update
            edge_attr_mpn = self.edge_model(x, edge_index, edge_attr)
            if self.use_same_frame:
                if self.same_frame_edge_model is not None:
                    same_frame_edge_attr_mpn = self.same_frame_edge_model(x, same_frame_edge_index, same_frame_edge_attr)
                else:
                    same_frame_edge_attr_mpn = self.edge_model(x, same_frame_edge_index, same_frame_edge_attr)
            else:
                same_frame_edge_attr_mpn = None

            if step == self.steps - 1:
                continue  # do not process nodes in the last step - only edge features are used for classification
            # Node Update
            x = self.node_model(x, edge_index, edge_attr_mpn,
                                same_frame_edge_index=same_frame_edge_index,
                                same_frame_edge_attr=same_frame_edge_attr_mpn)
        return x, edge_attr_mpn


class MessagePassingNetworkRecurrentNodeEdge(nn.Module):
    def __init__(self, edge_model: nn.Module, node_model: nn.Module, steps: int,
                 use_same_frame: bool = False, same_frame_edge_model: Optional[nn.Module] = None):
        """
        Args:
            edge_model: an Edge Update model
            node_model: an Node Update model
        """
        super().__init__()
        self.edge_model = edge_model
        self.same_frame_edge_model = same_frame_edge_model
        self.node_model = node_model
        self.steps = steps
        self.use_same_frame = use_same_frame

    def forward(self, x, edge_index, edge_attr, num_nodes: int, same_frame_edge_index=None, same_frame_edge_attr=None):
        """
        Does a single node and edge feature vectors update.
        Args:
            x: node features matrix
            edge_index: tensor with shape [2, M], with M being the number of edges, indicating nonzero entries in the
            graph adjacency (i.e. edges)
            edge_attr: edge features matrix (ordered by edge_index)
        Returns: Updated Node and Edge Feature matrices
        """
        edge_attr_mpn = edge_attr
        same_frame_edge_attr_mpn = same_frame_edge_attr
        for step in range(self.steps):
            # Node Update
            x = self.node_model(x, edge_index, edge_attr_mpn,
                                same_frame_edge_index=same_frame_edge_index,
                                same_frame_edge_attr=same_frame_edge_attr_mpn)

            # Edge Update
            edge_attr_mpn = self.edge_model(x, edge_index, edge_attr)
            if self.use_same_frame:
                if self.same_frame_edge_model is not None:
                    same_frame_edge_attr_mpn = self.same_frame_edge_model(x, same_frame_edge_index, same_frame_edge_attr)
                else:
                    same_frame_edge_attr_mpn = self.edge_model(x, same_frame_edge_index, same_frame_edge_attr)
            else:
                same_frame_edge_attr_mpn = None

        return x, edge_attr_mpn
