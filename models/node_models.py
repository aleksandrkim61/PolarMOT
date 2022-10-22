from tkinter import E
from typing import List, Any

import torch
from torch import nn
from torch_scatter import scatter
from torch_geometric.utils import softmax as edge_softmax

from models.mlp import MLP
from models.utils import aggregate_features


class TimeAwareNodeModel(nn.Module):
    """ Class used to peform a node update during neural message passing """

    def __init__(self, flow_forward_model: MLP, flow_backward_model: MLP, node_mlp: MLP, node_agg_mode: str):
        super().__init__()
        assert (flow_forward_model.output_dim + flow_backward_model.output_dim ==
                node_mlp.input_dim), f"Flow models have incompatible output/input dims"
        self.flow_forward_model = flow_forward_model
        self.flow_backward_model = flow_backward_model
        self.node_mlp = node_mlp
        self.node_agg_mode = node_agg_mode

    def forward(self, x, edge_index, edge_attr, **kwargs):
        past_nodes, future_nodes = edge_index

        """
        x[past_nodes]  # features of nodes emitting messages
        edge_attr       # emitted messages
        x[future_nodes] # features of nodes receiving messages
        """
        # Collect and process forward-directed edge messages (past->present)
        # print(f"x[past_nodes], {x[past_nodes].shape}")
        # print(f"edge_attr, {edge_attr.shape}")

        # TODO: Try actually sharing edges: use a single flow MLP and then scatter twice - with past/future_nodes as index

        flow_forward_input = torch.cat([x[future_nodes], x[past_nodes], edge_attr], dim=1)  # [n_edges x edge_feature_count]
        # print(f"flow_forward_input, {flow_forward_input.shape}")
        assert len(flow_forward_input) == len(edge_attr)

        flow_forward = self.flow_forward_model(flow_forward_input)
        # print(f"flow_forward, {flow_forward.shape}")
        # print(f"flow_forward {flow_forward}")
        # print(f"flow_forward {flow_forward.grad_fn}")

        flow_forward_aggregated = scatter(src=flow_forward, index=future_nodes,
                                          reduce=self.node_agg_mode, dim=0, dim_size=len(x))
        # print(f"flow_forward_aggregated {flow_forward_aggregated}")
        # print(f"flow_forward_aggregated {flow_forward_aggregated.grad_fn}")

        # Collect and process backward-directed edge messages (present->past)
        flow_backward_input = torch.cat([x[past_nodes], x[future_nodes], edge_attr], dim=1)  # [n_edges x edge_feature_count]
        flow_backward = self.flow_backward_model(flow_backward_input)
        # print(f"flow_backward {flow_backward}")
        flow_backward_aggregated = scatter(src=flow_backward, index=past_nodes,
                                           reduce=self.node_agg_mode, dim=0, dim_size=len(x))
        # print(f"flow_backward_aggregated {flow_backward_aggregated}")
        # print(f"flow_backward_aggregated {flow_backward_aggregated.grad_fn}")

        # Concat both flows for each node
        flow_total = torch.cat((flow_forward_aggregated, flow_backward_aggregated), dim=1)
        return self.node_mlp(flow_total)


class ContextualNodeModel(nn.Module):
    """ Class used to peform a node update during neural message passing """

    def __init__(self, flow_forward_model: MLP, flow_frame_model: MLP, flow_backward_model: MLP,
                 total_flow_model: MLP, node_agg_mode: str, attention_model: nn.Module = None, node_aggr_sections: int = 3):
        super().__init__()
        assert (flow_forward_model.output_dim + flow_frame_model.output_dim + flow_backward_model.output_dim ==
                total_flow_model.input_dim), f"Flow models have incompatible output/input dims"
        self.flow_forward_model = flow_forward_model
        self.flow_frame_model = flow_frame_model
        self.flow_backward_model = flow_backward_model
        self.node_agg_mode = node_agg_mode
        self.attention_model = attention_model
        self.total_flow_model = total_flow_model
        self.node_aggr_sections = node_aggr_sections

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor,
                same_frame_edge_index: torch.Tensor, same_frame_edge_attr: torch.Tensor, **kwargs):
        past_nodes, future_nodes = edge_index
        """
        x[past_nodes]  # features of nodes emitting messages
        edge_attr       # emitted messages
        x[future_nodes] # features of nodes receiving messages
        """
        early_frame_nodes, later_frame_nodes = same_frame_edge_index

        # Collect and process forward-directed edge messages (past->present)
        flow_forward_input = torch.cat([x[future_nodes], x[past_nodes], edge_attr], dim=1)  # [n_edges x edge_feature_count]
        flow_forward = self.flow_forward_model(flow_forward_input)

        # Collect and process backward-directed edge messages (present->past)
        # [n_edges x edge_feature_count]
        flow_backward_input = torch.cat([x[past_nodes], x[future_nodes], edge_attr], dim=1)
        flow_backward = self.flow_backward_model(flow_backward_input)

        # Collect and process same frame edge messages
        flow_frame_input = torch.cat([x[early_frame_nodes],
                                      x[later_frame_nodes],
                                      same_frame_edge_attr], dim=1)  # [n_edges x edge_feature_count]
        flow_frame = self.flow_frame_model(flow_frame_input)

        if self.node_aggr_sections == 3:
            flow_forward_aggregated = aggregate_features(flow_forward, future_nodes,
                                                        len(x), self.node_agg_mode, self.attention_model, edge_attr=edge_attr)
            flow_backward_aggregated = aggregate_features(flow_backward, past_nodes,
                                                        len(x), self.node_agg_mode, self.attention_model, edge_attr=edge_attr)
            flow_frame_aggregated = aggregate_features(torch.vstack((flow_frame, flow_frame)),
                                                    torch.cat((early_frame_nodes, later_frame_nodes)),
                                                    len(x), self.node_agg_mode, self.attention_model, 
                                                    edge_attr=torch.vstack((same_frame_edge_attr, same_frame_edge_attr)))
        elif self.node_aggr_sections == 2:
            flow_temporal_aggregated = aggregate_features(torch.vstack((flow_forward, flow_backward)), 
                                                        torch.cat((future_nodes, past_nodes)),
                                                        len(x), self.node_agg_mode, self.attention_model, 
                                                        edge_attr=edge_attr)
            # print("flow_temporal_aggregated", flow_temporal_aggregated.shape)
            flow_temporal_aggregated /= 2.0
            flow_forward_aggregated = flow_temporal_aggregated
            flow_backward_aggregated = flow_temporal_aggregated
            flow_frame_aggregated = aggregate_features(torch.vstack((flow_frame, flow_frame)),
                                                    torch.cat((early_frame_nodes, later_frame_nodes)),
                                                    len(x), self.node_agg_mode, self.attention_model, 
                                                    edge_attr=torch.vstack((same_frame_edge_attr, same_frame_edge_attr)))
        elif self.node_aggr_sections == 1:
            flow_total_aggregated = aggregate_features(torch.vstack((flow_forward, flow_frame, flow_frame, flow_backward)),
                                                    torch.cat((future_nodes, early_frame_nodes, later_frame_nodes, past_nodes)),
                                                    len(x), self.node_agg_mode, self.attention_model, 
                                                    edge_attr=torch.vstack((edge_attr, same_frame_edge_attr, same_frame_edge_attr, edge_attr)))
            # print("flow_total_aggregated", flow_total_aggregated.shape)
            flow_total_aggregated /= 3.0
            flow_forward_aggregated = flow_total_aggregated
            flow_backward_aggregated = flow_total_aggregated
            flow_frame_aggregated = flow_total_aggregated

        # stack and aggregate everything, then triple duplicate for the total_flow_model

        # Concat both flows for each node
        flow_total = torch.cat((flow_forward_aggregated, flow_frame_aggregated, flow_backward_aggregated), dim=1)
        return self.total_flow_model(flow_total)


class UniformAggNodeModel(nn.Module):
    """ Class used to peform a node update during neural message passing """

    def __init__(self, flow_model, node_mlp, node_agg_mode: str):
        super().__init__()
        assert (flow_model.output_dim == node_mlp.input_dim), f"Flow models have incompatible output/input dims"
        self.flow_model = flow_model
        self.node_mlp = node_mlp
        self.node_agg_mode = node_agg_mode

    def forward(self, x, edge_index, edge_attr, **kwargs):
        past_nodes, future_nodes = edge_index

        """
        x[past_nodes]  # features of nodes emitting messages, past -> future
        edge_attr       # emitted messages
        x[future_nodes] # features of nodes receiving messages, future nodes receiving from earlier ones
        """
        # input features order does not matter as long as it is symmetric between two flow inputs
        #                               nodes receiving, nodes sending, edges
        flow_forward_input = torch.hstack((x[future_nodes], x[past_nodes], edge_attr))
        assert len(flow_forward_input) == len(edge_attr)
        # [n_edges x edge_feature_count]
        flow_backward_input = torch.hstack((x[past_nodes], x[future_nodes], edge_attr))
        assert flow_forward_input.shape == flow_backward_input.shape, f"{flow_forward_input.shape} != {flow_backward_input.shape}"

        # [2*n_edges x edge_feature_count]
        flow_total_input = torch.vstack((flow_forward_input, flow_backward_input))
        flow_processed = self.flow_model(flow_total_input)

        # aggregate features for each node based on features taken over each node
        # the index has to account for both incoming and outgoing edges - so that each edge is considered by both of its nodes
        flow_total = scatter(src=flow_processed, index=torch.cat((future_nodes, past_nodes)),
                             reduce=self.node_agg_mode, dim=0, dim_size=len(x))
        return self.node_mlp(flow_total)


class InitialTimeAwareNodeModel(nn.Module):
    """ Class used to peform an initial update for empty nodes at the beginning of message passing
    The initial features are simply a transformation of edge features and therefore depend largely on the graph structure
    The aggregation logic is the same as for `TimeAwareNodeModel` but without node features.

    The abscense of node features means that transforming features before aggregation is the same as transforming the input edge features before the node update. Therefore, the transform is only applied after aggregation and instead the initial edge model is more powerful in comparison.
    """

    def __init__(self, node_mlp, node_agg_mode: str):
        super().__init__()
        self.node_mlp = node_mlp
        self.node_agg_mode = node_agg_mode

    def forward(self, edge_index, edge_attr, num_nodes: int, **kwargs):
        past_nodes, future_nodes = edge_index
        # print(f"edge_attr, {edge_attr.shape}")

        flow_forward_aggregated = scatter(src=edge_attr, index=future_nodes,
                                          reduce=self.node_agg_mode, dim=0, dim_size=num_nodes)
        flow_backward_aggregated = scatter(src=edge_attr, index=past_nodes,
                                           reduce=self.node_agg_mode, dim=0, dim_size=num_nodes)

        # Concat both flows for each node
        flow_total = torch.cat((flow_forward_aggregated, flow_backward_aggregated), dim=1)
        # print(f"flow_total {flow_total.shape}")
        # print(f"initial flow_total {flow_total.grad_fn}")
        assert len(flow_total) == num_nodes
        return self.node_mlp(flow_total)


class InitialContextualNodeModel(nn.Module):
    """ Class used to peform an initial update for empty nodes at the beginning of message passing
    The initial features are simply a transformation of edge features and therefore depend largely on the graph structure
    The aggregation logic is the same as for `ContextualNodeModel` but without node features.

    The abscense of node features means that transforming features before aggregation is the same as transforming the input edge features before the node update. Therefore, the transform is only applied after aggregation and instead the initial edge model is more powerful in comparison.
    """

    def __init__(self, node_mlp, node_agg_mode: str, attention_model: nn.Module = None):
        super().__init__()
        self.node_mlp = node_mlp
        self.node_agg_mode = node_agg_mode
        self.attention_model = attention_model

    def forward(self, edge_index: torch.Tensor, edge_attr: torch.Tensor, num_nodes: int,
                same_frame_edge_index: torch.Tensor, same_frame_edge_attr: torch.Tensor, **kwargs):
        past_nodes, future_nodes = edge_index
        early_frame_nodes, later_frame_nodes = same_frame_edge_index

        flow_forward_aggregated = aggregate_features(edge_attr, future_nodes,
                                                     num_nodes, self.node_agg_mode, self.attention_model)
        flow_backward_aggregated = aggregate_features(edge_attr, past_nodes,
                                                      num_nodes, self.node_agg_mode, self.attention_model)
        flow_frame_aggregated = aggregate_features(torch.vstack((same_frame_edge_attr, same_frame_edge_attr)),
                                                   torch.cat((early_frame_nodes, later_frame_nodes)),
                                                   num_nodes, self.node_agg_mode, self.attention_model)
        # Concat all flows for each node
        flow_total = torch.cat((flow_forward_aggregated, flow_frame_aggregated, flow_backward_aggregated), dim=1)
        assert len(flow_total) == num_nodes
        return self.node_mlp(flow_total)


class InitialUniformAggNodeModel(nn.Module):
    """ Class used to peform an initial update for empty nodes at the beginning of message passing
    The initial features are simply a transformation of edge features and therefore depend largely on the graph structure
    The aggregation logic is the same as for `UniformAggNodeModel` but without node features.

    The abscense of node features means that transforming features before aggregation is the same as transforming the input edge features before the node update. Therefore, the transform is only applied after aggregation and instead the initial edge model is more powerful in comparison.
    """

    def __init__(self, node_mlp, node_agg_mode: str):
        super().__init__()
        self.node_mlp = node_mlp
        self.node_agg_mode = node_agg_mode

    def forward(self, edge_index, edge_attr, num_nodes: int, **kwargs):
        past_nodes, future_nodes = edge_index
        flow_total = scatter(src=torch.vstack((edge_attr, edge_attr)),
                             index=torch.cat((future_nodes, past_nodes)),
                             reduce=self.node_agg_mode, dim=0, dim_size=num_nodes)
        return self.node_mlp(flow_total)


class InitialZeroNodeModel(nn.Module):
    """ Class used to peform an initial update for empty nodes at the beginning of message passing
    The initial features are simply a transformation of edge features and therefore depend largely on the graph structure
    The aggregation logic is the same as for `ContextualNodeModel` but without node features.

    The abscense of node features means that transforming features before aggregation is the same as transforming the input edge features before the node update. Therefore, the transform is only applied after aggregation and instead the initial edge model is more powerful in comparison.
    """

    def __init__(self, node_dim: int):
        super().__init__()
        self.node_dim = node_dim

    def forward(self, edge_index: torch.Tensor, edge_attr: torch.Tensor, num_nodes: int,
                same_frame_edge_index: torch.Tensor, same_frame_edge_attr: torch.Tensor, device, **kwargs):
        return torch.zeros((num_nodes, self.node_dim), dtype=edge_attr.dtype, device=device)
