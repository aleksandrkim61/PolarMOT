from typing import List, Mapping, Any, Iterable, Tuple
import traceback
import os

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.data import Data
from torchmetrics import Precision, Recall

from models.mlp import MLP
from models.mpn import MessagePassingNetworkRecurrent, MessagePassingNetworkRecurrentNodeEdge, MessagePassingNetworkNonRecurrent
from models.node_models import (InitialTimeAwareNodeModel, TimeAwareNodeModel,
                                UniformAggNodeModel, InitialUniformAggNodeModel,
                                ContextualNodeModel, InitialContextualNodeModel,
                                InitialZeroNodeModel)
from models.edge_models import BasicEdgeModel
import models.utils as utils_models
from training.radam import RAdam
from training.focal_loss import focal_loss_binary


def _build_params_dict(initial_edge_model_input_dim, edge_dim, fc_dims_initial_edge_model_multipliers, nonlinearity_initial_edge,
                       fc_dims_initial_node_model_multipliers, nonlinearity_initial_node, 
                       directed_flow_agg, fc_dims_directed_flow_attention_model_multipliers,
                       fc_dims_edge_model_multipliers, nonlinearity_edge,
                       fc_dims_directed_flow_model_multipliers, nonlinearity_directed_flow, 
                       fc_dims_total_flow_model_multipliers, nonlinearity_total_flow,
                       fc_dims_edge_classification_model_multipliers, nonlinearity_edge_classification,
                       use_batchnorm: bool,
                       mpn_steps: int, is_recurrent: bool, node_dim_multiplier: int, pos_weight_multiplier: int,
                       use_timeaware: bool, use_same_frame: bool, use_separate_edge_model: bool, use_initial_node_model: bool,
                       edge_mlps_count: int,
                       node_aggr_sections: int,
                       lr, wd, loss_type: str,
                       seed,
                       optimizer_type,
                       scheduler_params: Mapping,
                       trainer_params: Mapping,
                       **kwargs,
                       ):
    # workaround before adding sacred
    params = {
        "seed": seed,

        "initial_edge_model_input_dim": initial_edge_model_input_dim,
        "edge_dim": edge_dim,

        "fc_dims_initial_edge_model_multipliers": fc_dims_initial_edge_model_multipliers,
        "nonlinearity_initial_edge": nonlinearity_initial_edge,

        "fc_dims_initial_node_model_multipliers": fc_dims_initial_node_model_multipliers,
        "nonlinearity_initial_node": nonlinearity_initial_node,
        "directed_flow_agg": directed_flow_agg,
        "fc_dims_directed_flow_attention_model_multipliers": fc_dims_directed_flow_attention_model_multipliers,

        "fc_dims_edge_model_multipliers": fc_dims_edge_model_multipliers,
        "nonlinearity_edge": nonlinearity_edge,

        "fc_dims_directed_flow_model_multipliers": fc_dims_directed_flow_model_multipliers,
        "nonlinearity_directed_flow": nonlinearity_directed_flow,

        "fc_dims_total_flow_model_multipliers": fc_dims_total_flow_model_multipliers,
        "nonlinearity_total_flow": nonlinearity_total_flow,

        "fc_dims_edge_classification_model_multipliers": fc_dims_edge_classification_model_multipliers,
        "nonlinearity_edge_classification": nonlinearity_edge_classification,

        "use_batchnorm": use_batchnorm,

        "mpn_steps": mpn_steps,
        "is_recurrent": is_recurrent,
        "node_dim_multiplier": node_dim_multiplier,
        "pos_weight_multiplier": pos_weight_multiplier,

        "use_timeaware": use_timeaware,
        "use_same_frame": use_same_frame,
        "use_separate_edge_model": use_separate_edge_model,
        "use_initial_node_model": use_initial_node_model,
        "edge_mlps_count": edge_mlps_count,
        "node_aggr_sections": node_aggr_sections,

        "lr": lr,
        "wd": wd,
        "loss_type": loss_type,
        "optimizer_type": optimizer_type,
        "scheduler_params": scheduler_params,

        "trainer_params": trainer_params,
    }
    params.update(kwargs)
    return params


def _build_models(params: Mapping[str, Any]):
    use_batchnorm = params["use_batchnorm"]

    edge_dim = params["edge_dim"]
    node_dim_multiplier = params.get("node_dim_multiplier", 2)
    node_dim = edge_dim * node_dim_multiplier  # Have nodes hold 2x info of edges
    use_timeaware = params.get("use_timeaware", True)
    use_same_frame = params.get("use_same_frame", False)
    # separate backward/forward/sameframe MLPs or inter/intraframe or single MLP for all
    edge_mlps_count = params.get("edge_mlps_count", 3)
    assert edge_mlps_count > 0 and edge_mlps_count <= 3, f"edge_mlps_count must be 1/2/3, not {edge_mlps_count}"
    node_aggr_sections = params.get("node_aggr_sections", 3)
    assert node_aggr_sections > 0 and node_aggr_sections <= 3, f"node_aggr_sections must be 1/2/3, not {node_aggr_sections}"
    # only makes sense when using intraframe
    use_separate_edge_model = use_same_frame and params.get("use_separate_edge_model", False) 
    use_initial_node_model = params.get("use_initial_node_model", True)

    # Edge classification model
    fc_dims_edge_classification_model_multipliers = params["fc_dims_edge_classification_model_multipliers"]
    if fc_dims_edge_classification_model_multipliers is not None:
        fc_dims_edge_classification_model = utils_models.dims_from_multipliers(
            edge_dim, fc_dims_edge_classification_model_multipliers) + (1,)
    else:
        fc_dims_edge_classification_model = (1,)
    edge_classifier = MLP(edge_dim, fc_dims_edge_classification_model,
                          params["nonlinearity_edge_classification"], last_output_free=True)

    # Initial edge model:
    fc_dims_initial_edge = utils_models.dims_from_multipliers(
        edge_dim, params["fc_dims_initial_edge_model_multipliers"])
    initial_edge_model = MLP(params["initial_edge_model_input_dim"], fc_dims_initial_edge,
                            params["nonlinearity_initial_edge"], use_batchnorm=use_batchnorm)
    if use_separate_edge_model:
        initial_same_frame_edge_model = MLP(params["initial_edge_model_input_dim"], fc_dims_initial_edge,
                                            params["nonlinearity_initial_edge"], use_batchnorm=use_batchnorm)
    else:
        initial_same_frame_edge_model = None

    # Initial node model
    if use_initial_node_model:
        initial_node_agg_mode = params["directed_flow_agg"]
        if "attention" in initial_node_agg_mode:
            if "classifier" in initial_node_agg_mode:
                initial_node_attention_model = edge_classifier
            else:
                fc_dims_directed_flow_attention_model_multipliers = params["fc_dims_directed_flow_attention_model_multipliers"]
                if fc_dims_directed_flow_attention_model_multipliers is not None:
                    fc_dims_initial_node_attention = utils_models.dims_from_multipliers(
                        edge_dim, fc_dims_directed_flow_attention_model_multipliers) + (1,)
                else:
                    fc_dims_initial_node_attention = (1,)
                initial_node_attention_model = MLP(edge_dim, fc_dims_initial_node_attention,
                                                params["nonlinearity_initial_node"], last_output_free=True)
        else:
            initial_node_attention_model = None

        fc_dims_initial_node = utils_models.dims_from_multipliers(
            node_dim, params["fc_dims_initial_node_model_multipliers"])
        if use_timeaware:
            if use_same_frame:
                initial_node_model = InitialContextualNodeModel(MLP(edge_dim * 3, fc_dims_initial_node,
                                                                params["nonlinearity_initial_node"], use_batchnorm=use_batchnorm),
                                                                initial_node_agg_mode, initial_node_attention_model)
            else:
                initial_node_model = InitialTimeAwareNodeModel(MLP(edge_dim * 2, fc_dims_initial_node,  # x2 for [forward|backward] edge features
                                                                   params["nonlinearity_initial_node"], use_batchnorm=use_batchnorm),
                                                               initial_node_agg_mode)
        else:
            assert not use_same_frame
            initial_node_model = InitialUniformAggNodeModel(MLP(edge_dim, fc_dims_initial_node,
                                                                params["nonlinearity_initial_node"], use_batchnorm=use_batchnorm),
                                                            initial_node_agg_mode)
    else:  # initial nodes are zero vectors
        initial_node_model = InitialZeroNodeModel(node_dim)

    # Define models in MPN
    edge_models, node_models = [], []
    steps = params["mpn_steps"]
    assert steps > 1, "Fewer than 2 MPN steps does not make sense as in that case nodes do not get a chance to update"
    is_recurrent = params["is_recurrent"]
    for step in range(steps):
        # Edge model
        edge_model_input = node_dim * 2 + edge_dim  # edge_dim * 5
        fc_dims_edge = utils_models.dims_from_multipliers(
            edge_dim, params["fc_dims_edge_model_multipliers"])
        edge_models.append(BasicEdgeModel(MLP(edge_model_input, fc_dims_edge,
                                              params["nonlinearity_edge"], use_batchnorm=use_batchnorm)))

        if step == steps - 1: # don't need a node update at the last step
            continue

        # Node model
        flow_model_input = node_dim * 2 + edge_dim  # two nodes and their edge
        fc_dims_directed_flow = utils_models.dims_from_multipliers(
            node_dim, params["fc_dims_directed_flow_model_multipliers"])
        fc_dims_aggregated_flow = utils_models.dims_from_multipliers(
            node_dim, params["fc_dims_total_flow_model_multipliers"])
        
        node_agg_mode = params["directed_flow_agg"]
        if "attention" in node_agg_mode:
            if "classifier" in node_agg_mode:
                attention_model = edge_classifier
            else:
                fc_dims_directed_flow_attention_model_multipliers = params["fc_dims_directed_flow_attention_model_multipliers"]
                if fc_dims_directed_flow_attention_model_multipliers is not None:
                    fc_dims_directed_flow_attention = utils_models.dims_from_multipliers(
                        node_dim, fc_dims_directed_flow_attention_model_multipliers) + (1,)
                else:
                    fc_dims_directed_flow_attention = (1,)
                attention_model = MLP(node_dim, fc_dims_directed_flow_attention,
                                    params["nonlinearity_directed_flow"], last_output_free=True)
        else:
            attention_model = None

        if use_timeaware:
            forward_flow_model = MLP(flow_model_input, fc_dims_directed_flow,
                                     params["nonlinearity_directed_flow"], use_batchnorm=use_batchnorm)
            if edge_mlps_count < 3:
                backward_flow_model = forward_flow_model
            else:
                backward_flow_model = MLP(flow_model_input, fc_dims_directed_flow,
                                        params["nonlinearity_directed_flow"], use_batchnorm=use_batchnorm)
            if use_same_frame:
                if edge_mlps_count == 1:
                    frame_flow_model = forward_flow_model
                else:
                    frame_flow_model = MLP(flow_model_input, fc_dims_directed_flow,
                                        params["nonlinearity_directed_flow"], use_batchnorm=use_batchnorm)
                aggregated_flow_model = MLP(node_dim * 3, fc_dims_aggregated_flow,
                                            params["nonlinearity_total_flow"], use_batchnorm=use_batchnorm)
                node_models.append(ContextualNodeModel(
                    forward_flow_model, frame_flow_model, backward_flow_model, aggregated_flow_model, node_agg_mode, attention_model, node_aggr_sections=node_aggr_sections))

            else:
                aggregated_flow_model = MLP(node_dim * 2, fc_dims_aggregated_flow,
                                            params["nonlinearity_total_flow"], use_batchnorm=use_batchnorm)
                node_models.append(TimeAwareNodeModel(
                    forward_flow_model, backward_flow_model, aggregated_flow_model, node_agg_mode))
        else:
            individual_flow_model = MLP(flow_model_input, fc_dims_directed_flow,
                                        params["nonlinearity_directed_flow"], use_batchnorm=use_batchnorm)
            aggregated_flow_model = MLP(node_dim, fc_dims_aggregated_flow,
                                        params["nonlinearity_total_flow"], use_batchnorm=use_batchnorm)
            node_models.append(UniformAggNodeModel(individual_flow_model,
                               aggregated_flow_model, node_agg_mode))

        if is_recurrent:  # only one model to use at each step
            break

    if is_recurrent:
        assert len(edge_models) == len(node_models) == 1
        if use_separate_edge_model:
            same_frame_edge_model = BasicEdgeModel(MLP(edge_model_input, fc_dims_edge, params["nonlinearity_edge"],
                                                    use_batchnorm=use_batchnorm))
        else:
            same_frame_edge_model = None

        if use_initial_node_model:
            mpn_model = MessagePassingNetworkRecurrent(edge_models[0], node_models[0], steps,
                                                    use_same_frame, same_frame_edge_model)
        else:  # use a node-to-edge MPN
            mpn_model = MessagePassingNetworkRecurrentNodeEdge(edge_models[0], node_models[0], steps,
                                                               use_same_frame, same_frame_edge_model)
    else:
        mpn_model = MessagePassingNetworkNonRecurrent(edge_models, node_models, steps, use_same_frame)

    return initial_edge_model, initial_same_frame_edge_model, initial_node_model, mpn_model, edge_classifier


class GraphTrackerOffline(pl.LightningModule):
    def __init__(self, params: Mapping):
        """ Top level model class holding all components necessary to perform tracking on a graph

        :param initial_edge_model: a torch model processing initial edge attributes
        :param initial_same_frame_edge_model: a torch model processing initial edge attributes for same frame edges
        :param initial_node_model: a torch model processing edge attributes to get initial node features
        :param mpn_model: a message passing model
        :param edge_classifier: a final classification model operating on final edge features
        :param params: params
        """
        super().__init__()
        self.params = params
        (self.initial_edge_model, self.initial_same_frame_edge_model, self.initial_node_model,
         self.mpn_model, self.edge_classifier) = _build_models(params)

        self.loss_type = self.params["loss_type"]
        self.use_same_frame = self.params["use_same_frame"]
        self.pos_weight_multiplier = self.params["pos_weight_multiplier"]

        self.recall_train = Recall(2, threshold=0.5, average='none', multiclass=True)
        self.recall_val = Recall(2, threshold=0.5, average='none', multiclass=True)
        self.precision_train = Precision(2, threshold=0.5, average='none', multiclass=True)
        self.precision_val = Precision(2, threshold=0.5, average='none', multiclass=True)

        self.save_hyperparameters()

    def forward(self, data):
        edge_index, edge_attr, num_nodes = data.edge_index.long(), data.edge_attr, data.num_nodes
        same_frame_edge_index = data.same_frame_edge_index.long() if self.use_same_frame else None
        same_frame_edge_attr = data.same_frame_edge_attr if self.use_same_frame else None

        # Initial Edge embeddings with Null node embeddings
        edge_attr = self.initial_edge_model(edge_attr)
        if self.use_same_frame:
            if self.initial_same_frame_edge_model is not None:
                same_frame_edge_attr = self.initial_same_frame_edge_model(same_frame_edge_attr)
            else:
                same_frame_edge_attr = self.initial_edge_model(same_frame_edge_attr)

        # Initial Node embeddings with Null original embeddings
        x = self.initial_node_model(edge_index, edge_attr, num_nodes,
                                    same_frame_edge_index=same_frame_edge_index, 
                                    same_frame_edge_attr=same_frame_edge_attr, 
                                    device=self.device)
        assert len(x) == num_nodes

        # Message Passing
        x, final_edge_embeddings = self.mpn_model(x, edge_index, edge_attr, num_nodes,
                                                 same_frame_edge_index=same_frame_edge_index,
                                                 same_frame_edge_attr=same_frame_edge_attr)
        return self.edge_classifier(final_edge_embeddings)

    def _compute_bce_loss(self, final_logits, y, pos_weight):
        return F.binary_cross_entropy_with_logits(final_logits.view(-1), y.view(-1),
                                                  pos_weight=pos_weight, reduction="none")

    def _compute_focal_loss(self, final_logits, y, pos_weight, gamma: float):
        return focal_loss_binary(final_logits.view(-1), y.view(-1), pos_weight=pos_weight,
                                 gamma=gamma, reduction="none")

    def compute_loss(self, final_class_logits, y):
        pos_count = y.sum()
        pos_weight = ((len(y) - pos_count) / pos_count) * self.pos_weight_multiplier if pos_count else None

        # TODO: extract focal gamma into a hparam
        if self.loss_type == "bce":
            loss = self._compute_bce_loss(final_class_logits, y, pos_weight)
        elif self.loss_type == "focal":
            loss = self._compute_focal_loss(final_class_logits, y, pos_weight, gamma=2)
        else:
            raise NotImplementedError(f"Unknown {self.loss_type} loss")

        loss = loss.mean()
        return loss, loss

    def training_step(self, batch, batch_idx):
        mode = "train"
        y = batch.y.float()
        final_class_logits = self.forward(batch)
        loss, final_mpn_edge_loss = self.compute_loss(final_class_logits, y)

        self.log(f"{mode}/loss", loss.detach().item(), on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log(f"{mode}_loss", loss.detach().item(), on_step=False, on_epoch=True,  prog_bar=True, logger=True)
        self.compute_and_log_metrics(final_class_logits, y, mode=mode)
        return loss

    def validation_step(self, batch, batch_idx):
        mode = "val"
        y = batch.y.float()
        final_class_logits = self.forward(batch)
        loss, final_mpn_edge_loss = self.compute_loss(final_class_logits, y)
        self.log("hp_metric", loss.detach().item(), on_step=False, on_epoch=True, prog_bar=False, logger=True)

        self.log_value_pairs((("loss", loss.detach().item()),), prefix=mode)
        self.compute_and_log_metrics(final_class_logits, y, mode=mode)
        return

    def predict_step(self, batch, batch_idx=None, dataloader_idx=None):
        with torch.no_grad():
            cont_preds = self._cont_preds_from_forward(self.forward(batch))
            labels = batch.y.detach().int() if batch.y is not None else None
            return cont_preds, labels

    def _cont_preds_from_forward(self, final_class_logits):  # returns continuous preds [0, 1]
        with torch.no_grad():
            return nn.Sigmoid()(final_class_logits).detach()  # .round().int()

    def compute_and_log_metrics(self, final_class_logits, y, mode: str):
        preds = self._cont_preds_from_forward(final_class_logits)
        targets = y.detach().int()
        if mode == "train":
            recall_metric = self.recall_train
            precision_metric = self.precision_train
        elif mode == "val":
            recall_metric = self.recall_val
            precision_metric = self.precision_val
        else:
            raise NotImplementedError(f"Unknown mode {mode} for compute_and_log_metrics")
        recall_metric(preds, targets)
        precision_metric(preds, targets)
        self.log_value_pairs((("positive_recall", recall_metric[1]),
                              ("positive_precision", precision_metric[1]),
                              ("negative_recall", recall_metric[0]),
                              ("negative_precision", precision_metric[0]),
                              ), prefix=mode)

    def configure_optimizers(self):
        lr = self.params["lr"]
        wd = self.params["wd"]
        optimizer_type: str = self.params["optimizer_type"].lower()
        if optimizer_type == "radam":
            self.optimizer = RAdam(self.parameters(), lr=lr, weight_decay=wd)
        elif optimizer_type == "adamw":
            self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=wd)
        elif optimizer_type == "sgd":
            self.optimizer = torch.optim.SGD(self.parameters(), lr=lr, weight_decay=wd,
                                             nesterov=True, momentum=0.9)

        scheduler_params: Mapping = self.params["scheduler_params"]
        if "T_0" in scheduler_params:
            self.annealing_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer,
                                                                                            T_0=scheduler_params["T_0"],
                                                                                            T_mult=scheduler_params.get("T_mult", 1),
                                                                                            eta_min=scheduler_params.get("eta_min", 1e-5))
        else:
            self.annealing_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                                  T_max=scheduler_params["T_max"],
                                                                                  eta_min=scheduler_params.get("eta_min", 1e-5))

        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.annealing_scheduler,
            }
        }

    def log_value_pairs(self, label_value_pairs: Iterable[Tuple[str, Any]], prefix=""):
        for label, value in label_value_pairs:
            self.log(f"{prefix}/{label}", value, on_step=False, on_epoch=True, prog_bar=False, logger=True)
            self.log(f"{prefix}_{label}", value, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def training_epoch_end(self, outputs):
        for name, params in self.named_parameters():
            try:
                self.logger[0].experiment.add_histogram(name, params, self.current_epoch)
            except:
                os.write(1, f"{name} cannot be added as a histogram\n{traceback.format_exc()}\n\n".encode())
                pass
