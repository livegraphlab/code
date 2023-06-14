from typing import Optional, List, Type

import torch
import deepsnap

from config import cfg

from gnn_recurrent import *


def create_model(
    datasets: Optional[List[deepsnap.dataset.GraphDataset]] = None,
    to_device: bool = True,
    dim_in: Optional[int] = None,
    dim_out: Optional[int] = None
) -> Type[torch.nn.Module]:
    r"""Constructs the pytorch-geometric model.

    Args:
        datasets: A list of deepsnap.dataset.GraphDataset objects.
        to_device: A bool indicating whether to move the constructed model to the device in configuration.
        dim_in: An integer indicating the input dimension of the model.
            If not provided (None), infer from datasets and use `num_node_features`
        dim_out: An integer indicating the output dimension of the model
            If not provided (None), infer from datasets and use `num_node_features`
    Returns:
        The constructed pytorch model.
    """
    # FIXME: num_node_features/num_labels not working properly for HeteroGraph.
    dim_in = datasets[0].num_node_features if dim_in is None else dim_in
    dim_out = datasets[0].num_labels if dim_out is None else dim_out
    if 'classification' in cfg.dataset.task_type and dim_out == 2:
        # binary classification, output dim = 1
        dim_out = 1
    model = GNN(dim_in=dim_in, dim_out=dim_out)
    if to_device:
        model.to(torch.device(cfg.device))
    return model
