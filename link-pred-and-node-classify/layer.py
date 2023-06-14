import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
from torch_geometric.nn import GCNConv, ChebConv, TopKPooling

from config import cfg
from act import act_dict
from generalconv import GeneralConvLayer, GeneralEdgeConvLayer

from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import zeros, glorot
from torch_geometric.utils import add_remaining_self_loops
from torch_scatter import scatter_add

import math
from torch.nn import LSTM, GRU

import deepsnap

# General classes
class GeneralLayer(nn.Module):
    r"""General wrapper for layers that automatically constructs the
    learnable layer (e.g., graph convolution),
        and adds optional post-layer operations such as
            - batch normalization,
            - dropout,
            - activation functions.

    Note that this general layer only handle node features.
    """

    def __init__(self, name, dim_in, dim_out, has_act=True, has_bn=True,
                 has_l2norm=False, **kwargs):
        super(GeneralLayer, self).__init__()
        self.has_l2norm = has_l2norm
        has_bn = has_bn and cfg.gnn.batchnorm
        self.layer = layer_dict[name](dim_in, dim_out,
                                      bias=not has_bn, **kwargs)
        layer_wrapper = []
        if has_bn:
            layer_wrapper.append(nn.BatchNorm1d(
                dim_out, eps=cfg.bn.eps, momentum=cfg.bn.mom))
        if cfg.gnn.dropout > 0:
            layer_wrapper.append(nn.Dropout(
                p=cfg.gnn.dropout, inplace=cfg.mem.inplace))
        if has_act:
            layer_wrapper.append(act_dict[cfg.gnn.act])
        self.post_layer = nn.Sequential(*layer_wrapper)

    def forward(self, batch):
        batch = self.layer(batch)
        if isinstance(batch, torch.Tensor):
            batch = self.post_layer(batch)
            if self.has_l2norm:
                batch = F.normalize(batch, p=2, dim=1)
        else:
            # Only modify node features here.
            batch.node_feature = self.post_layer(batch.node_feature)
            if self.has_l2norm:
                batch.node_feature = F.normalize(batch.node_feature, p=2,
                                                 dim=1)
        return batch


# General classes
class GeneralRecurrentLayer(nn.Module):
    r"""General wrapper for layers that automatically constructs the
    learnable layer (e.g., graph convolution),
        and adds optional post-layer operations such as
            - batch normalization,
            - dropout,
            - activation functions.

    Note that this general layer only handle node features.
    """

    def __init__(self, name, dim_in, dim_out, has_act=True, has_bn=True,
                 has_l2norm=False, id=0, **kwargs):
        super(GeneralRecurrentLayer, self).__init__()
        self.has_l2norm = has_l2norm
        self.id = id
        has_bn = has_bn and cfg.gnn.batchnorm
        self.layer = layer_dict[name](dim_in, dim_out,
                                      bias=not has_bn, **kwargs)
        layer_wrapper = []
        if has_bn:
            layer_wrapper.append(nn.BatchNorm1d(
                dim_out, eps=cfg.bn.eps, momentum=cfg.bn.mom))
        if cfg.gnn.dropout > 0:
            layer_wrapper.append(nn.Dropout(
                p=cfg.gnn.dropout, inplace=cfg.mem.inplace))
        if has_act:
            layer_wrapper.append(act_dict[cfg.gnn.act])
        self.post_layer = nn.Sequential(*layer_wrapper)

    def forward(self, batch):
        if isinstance(batch, torch.Tensor):
            batch = self.layer(batch)
            batch = self.post_layer(batch)
            if self.has_l2norm:
                batch = F.normalize(batch, p=2, dim=1)
        elif isinstance(batch.node_feature, dict):
            # Heterogeneous GNN.
            batch = self.layer(batch)
            for key in batch.node_feature.keys():
                # apply the same operations on every node type.
                batch.node_feature[key] = self.post_layer(
                    batch.node_feature[key])
                if self.has_l2norm:
                    batch.node_feature[key] = F.normalize(
                        batch.node_feature[key], p=2, dim=1)
                # weighted sum of new emb and old embedding
                batch.node_states[key][self.id] = \
                    batch.node_states[key][self.id] * batch.keep_ratio[key] \
                    + batch.node_feature[key] * (1 - batch.keep_ratio[key])
                batch.node_feature[key] = batch.node_states[key][self.id]
        else:
            # # Only modify node features here.
            # if self.id == 0:
            #     node_feature_input = batch.node_feature
            # else:
            #     node_feature_input = batch.node_states[self.id - 1]
            # # weighted sum of new emb and old embedding
            # if self.id == len(batch.node_states):
            #     # the final layer, output to head function
            #     batch.node_feature = \
            #         batch.node_states[self.id] * batch.keep_ratio \
            #         + self.post_layer(node_feature_input) * (
            #                     1 - batch.keep_ratio)
            # else:
            #     batch.node_states[self.id] = \
            #         batch.node_states[self.id] * batch.keep_ratio \
            #         + self.post_layer(node_feature_input) * (
            #                 1 - batch.keep_ratio)
            # Only modify node features here.
            # output to batch.node_feature
            # if torch.is_tensor(batch.node_states[self.id - 1]):
            #     print('before', batch.node_feature.sum(), batch.node_states[self.id - 1].sum())
            batch = self.layer(batch)
            # if torch.is_tensor(batch.node_states[self.id - 1]):
            #     print('after', batch.node_feature.sum(), batch.node_states[self.id - 1].sum())
            batch.node_feature = self.post_layer(batch.node_feature)
            if self.has_l2norm:
                batch.node_feature = F.normalize(batch.node_feature, p=2,
                                                 dim=1)
            # weighted sum of new emb and old embedding
            batch.node_states[self.id] = \
                batch.node_states[self.id] * batch.keep_ratio \
                + batch.node_feature * (1 - batch.keep_ratio)
            batch.node_feature = batch.node_states[self.id]
            # print('verify', batch.node_feature.sum(),
            #       batch.node_states[self.id].sum())

        return batch


class GRUGraphRecurrentLayer(nn.Module):
    r"""General wrapper for layers that automatically constructs the
    learnable layer (e.g., graph convolution),
        and adds optional post-layer operations such as
            - batch normalization,
            - dropout,
            - activation functions.

    This module updates nodes' hidden states differently based on nodes'
    activities. For nodes with edges in the current snapshot, their states
    are updated using an internal GRU; for other inactive nodes, their
    states are updated using simple MLP.
    """

    def __init__(self, name, dim_in, dim_out, has_act=True, has_bn=True,
                 has_l2norm=False, id=0, **kwargs):
        super(GRUGraphRecurrentLayer, self).__init__()
        self.has_l2norm = has_l2norm
        self.layer_id = id
        has_bn = has_bn and cfg.gnn.batchnorm
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.layer = layer_dict[name](dim_in, dim_out,
                                      bias=not has_bn, **kwargs)
        layer_wrapper = []
        if has_bn:
            layer_wrapper.append(nn.BatchNorm1d(
                dim_out, eps=cfg.bn.eps, momentum=cfg.bn.mom))
        if cfg.gnn.dropout > 0:
            layer_wrapper.append(nn.Dropout(
                p=cfg.gnn.dropout, inplace=cfg.mem.inplace))
        if has_act:
            layer_wrapper.append(act_dict[cfg.gnn.act])
        self.post_layer = nn.Sequential(*layer_wrapper)

        # dim_out = dim_hidden.
        # update gate.
        self.GRU_Z = nn.Sequential(
            nn.Linear(dim_in + dim_out, dim_out, bias=True),
            nn.Sigmoid())
        # reset gate.
        self.GRU_R = nn.Sequential(
            nn.Linear(dim_in + dim_out, dim_out, bias=True),
            nn.Sigmoid())
        # new embedding gate.
        self.GRU_H_Tilde = nn.Sequential(
            nn.Linear(dim_in + dim_out, dim_out, bias=True),
            nn.Tanh())

        # self.direct_forward = nn.Sequential(
        #     nn.Linear(dim_in + dim_out, dim_out),
        #     nn.ReLU(),
        #     nn.Linear(dim_out, dim_out))

    def _init_hidden_state(self, batch):
        # Initialize hidden states of all nodes to zero.
        if not isinstance(batch.node_states[self.layer_id], torch.Tensor):
            batch.node_states[self.layer_id] = torch.zeros(
                batch.node_feature.shape[0], self.dim_out).to(
                batch.node_feature.device)

    def forward(self, batch):
        batch = self.layer(batch)
        batch.node_feature = self.post_layer(batch.node_feature)
        if self.has_l2norm:
            batch.node_feature = F.normalize(batch.node_feature, p=2, dim=1)

        self._init_hidden_state(batch)
        # Compute output from GRU module.
        H_prev = batch.node_states[self.layer_id]
        X = batch.node_feature
        Z = self.GRU_Z(torch.cat([X, H_prev], dim=1))
        R = self.GRU_R(torch.cat([X, H_prev], dim=1))
        H_tilde = self.GRU_H_Tilde(torch.cat([X, R * H_prev], dim=1))
        H_gru = Z * H_prev + (1 - Z) * H_tilde

        if cfg.gnn.embed_update_method == 'masked_gru':
            # Update for active nodes only, use output from GRU.
            keep_mask = (batch.node_degree_new == 0)
            H_out = H_gru
            # Reset inactive nodes' embedding.
            H_out[keep_mask, :] = H_prev[keep_mask, :]
        elif cfg.gnn.embed_update_method == 'moving_average_gru':
            # Only update for active nodes, using moving average with output from GRU.
            H_out = H_prev * batch.keep_ratio + H_gru * (1 - batch.keep_ratio)
        elif cfg.gnn.embed_update_method == 'gru':
            # Update all nodes' embedding using output from GRU.
            H_out = H_gru
        else:
            raise ValueError(f'Invalid embedding update rule: {cfg.gnn.embed_update_method}')

        batch.node_states[self.layer_id] = H_out
        batch.node_feature = batch.node_states[self.layer_id]
        return batch


class GeneralMultiLayer(nn.Module):
    r"""General wrapper for stack of layers"""

    def __init__(self, name, num_layers, dim_in, dim_out, dim_inner=None,
                 final_act=True, **kwargs):
        super(GeneralMultiLayer, self).__init__()
        dim_inner = dim_in if dim_inner is None else dim_inner
        for i in range(num_layers):
            d_in = dim_in if i == 0 else dim_inner
            d_out = dim_out if i == num_layers - 1 else dim_inner
            has_act = final_act if i == num_layers - 1 else True
            layer = GeneralLayer(name, d_in, d_out, has_act, **kwargs)
            self.add_module('Layer_{}'.format(i), layer)

    def forward(self, batch):
        for layer in self.children():
            batch = layer(batch)
        return batch


# Core basic layers
# Input: batch; Output: batch
class Linear(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(Linear, self).__init__()
        self.model = nn.Linear(dim_in, dim_out, bias=bias)

    def forward(self, batch):
        if isinstance(batch, torch.Tensor):
            batch = self.model(batch)
        else:
            batch.node_feature = self.model(batch.node_feature)
        return batch


class BatchNorm1dNode(nn.Module):
    '''General wrapper for layers'''

    def __init__(self, dim_in):
        super(BatchNorm1dNode, self).__init__()
        self.bn = nn.BatchNorm1d(dim_in, eps=cfg.bn.eps, momentum=cfg.bn.mom)

    def forward(self, batch):
        batch.node_feature = self.bn(batch.node_feature)
        return batch


class BatchNorm1dEdge(nn.Module):
    '''General wrapper for layers'''

    def __init__(self, dim_in):
        super(BatchNorm1dEdge, self).__init__()
        self.bn = nn.BatchNorm1d(dim_in, eps=cfg.bn.eps, momentum=cfg.bn.mom)

    def forward(self, batch):
        if batch.edge_feature.size(0) > 1:
            batch.edge_feature = self.bn(batch.edge_feature)
        return batch


class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, bias=True, dim_inner=None,
                 num_layers=2, **kwargs):
        '''
        Note: MLP works for 0 layers
        '''
        super(MLP, self).__init__()
        dim_inner = dim_in if dim_inner is None else dim_inner
        layers = []
        if num_layers > 1:
            layers.append(
                GeneralMultiLayer('linear', num_layers - 1, dim_in, dim_inner,
                                  dim_inner, final_act=True))
            layers.append(Linear(dim_inner, dim_out, bias))
        else:
            layers.append(Linear(dim_in, dim_out, bias))
        self.model = nn.Sequential(*layers)

    def forward(self, batch):
        if isinstance(batch, torch.Tensor):
            batch = self.model(batch)
        else:
            batch.node_feature = self.model(batch.node_feature)
        return batch


# class GCNConv(nn.Module):
#     def __init__(self, dim_in, dim_out, bias=False, **kwargs):
#         super(GCNConv, self).__init__()
#         self.model = pyg.nn.GCNConv(dim_in, dim_out, bias=bias)

#     def forward(self, batch):
#         batch.node_feature = self.model(batch.node_feature, batch.edge_index)
#         return batch


class SAGEConv(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(SAGEConv, self).__init__()
        self.model = pyg.nn.SAGEConv(dim_in, dim_out, bias=bias, concat=True)

    def forward(self, batch):
        batch.node_feature = self.model(batch.node_feature, batch.edge_index)
        return batch


class GATConv(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(GATConv, self).__init__()
        self.model = pyg.nn.GATConv(dim_in, dim_out, bias=bias)

    def forward(self, batch):
        batch.node_feature = self.model(batch.node_feature, batch.edge_index)
        return batch


class GINConv(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(GINConv, self).__init__()
        gin_nn = nn.Sequential(nn.Linear(dim_in, dim_out), nn.ReLU(),
                               nn.Linear(dim_out, dim_out))
        self.model = pyg.nn.GINConv(gin_nn)

    def forward(self, batch):
        batch.node_feature = self.model(batch.node_feature, batch.edge_index)
        return batch


class SplineConv(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(SplineConv, self).__init__()
        self.model = pyg.nn.SplineConv(dim_in, dim_out,
                                       dim=1, kernel_size=2, bias=bias)

    def forward(self, batch):
        batch.node_feature = self.model(batch.node_feature, batch.edge_index,
                                        batch.edge_feature)
        return batch


class GeneralConv(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(GeneralConv, self).__init__()
        self.model = GeneralConvLayer(dim_in, dim_out, bias=bias)

    def forward(self, batch):
        batch.node_feature = self.model(batch.node_feature, batch.edge_index)
        return batch


class GeneralEdgeConv(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(GeneralEdgeConv, self).__init__()
        self.model = GeneralEdgeConvLayer(dim_in, dim_out, bias=bias)

    def forward(self, batch):
        batch.node_feature = self.model(batch.node_feature, batch.edge_index,
                                        edge_feature=batch.edge_feature)
        return batch


class GeneralSampleEdgeConv(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(GeneralSampleEdgeConv, self).__init__()
        self.model = GeneralEdgeConvLayer(dim_in, dim_out, bias=bias)

    def forward(self, batch):
        edge_mask = torch.rand(batch.edge_index.shape[1]) < cfg.gnn.keep_edge
        edge_index = batch.edge_index[:, edge_mask]
        edge_feature = batch.edge_feature[edge_mask, :]
        batch.node_feature = self.model(batch.node_feature, edge_index,
                                        edge_feature=edge_feature)
        return batch


class GraphRecurrentLayerWrapper(nn.Module):
    """
    The most general wrapper for graph recurrent layer, users can customize
        (1): the GNN block for message passing.
        (2): the update block takes {previous embedding, new node feature} and
            returns new node embedding.
    """
    def __init__(self, name, dim_in, dim_out, has_act=True, has_bn=True,
                 has_l2norm=False, id=0, **kwargs):
        super(GraphRecurrentLayerWrapper, self).__init__()
        self.has_l2norm = has_l2norm
        self.layer_id = id
        has_bn = has_bn and cfg.gnn.batchnorm
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.layer = layer_dict[name](dim_in, dim_out,
                                      bias=not has_bn, **kwargs)
        layer_wrapper = []
        if has_bn:
            layer_wrapper.append(nn.BatchNorm1d(
                dim_out, eps=cfg.bn.eps, momentum=cfg.bn.mom))
        if cfg.gnn.dropout > 0:
            layer_wrapper.append(nn.Dropout(
                p=cfg.gnn.dropout, inplace=cfg.mem.inplace))
        if has_act:
            layer_wrapper.append(act_dict[cfg.gnn.act])
        self.post_layer = nn.Sequential(*layer_wrapper)
        self.embedding_updater = construct_update_block(self.dim_in,
                                                        self.dim_out,
                                                        self.layer_id)

    def _init_hidden_state(self, batch):
        # Initialize hidden states of all nodes to zero.
        if not isinstance(batch.node_states[self.layer_id], torch.Tensor):
            batch.node_states[self.layer_id] = torch.zeros(
                batch.node_feature.shape[0], self.dim_out).to(
                batch.node_feature.device)

    def forward(self, batch):
        # Message passing.
        batch = self.layer(batch)
        batch.node_feature = self.post_layer(batch.node_feature)
        if self.has_l2norm:
            batch.node_feature = F.normalize(batch.node_feature, p=2, dim=1)

        self._init_hidden_state(batch)
        # Compute output from updater block.
        node_states_new = self.embedding_updater(batch)
        batch.node_states[self.layer_id] = node_states_new
        batch.node_feature = batch.node_states[self.layer_id]
        return batch


def construct_update_block(dim_in, dim_out, layer_id):
    # Helper function to construct an embedding updating module.
    # GRU-based models.
    # TODO: for code-release: make this clear.
    if cfg.gnn.embed_update_method in ['masked_gru', 'moving_average_gru', 'gru']:
        if cfg.gnn.gru_kernel == 'linear':
            # Simple GRU.
            return GRUUpdater(dim_in, dim_out, layer_id)
        else:
            # GNN-empowered GRU.
            return GraphConvGRUUpdater(dim_in, dim_out, layer_id,
                                       layer_dict[cfg.gnn.gru_kernel])
    elif cfg.gnn.embed_update_method == 'mlp':
        return MLPUpdater(dim_in, dim_out, layer_id, cfg.gnn.mlp_update_layers)
    else:
        raise NameError(f'Unknown embedding update method: {cfg.gnn.embed_update_method}.')


class MovingAverageUpdater(nn.Module):
    # TODO: complete this for code release.
    def __init__(self,):
        raise NotImplementedError()

    def forward(self, batch):
        # TODO: copy from the old implementation.
        raise NotImplementedError()


class MLPUpdater(nn.Module):
    """
    Node embedding update block using simple MLP.
    embedding_new <- MLP([embedding_old, node_feature_new])
    """
    def __init__(self, dim_in, dim_out, layer_id, num_layers):
        super(MLPUpdater, self).__init__()
        self.layer_id = layer_id
        # FIXME:
        assert num_layers > 1, 'There is a problem with layer=1 now, pending fix.'
        self.mlp = MLP(dim_in=dim_in + dim_out, dim_out=dim_out,
                       num_layers=num_layers)

    def forward(self, batch):
        H_prev = batch.node_states[self.layer_id]
        X = batch.node_feature
        concat = torch.cat((H_prev, X), axis=1)
        H_new = self.mlp(concat)
        batch.node_states[self.layer_id] = H_new
        return H_new


class GRUUpdater(nn.Module):
    """
    Node embedding update block using standard GRU and variations of it.
    """
    def __init__(self, dim_in, dim_out, layer_id):
        # dim_in (dim of X): dimension of input node_feature.
        # dim_out (dim of H): dimension of previous and current hidden states.
        # forward(X, H) --> H.
        super(GRUUpdater, self).__init__()
        self.layer_id = layer_id
        self.GRU_Z = nn.Sequential(
            nn.Linear(dim_in + dim_out, dim_out, bias=True),
            nn.Sigmoid())
        # reset gate.
        self.GRU_R = nn.Sequential(
            nn.Linear(dim_in + dim_out, dim_out, bias=True),
            nn.Sigmoid())
        # new embedding gate.
        self.GRU_H_Tilde = nn.Sequential(
            nn.Linear(dim_in + dim_out, dim_out, bias=True),
            nn.Tanh())
    
    def forward(self, batch):
        H_prev = batch.node_states[self.layer_id]
        X = batch.node_feature
        Z = self.GRU_Z(torch.cat([X, H_prev], dim=1))
        R = self.GRU_R(torch.cat([X, H_prev], dim=1))
        H_tilde = self.GRU_H_Tilde(torch.cat([X, R * H_prev], dim=1))
        H_gru = Z * H_prev + (1 - Z) * H_tilde

        if cfg.gnn.embed_update_method == 'masked_gru':
            # Update for active nodes only, use output from GRU.
            keep_mask = (batch.node_degree_new == 0)
            H_out = H_gru
            # Reset inactive nodes' embedding.
            H_out[keep_mask, :] = H_prev[keep_mask, :]
        elif cfg.gnn.embed_update_method == 'moving_average_gru':
            # Only update for active nodes, using moving average with output from GRU.
            H_out = H_prev * batch.keep_ratio + H_gru * (1 - batch.keep_ratio)
        elif cfg.gnn.embed_update_method == 'gru':
            # Update all nodes' embedding using output from GRU.
            H_out = H_gru
        return H_out


class GraphConvGRUUpdater(nn.Module):
    """
    Node embedding update block using GRU with internal GNN and variations of
    it.
    """
    def __init__(self, dim_in, dim_out, layer_id, conv):
        # dim_in (dim of X): dimension of input node_feature.
        # dim_out (dim of H): dimension of previous and current hidden states.
        # forward(X, H) --> H.
        super(GraphConvGRUUpdater, self).__init__()
        self.layer_id = layer_id
        
        self.GRU_Z = conv(dim_in=dim_in + dim_out, dim_out=dim_out)
        # reset gate.
        self.GRU_R = conv(dim_in=dim_in + dim_out, dim_out=dim_out)
        # new embedding gate.
        self.GRU_H_Tilde = conv(dim_in=dim_in + dim_out, dim_out=dim_out)

    def forward(self, batch):
        H_prev = batch.node_states[self.layer_id]
        X = batch.node_feature
        # Combe previous node embedding and current feature for message passing.
        batch_z = deepsnap.graph.Graph()
        batch_z.node_feature = torch.cat([X, H_prev], dim=1).clone()
        batch_z.edge_feature = batch.edge_feature.clone()
        batch_z.edge_index = batch.edge_index.clone()

        batch_r = deepsnap.graph.Graph()
        batch_r.node_feature = torch.cat([X, H_prev], dim=1).clone()
        batch_r.edge_feature = batch.edge_feature.clone()
        batch_r.edge_index = batch.edge_index.clone()

        # (num_nodes, dim_out)
        Z = nn.functional.sigmoid(self.GRU_Z(batch_z).node_feature)
        # (num_nodes, dim_out)
        R = nn.functional.sigmoid(self.GRU_R(batch_r).node_feature)

        batch_h = deepsnap.graph.Graph()
        batch_h.node_feature = torch.cat([X, R * H_prev], dim=1).clone()
        batch_h.edge_feature = batch.edge_feature.clone()
        batch_h.edge_index = batch.edge_index.clone()

        # (num_nodes, dim_out)
        H_tilde = nn.functional.tanh(self.GRU_H_Tilde(batch_h).node_feature)
        H_gru = Z * H_prev + (1 - Z) * H_tilde

        if cfg.gnn.embed_update_method == 'masked_gru':
            # Update for active nodes only, use output from GRU.
            keep_mask = (batch.node_degree_new == 0)
            H_out = H_gru
            # Reset inactive nodes' embedding.
            H_out[keep_mask, :] = H_prev[keep_mask, :]
        elif cfg.gnn.embed_update_method == 'moving_average_gru':
            # Only update for active nodes, using moving average with output from GRU.
            H_out = H_prev * batch.keep_ratio + H_gru * (1 - batch.keep_ratio)
        elif cfg.gnn.embed_update_method == 'gru':
            # Update all nodes' embedding using output from GRU.
            H_out = H_gru

        return H_out


class ResidualEdgeConvLayer(MessagePassing):
    r"""General GNN layer, with arbitrary edge features.
    """

    def __init__(self, in_channels, out_channels, improved=False, cached=False,
                 bias=True, **kwargs):
        super(ResidualEdgeConvLayer, self).__init__(aggr=cfg.gnn.agg, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.normalize = cfg.gnn.normalize_adj
        self.msg_direction = cfg.gnn.msg_direction

        if self.msg_direction == 'single':
            self.linear_msg = nn.Linear(in_channels + cfg.dataset.edge_dim,
                                        out_channels, bias=False)
        elif self.msg_direction == 'both':
            self.linear_msg = nn.Linear(in_channels * 2 + cfg.dataset.edge_dim,
                                        out_channels, bias=False)
        else:
            raise ValueError

        if cfg.gnn.skip_connection == 'affine':
            self.linear_skip = nn.Linear(in_channels, out_channels, bias=True)
        elif cfg.gnn.skip_connection == 'identity':
            assert self.in_channels == self.out_channels

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        zeros(self.bias)
        self.cached_result = None
        self.cached_num_edges = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=False,
             dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None, edge_feature=None):
        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}. Please '
                    'disable the caching behavior of this layer by removing '
                    'the `cached=True` argument in its constructor.'.format(
                        self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            if self.normalize:
                edge_index, norm = self.norm(edge_index, x.size(self.node_dim),
                                             edge_weight, self.improved,
                                             x.dtype)
            else:
                norm = edge_weight
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result
        if cfg.gnn.skip_connection == 'affine':
            skip_x = self.linear_skip(x)
        elif cfg.gnn.skip_connection == 'identity':
            skip_x = x
        else:
            skip_x = 0
        return self.propagate(edge_index, x=x, norm=norm,
                              edge_feature=edge_feature) + skip_x

    def message(self, x_i, x_j, norm, edge_feature):
        if self.msg_direction == 'both':
            x_j = torch.cat((x_i, x_j, edge_feature), dim=-1)
        elif self.msg_direction == 'single':
            x_j = torch.cat((x_j, edge_feature), dim=-1)
        else:
            raise ValueError
        x_j = self.linear_msg(x_j)
        return norm.view(-1, 1) * x_j if norm is not None else x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class ResidualEdgeConv(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(ResidualEdgeConv, self).__init__()
        self.model = ResidualEdgeConvLayer(dim_in, dim_out, bias=bias)

    def forward(self, batch):
        batch.node_feature = self.model(batch.node_feature, batch.edge_index,
                                        edge_feature=batch.edge_feature)
        return batch


class TGCN(torch.nn.Module):
    r"""An implementation of the Temporal Graph Convolutional Gated Recurrent Cell.
    For details see this paper: `"T-GCN: A Temporal Graph ConvolutionalNetwork for
    Traffic Prediction." <https://arxiv.org/abs/1811.05320>`_
    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        improved (bool): Stronger self loops. Default is True.
        cached (bool): Caching the message weights. Default is False.
        add_self_loops (bool): Adding self-loops for smoothing. Default is True.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 improved: bool = False, cached: bool = False,
                 id: int = -1):
        super(TGCN, self).__init__()
        assert id >= 0, 'kwarg id is required.'

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.id = id

        self.graph_conv1 = GCNConv(self.in_channels + self.out_channels,
                                   self.out_channels * 2,
                                   improved=self.improved,
                                   cached=self.cached,
                                   normalize=True,
                                   bias=True,
                                   add_self_loops=True)
        # NOTE: possible issues here, by forcefully setting parameters.
        # but the original TGCN implementation initialized bias to ones.
        
        self.graph_conv1.bias.data = torch.ones_like(self.graph_conv1.bias.data)

        self.graph_conv2 = GCNConv(self.in_channels + self.out_channels,
                                   self.out_channels,
                                   improved=self.improved,
                                   cached=self.cached,
                                   normalize=True,
                                   bias=True,
                                   add_self_loops=True)

        # self._create_parameters_and_layers()
    #
    # def _create_update_gate_parameters_and_layers(self):
    #     self.conv_z = GCNConv(in_channels=self.in_channels,
    #                           out_channels=self.out_channels,
    #                           improved=self.improved,
    #                           cached=self.cached,
    #                           normalize=True,
    #                           bias=True,
    #                           add_self_loops=True)
    #
    #     self.linear_z = torch.nn.Linear(2 * self.out_channels,
    #                                     self.out_channels)
    #
    # def _create_reset_gate_parameters_and_layers(self):
    #     self.conv_r = GCNConv(in_channels=self.in_channels,
    #                           out_channels=self.out_channels,
    #                           improved=self.improved,
    #                           cached=self.cached,
    #                           normalize=True,
    #                           bias=True,
    #                           add_self_loops=True)
    #
    #     self.linear_r = torch.nn.Linear(2 * self.out_channels,
    #                                     self.out_channels)
    #
    # def _create_candidate_state_parameters_and_layers(self):
    #     self.conv_h = GCNConv(in_channels=self.in_channels,
    #                           out_channels=self.out_channels,
    #                           improved=self.improved,
    #                           cached=self.cached,
    #                           normalize=True,
    #                           add_self_loops=True)
    #
    #     self.linear_h = torch.nn.Linear(2 * self.out_channels,
    #                                     self.out_channels)
    #
    # def _create_parameters_and_layers(self):
    #     self._create_update_gate_parameters_and_layers()
    #     self._create_reset_gate_parameters_and_layers()
    #     self._create_candidate_state_parameters_and_layers()
    #
    # def _set_hidden_state(self, X, H):
    #     if not isinstance(H, torch.Tensor):
    #         H = torch.zeros(X.shape[0], self.out_channels).to(X.device)
    #     return H

    # def _calculate_update_gate(self, X, edge_index, edge_weight, H):
    #     Z = torch.cat([self.conv_z(X, edge_index, edge_weight), H], axis=1)
    #     Z = self.linear_z(Z)
    #     Z = torch.sigmoid(Z)
    #     return Z
    #
    # def _calculate_reset_gate(self, X, edge_index, edge_weight, H):
    #     R = torch.cat([self.conv_r(X, edge_index, edge_weight), H], axis=1)
    #     R = self.linear_r(R)
    #     R = torch.sigmoid(R)
    #     return R
    #
    # def _calculate_candidate_state(self, X, edge_index, edge_weight, H, R):
    #     H_tilde = torch.cat([self.conv_h(X, edge_index, edge_weight), H * R],
    #                         axis=1)
    #     H_tilde = self.linear_h(H_tilde)
    #     H_tilde = torch.tanh(H_tilde)
    #     return H_tilde
    #
    # def _calculate_hidden_state(self, Z, H, H_tilde):
    #     H = Z * H + (1 - Z) * H_tilde
    #     return H

    def _forward(self, X: torch.FloatTensor, edge_index: torch.LongTensor,
                 edge_weight: torch.FloatTensor = None,
                 H: torch.FloatTensor = None) -> torch.FloatTensor:
        # breakpoint()
        if not isinstance(H, torch.Tensor):
            H = torch.zeros(X.shape[0], self.out_channels).to(X.device)
        # print('H:', H.shape)
        concatenation = torch.sigmoid(
            self.graph_conv1(torch.cat([X, H], dim=1), edge_index, edge_weight)
        )
        # print('concatenation:', concatenation.shape)
        # r = concatenation[:, :self.out_channels]
        # u = concatenation[:, self.out_channels:]
        r, u = torch.chunk(concatenation, chunks=2, dim=1)
        # print('r:', r.shape)
        # print('u:', u.shape)
        c = torch.tanh(self.graph_conv2(
            torch.cat([X, H * r], dim=1), edge_index, edge_weight
        ))
        # print('c:', c.shape)
        H = u * H + (1.0 - u) * c
        # breakpoint()
        return H

    def forward(self, batch):
        if hasattr(batch, 'edge_weight'):
            edge_weight = batch.edge_weight
        else:
            edge_weight = None

        H = self._forward(X=batch.node_feature, edge_index=batch.edge_index,
                          edge_weight=edge_weight,
                          H=batch.node_states[self.id])
        batch.node_states[self.id] = H
        batch.node_feature = H
        return batch


class GConvGRULayer(nn.Module):
    r"""An implementation of the Chebyshev Graph Convolutional Gated Recurrent Unit
    Cell. For details see this paper: `"Structured Sequence Modeling with Graph
    Convolutional Recurrent Networks." <https://arxiv.org/abs/1612.07659>`_

    Adapted from torch_geometric_temporal.nn.recurrent.gconv_gru.GConvGRU.
    """

    def __init__(self, in_channels: int, out_channels: int, K: int = 7,
                 normalization: str = "sym", id: int = -1, bias: bool = True):
        super(GConvGRULayer, self).__init__()
        assert id >= 0, 'kwarg id is required.'

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.normalization = normalization
        self.bias = bias
        self._create_parameters_and_layers()
        self.id = id

    def _create_update_gate_parameters_and_layers(self):
        self.conv_x_z = ChebConv(in_channels=self.in_channels,
                                 out_channels=self.out_channels,
                                 K=self.K,
                                 normalization=self.normalization,
                                 bias=self.bias)

        self.conv_h_z = ChebConv(in_channels=self.out_channels,
                                 out_channels=self.out_channels,
                                 K=self.K,
                                 normalization=self.normalization,
                                 bias=self.bias)

    def _create_reset_gate_parameters_and_layers(self):
        self.conv_x_r = ChebConv(in_channels=self.in_channels,
                                 out_channels=self.out_channels,
                                 K=self.K,
                                 normalization=self.normalization,
                                 bias=self.bias)

        self.conv_h_r = ChebConv(in_channels=self.out_channels,
                                 out_channels=self.out_channels,
                                 K=self.K,
                                 normalization=self.normalization,
                                 bias=self.bias)

    def _create_candidate_state_parameters_and_layers(self):
        self.conv_x_h = ChebConv(in_channels=self.in_channels,
                                 out_channels=self.out_channels,
                                 K=self.K,
                                 normalization=self.normalization,
                                 bias=self.bias)

        self.conv_h_h = ChebConv(in_channels=self.out_channels,
                                 out_channels=self.out_channels,
                                 K=self.K,
                                 normalization=self.normalization,
                                 bias=self.bias)

    def _create_parameters_and_layers(self):
        self._create_update_gate_parameters_and_layers()
        self._create_reset_gate_parameters_and_layers()
        self._create_candidate_state_parameters_and_layers()

    def _set_hidden_state(self, X, H):
        if not isinstance(H, torch.Tensor):
            H = torch.zeros(X.shape[0], self.out_channels).to(X.device)
        return H

    def _calculate_update_gate(self, X, edge_index, edge_weight, H):
        Z = self.conv_x_z(X, edge_index, edge_weight)
        Z = Z + self.conv_h_z(H, edge_index, edge_weight)
        Z = torch.sigmoid(Z)  # (num_nodes, hidden_dim)
        return Z

    def _calculate_reset_gate(self, X, edge_index, edge_weight, H):
        R = self.conv_x_r(X, edge_index, edge_weight)
        R = R + self.conv_h_r(H, edge_index, edge_weight)
        R = torch.sigmoid(R)  # (num_nodes, hidden_dim)
        return R

    def _calculate_candidate_state(self, X, edge_index, edge_weight, H, R):
        H_tilde = self.conv_x_h(X, edge_index, edge_weight)
        H_tilde = H_tilde + self.conv_h_h(H * R, edge_index, edge_weight)
        H_tilde = torch.tanh(H_tilde)
        return H_tilde  # (num_nodes, hidden_dim)

    def _calculate_hidden_state(self, Z, H, H_tilde):
        H = Z * H + (1 - Z) * H_tilde
        return H  # (num_nodes, hidden_dim)

    def forward(self, batch):
        # X = raw input feature from pre_mp if self.id == 0,
        # otherwise, X = the hidden state from previous layer.
        X = batch.node_feature
        edge_index = batch.edge_index
        if hasattr(batch, 'edge_weight'):
            edge_weight = batch.edge_weight
        else:
            edge_weight = None
        H = batch.node_states[self.id]

        H = self._set_hidden_state(X, H)
        Z = self._calculate_update_gate(X, edge_index, edge_weight, H)
        R = self._calculate_reset_gate(X, edge_index, edge_weight, H)
        H_tilde = self._calculate_candidate_state(X, edge_index, edge_weight,
                                                  H, R)
        H = self._calculate_hidden_state(Z, H, H_tilde)

        batch.node_states[self.id] = H
        batch.node_feature = H
        return batch


class GConvLSTM(torch.nn.Module):
    r"""An implementation of the Chebyshev Graph Convolutional Long Short Term Memory
    Cell. For details see this paper: `"Structured Sequence Modeling with Graph
    Convolutional Recurrent Networks." <https://arxiv.org/abs/1612.07659>`_
    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        K (int): Chebyshev filter size :math:`K`.
        normalization (str, optional): The normalization scheme for the graph
            Laplacian (default: :obj:`"sym"`):
            1. :obj:`None`: No normalization
            :math:`\mathbf{L} = \mathbf{D} - \mathbf{A}`
            2. :obj:`"sym"`: Symmetric normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A}
            \mathbf{D}^{-1/2}`
            3. :obj:`"rw"`: Random-walk normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1} \mathbf{A}`
            You need to pass :obj:`lambda_max` to the :meth:`forward` method of
            this operator in case the normalization is non-symmetric.
            :obj:`\lambda_max` should be a :class:`torch.Tensor` of size
            :obj:`[num_graphs]` in a mini-batch scenario and a
            scalar/zero-dimensional tensor when operating on single graphs.
            You can pre-compute :obj:`lambda_max` via the
            :class:`torch_geometric.transforms.LaplacianLambdaMax` transform.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self, in_channels: int, out_channels: int, K: int = 7,
                 normalization: str = "sym", id: int = -1, bias: bool = True):
        super(GConvLSTM, self).__init__()
        assert id >= 0, 'kwarg id is required.'

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.normalization = normalization
        self.bias = bias
        self._create_parameters_and_layers()
        self._set_parameters()
        self.id = id

    def _create_input_gate_parameters_and_layers(self):

        self.conv_x_i = ChebConv(in_channels=self.in_channels,
                                 out_channels=self.out_channels,
                                 K=self.K,
                                 normalization=self.normalization,
                                 bias=self.bias)

        self.conv_h_i = ChebConv(in_channels=self.out_channels,
                                 out_channels=self.out_channels,
                                 K=self.K,
                                 normalization=self.normalization,
                                 bias=self.bias)

        self.w_c_i = Parameter(torch.Tensor(1, self.out_channels))
        self.b_i = Parameter(torch.Tensor(1, self.out_channels))

    def _create_forget_gate_parameters_and_layers(self):

        self.conv_x_f = ChebConv(in_channels=self.in_channels,
                                 out_channels=self.out_channels,
                                 K=self.K,
                                 normalization=self.normalization,
                                 bias=self.bias)

        self.conv_h_f = ChebConv(in_channels=self.out_channels,
                                 out_channels=self.out_channels,
                                 K=self.K,
                                 normalization=self.normalization,
                                 bias=self.bias)

        self.w_c_f = Parameter(torch.Tensor(1, self.out_channels))
        self.b_f = Parameter(torch.Tensor(1, self.out_channels))

    def _create_cell_state_parameters_and_layers(self):

        self.conv_x_c = ChebConv(in_channels=self.in_channels,
                                 out_channels=self.out_channels,
                                 K=self.K,
                                 normalization=self.normalization,
                                 bias=self.bias)

        self.conv_h_c = ChebConv(in_channels=self.out_channels,
                                 out_channels=self.out_channels,
                                 K=self.K,
                                 normalization=self.normalization,
                                 bias=self.bias)

        self.b_c = Parameter(torch.Tensor(1, self.out_channels))

    def _create_output_gate_parameters_and_layers(self):

        self.conv_x_o = ChebConv(in_channels=self.in_channels,
                                 out_channels=self.out_channels,
                                 K=self.K,
                                 normalization=self.normalization,
                                 bias=self.bias)

        self.conv_h_o = ChebConv(in_channels=self.out_channels,
                                 out_channels=self.out_channels,
                                 K=self.K,
                                 normalization=self.normalization,
                                 bias=self.bias)

        self.w_c_o = Parameter(torch.Tensor(1, self.out_channels))
        self.b_o = Parameter(torch.Tensor(1, self.out_channels))

    def _create_parameters_and_layers(self):
        self._create_input_gate_parameters_and_layers()
        self._create_forget_gate_parameters_and_layers()
        self._create_cell_state_parameters_and_layers()
        self._create_output_gate_parameters_and_layers()

    def _set_parameters(self):
        glorot(self.w_c_i)
        glorot(self.w_c_f)
        glorot(self.w_c_o)
        zeros(self.b_i)
        zeros(self.b_f)
        zeros(self.b_c)
        zeros(self.b_o)

    def _set_hidden_state(self, X, H):
        if not isinstance(H, torch.Tensor):
            H = torch.zeros(X.shape[0], self.out_channels).to(X.device)
        return H

    def _set_cell_state(self, X, C):
        if not isinstance(C, torch.Tensor):
            C = torch.zeros(X.shape[0], self.out_channels).to(X.device)
        return C

    def _calculate_input_gate(self, X, edge_index, edge_weight, H, C):
        I = self.conv_x_i(X, edge_index, edge_weight)
        I = I + self.conv_h_i(H, edge_index, edge_weight)
        I = I + (self.w_c_i * C)
        I = I + self.b_i
        I = torch.sigmoid(I)
        return I

    def _calculate_forget_gate(self, X, edge_index, edge_weight, H, C):
        F = self.conv_x_f(X, edge_index, edge_weight)
        F = F + self.conv_h_f(H, edge_index, edge_weight)
        F = F + (self.w_c_f * C)
        F = F + self.b_f
        F = torch.sigmoid(F)
        return F

    def _calculate_cell_state(self, X, edge_index, edge_weight, H, C, I, F):
        T = self.conv_x_c(X, edge_index, edge_weight)
        T = T + self.conv_h_c(H, edge_index, edge_weight)
        T = T + self.b_c
        T = torch.tanh(T)
        C = F * C + I * T
        return C

    def _calculate_output_gate(self, X, edge_index, edge_weight, H, C):
        O = self.conv_x_o(X, edge_index, edge_weight)
        O = O + self.conv_h_o(H, edge_index, edge_weight)
        O = O + (self.w_c_o * C)
        O = O + self.b_o
        O = torch.sigmoid(O)
        return O

    def _calculate_hidden_state(self, O, C):
        H = O * torch.tanh(C)
        return H

    def _forward(self, X: torch.FloatTensor, edge_index: torch.LongTensor,
                 edge_weight: torch.FloatTensor = None,
                 H: torch.FloatTensor = None,
                 C: torch.FloatTensor = None
                 ) -> (torch.FloatTensor, torch.FloatTensor):
        """
        Making a forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph. If the hidden state and cell state
        matrices are not present when the forward pass is called these are
        initialized with zeros.
        Arg types:
            * **X** *(PyTorch Float Tensor)* - Node features.
            * **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
            * **edge_weight** *(PyTorch Long Tensor, optional)* - Edge weight vector.
            * **H** *(PyTorch Float Tensor, optional)* - Hidden state matrix for all nodes.
            * **C** *(PyTorch Float Tensor, optional)* - Cell state matrix for all nodes.
        Return types:
            * **H** *(PyTorch Float Tensor)* - Hidden state matrix for all nodes.
            * **C** *(PyTorch Float Tensor)* - Cell state matrix for all nodes.
        """
        H = self._set_hidden_state(X, H)
        C = self._set_cell_state(X, C)
        I = self._calculate_input_gate(X, edge_index, edge_weight, H, C)
        F = self._calculate_forget_gate(X, edge_index, edge_weight, H, C)
        C = self._calculate_cell_state(X, edge_index, edge_weight, H, C, I, F)
        O = self._calculate_output_gate(X, edge_index, edge_weight, H, C)
        H = self._calculate_hidden_state(O, C)
        return H, C

    def forward(self, batch):
        if hasattr(batch, 'edge_weight'):
            edge_weight = batch.edge_weight
        else:
            edge_weight = None

        H, C = self._forward(X=batch.node_feature,
                             edge_index=batch.edge_index,
                             edge_weight=edge_weight,
                             H=batch.node_states[self.id],
                             C=batch.node_cells[self.id])

        batch.node_states[self.id] = H
        batch.node_cells[self.id] = C
        batch.node_feature = H
        return batch


class EvolveGCNH(torch.nn.Module):
    r"""An implementation of the Evolving Graph Convolutional Hidden Layer.
    For details see this paper: `"EvolveGCN: Evolving Graph Convolutional
    Networks for Dynamic Graph." <https://arxiv.org/abs/1902.10191>`_
    Args:
        num_of_nodes (int): Number of vertices.
        in_channels (int): Number of filters.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        normalize (bool, optional): Whether to add self-loops and apply
            symmetric normalization. (default: :obj:`True`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
    """

    def __init__(self, in_channels: int, out_channels: int, id: bool = -1,
                 improved: bool = False,
                 cached: bool = False, normalize: bool = True,
                 add_self_loops: bool = True,
                 bias: bool = True):
        super(EvolveGCNH, self).__init__()
        self.num_of_nodes = cfg.dataset.num_nodes
        self.in_channels = in_channels
        self.improved = improved
        self.cached = cached
        self.normalize = normalize
        self.add_self_loops = add_self_loops
        self.id = id
        self._create_layers()
        std = 1. / math.sqrt(in_channels)
        self.conv_layer.lin.weight.data.uniform_(-std, std)

    def _create_layers(self):
        self.ratio = self.in_channels / self.num_of_nodes

        self.pooling_layer = TopKPooling(self.in_channels, self.ratio)

        self.recurrent_layer = GRU(input_size=self.in_channels,
                                   hidden_size=self.in_channels,
                                   num_layers=1)

        self.conv_layer = GCNConv(in_channels=self.in_channels,
                                  out_channels=self.in_channels,
                                  improved=self.improved,
                                  cached=self.cached,
                                  normalize=self.normalize,
                                  add_self_loops=self.add_self_loops,
                                  bias=True)

    def _forward(self, X: torch.FloatTensor, edge_index: torch.LongTensor,
                 edge_weight: torch.FloatTensor = None) -> torch.FloatTensor:
        """
        Making a forward pass.
        Arg types:
            * **X** *(PyTorch Float Tensor)* - Node embedding.
            * **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
            * **edge_weight** *(PyTorch Float Tensor, optional)* - Edge weight vector.
        Return types:
            * **X** *(PyTorch Float Tensor)* - Output matrix for all nodes.
        """
        # X: (num_nodes, dim_inner).
        # X_tilde = self.pooling_layer(X, edge_index)
        # X_tilde = X_tilde[0][None, :, :]  # (dim_inner, dim_inner)
        # W = self.conv_layer.lin.weight[None, :, :]  # (dim_inner, dim_inner)
        X_tilde = self.pooling_layer(X, edge_index)[0].detach().clone().unsqueeze(0)
        W = self.conv_layer.lin.weight.detach().clone().unsqueeze(0)
        X_tilde, W = self.recurrent_layer(X_tilde, W)
        # self.conv_layer.lin.weight = torch.nn.Parameter(W.squeeze())
        self.conv_layer.lin.weight.data = W.squeeze().clone()
        X = self.conv_layer(X, edge_index, edge_weight)
        return X

    def forward(self, batch):
        if hasattr(batch, 'edge_weight'):
            edge_weight = batch.edge_weight
        else:
            edge_weight = None
        H = self._forward(X=batch.node_feature,
                          edge_index=batch.edge_index,
                          edge_weight=edge_weight)
        batch.node_states[self.id] = H
        batch.node_feature = H
        return batch


class EvolveGCNO(torch.nn.Module):
    """
    The O-version of Evolve GCN, the H-version is too restricted, and the
    transaction graph is more about constructing meaningful embeddings from
    graph structure, initial node features are not that important.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 improved: bool = False,
                 cached: bool = False, normalize: bool = True,
                 add_self_loops: bool = True,
                 bias: bool = False,
                 id: int = -1):
        """
        NOTE: EvolveGCNO does not change size of representation,
            i.e., out_channels == in_channels.
        This can be easily modified, but not necessary in the ROLAND use case
            as we have out_channels == in_channels == inner_dim.
        """
        super(EvolveGCNO, self).__init__()
        assert id >= 0, 'kwarg id is required.'

        self.in_channels = in_channels
        assert in_channels == out_channels
        self.improved = improved
        self.cached = cached
        self.normalize = normalize
        self.add_self_loops = add_self_loops
        self.id = id
        self._create_layers()
        std = 1. / math.sqrt(in_channels)
        self.conv_layer.lin.weight.data.uniform_(-std, std)

    def _create_layers(self):
        # self.recurrent_layer = GRU(input_size=self.in_channels,
        #                            hidden_size=self.in_channels,
        #                            num_layers=1)

        # self.update_gate = nn.Sequential(
        #     nn.Linear(2 * self.in_channels, self.in_channels, bias=True),
        #     nn.Sigmoid()
        # )

        # self.reset_gate = nn.Sequential(
        #     nn.Linear(2 * self.in_channels, self.in_channels, bias=True),
        #     nn.Sigmoid()
        # )

        # self.h_tilde = nn.Sequential(
        #     nn.Linear(2 * self.in_channels, self.in_channels),
        #     nn.Tanh()
        # )

        self.update = mat_GRU_gate(self.in_channels,
                                   self.in_channels,
                                   torch.nn.Sigmoid())

        self.reset = mat_GRU_gate(self.in_channels,
                                  self.in_channels,
                                  torch.nn.Sigmoid())

        self.htilda = mat_GRU_gate(self.in_channels,
                                   self.in_channels,
                                   torch.nn.Tanh())

        self.conv_layer = GCNConv(in_channels=self.in_channels,
                                  out_channels=self.in_channels,
                                  improved=self.improved,
                                  cached=self.cached,
                                  normalize=self.normalize,
                                  add_self_loops=self.add_self_loops,
                                  bias=True)

    def _forward(self, X: torch.FloatTensor, edge_index: torch.LongTensor,
                 edge_weight: torch.FloatTensor = None) -> torch.FloatTensor:
        """
        Making a forward pass.
        Arg types:
            * **X** *(PyTorch Float Tensor)* - Node embedding.
            * **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
            * **edge_weight** *(PyTorch Float Tensor, optional)* - Edge weight vector.
        Return types:
            * **X** *(PyTorch Float Tensor)* - Output matrix for all nodes.
        """
        # W = self.conv_layer.lin.weight[None, :, :].detach().clone()
        # # W has shape (1, inner_dim, inner_dim),
        # # corresponds to (seq_len, batch, input_size).
        # W, _ = self.recurrent_layer(W, W.clone())
        # self.conv_layer.lin.weight = torch.nn.Parameter(W.squeeze())
        # # breakpoint()
        W = self.conv_layer.lin.weight.detach().clone()
        # update = self.update_gate(torch.cat((W, W), axis=1))
        # reset = self.reset_gate(torch.cat((W, W), axis=1))
        # h_tilde = self.h_tilde(torch.cat((W, reset * W), axis=1))
        # W = (1 - update) * W + update * h_tilde

        update = self.update(W, W)
        reset = self.reset(W, W)

        h_cap = reset * W
        h_cap = self.htilda(W, h_cap)

        new_W = (1 - update) * W + update * h_cap

        self.conv_layer.lin.weight.data = new_W.clone()
        X = self.conv_layer(X, edge_index, edge_weight)
        return X

    def forward(self, batch):
        if hasattr(batch, 'edge_weight'):
            edge_weight = batch.edge_weight
        else:
            edge_weight = None
        out = self._forward(batch.node_feature, batch.edge_index, edge_weight)
        # For consistency with the training pipeline, node_states are not
        # used in this model.
        batch.node_states[self.id] = out
        batch.node_feature = out
        return batch


class mat_GRU_gate(torch.nn.Module):
    def __init__(self, rows, cols, activation):
        super().__init__()
        self.activation = activation
        # the k here should be in_feats which is actually the rows
        self.W = nn.Parameter(torch.Tensor(rows, rows))
        self.reset_param(self.W)

        self.U = nn.Parameter(torch.Tensor(rows, rows))
        self.reset_param(self.U)

        self.bias = nn.Parameter(torch.zeros(rows, cols))

    def reset_param(self, t):
        # Initialize based on the number of columns
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv, stdv)

    def forward(self, x, hidden):
        out = self.activation(self.W.matmul(x) +
                              self.U.matmul(hidden) +
                              self.bias)

        return out


layer_dict = {
    'linear': Linear,
    'mlp': MLP,
    # 'gcnconv': GCNConv,
    'sageconv': SAGEConv,
    'gatconv': GATConv,
    'splineconv': SplineConv,
    'ginconv': GINConv,
    'generalconv': GeneralConv,
    'generaledgeconv': GeneralEdgeConv,
    'generalsampleedgeconv': GeneralSampleEdgeConv,
    'residual_edge_conv': ResidualEdgeConv,
    'tgcn': TGCN,
    'gconv_gru': GConvGRULayer,
    'gconv_lstm': GConvLSTM,
    'evolve_gcn_h': EvolveGCNH,
    'evolve_gcn_o': EvolveGCNO
}
