import torch
from torch_scatter import scatter

# Pooling options (pool nodes into graph representations)
# pooling function takes in node embedding [num_nodes x emb_dim] and
# batch (indices) and outputs graph embedding [num_graphs x emb_dim].
def global_add_pool(x, batch, id=None, size=None):
    size = batch.max().item() + 1 if size is None else size
    return scatter(x, batch, dim=0, dim_size=size, reduce='add')


def global_mean_pool(x, batch, id=None, size=None):
    size = batch.max().item() + 1 if size is None else size
    return scatter(x, batch, dim=0, dim_size=size, reduce='mean')


def global_max_pool(x, batch, id=None, size=None):
    size = batch.max().item() + 1 if size is None else size
    return scatter(x, batch, dim=0, dim_size=size, reduce='max')


pooling_dict = {
    'add': global_add_pool,
    'mean': global_mean_pool,
    'max': global_max_pool
}
