import networkx as nx
import time
import pickle

from deepsnap.dataset import GraphDataset
import torch
from torch.utils.data import DataLoader
from config import cfg
from deepsnap.batch import Batch

import os
from typing import List, Union

import deepsnap
import numpy as np
import pandas as pd
from deepsnap.graph import Graph
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder


def load_node_classification_dataset(dataset_dir: str, label_dir: str) -> Graph:
    df_trans = pd.read_csv(dataset_dir, sep=',', header=None, index_col=None)
    df_trans.columns = ['SOURCE', 'TARGET', 'RATING', 'TIME']
    # NOTE: 'SOURCE' and 'TARGET' are not consecutive.
    num_nodes = len(pd.unique(df_trans[['SOURCE', 'TARGET']].to_numpy().ravel()))

    # bitcoin OTC contains decimal numbers, round them.
    df_trans['TIME'] = df_trans['TIME'].astype(np.int).astype(np.float)
    assert not np.any(pd.isna(df_trans).values)

    time_scaler = MinMaxScaler((0, 2))
    df_trans['TimestampScaled'] = time_scaler.fit_transform(
        df_trans['TIME'].values.reshape(-1, 1))

    edge_feature = torch.Tensor(
        df_trans[['RATING', 'TimestampScaled']].values)  # (E, edge_dim)
    # SOURCE and TARGET IDs are already encoded in the csv file.
    # edge_index = torch.Tensor(
    #     df_trans[['SOURCE', 'TARGET']].values.transpose()).long()  # (2, E)

    node_indices = np.sort(pd.unique(df_trans[['SOURCE', 'TARGET']].to_numpy().ravel()))
    enc = OrdinalEncoder(categories=[node_indices, node_indices])
    raw_edges = df_trans[['SOURCE', 'TARGET']].values

    edge_index = enc.fit_transform(raw_edges).transpose()
    edge_index = torch.LongTensor(edge_index)

    # num_nodes = torch.max(edge_index) + 1
    # Use dummy node features.
    # node_feature = torch.ones(num_nodes, 1).float()

    _, node_feature = torch.unique(edge_index, return_counts=True)
    node_feature = node_feature.unsqueeze(-1).float()

    # Read node labels
    node_label = np.arange(num_nodes)
    with open(label_dir, 'r') as f:
        for line in f:
            l = line.strip('\n\r').split(',')
            idx = eval(l[0])
            label = eval(l[1])
            node_label[idx] = label
    
    node_label = torch.LongTensor(node_label)

    edge_time = torch.FloatTensor(df_trans['TIME'].values)

    if cfg.train.mode in ['live_update_fixed_split']:
        edge_feature = torch.cat((edge_feature, edge_feature.clone()), dim=0)
        reversed_idx = torch.stack([edge_index[1], edge_index[0]]).clone()
        edge_index = torch.cat((edge_index, reversed_idx), dim=1)
        edge_time = torch.cat((edge_time, edge_time.clone()))

    graph = Graph(
        node_feature=node_feature,
        node_label=node_label,
        edge_feature=edge_feature,
        edge_index=edge_index,
        edge_time=edge_time,
        directed=True
    )
    return graph


def load_single_dataset(dataset_dir: str) -> Graph:
    print(dataset_dir)
    df_trans = pd.read_csv(dataset_dir, sep=',', header=None, index_col=None)
    df_trans.columns = ['SOURCE', 'TARGET', 'RATING', 'TIME']
    # NOTE: 'SOURCE' and 'TARGET' are not consecutive.
    num_nodes = len(pd.unique(df_trans[['SOURCE', 'TARGET']].to_numpy().ravel()))

    # bitcoin OTC contains decimal numbers, round them.
    df_trans['TIME'] = df_trans['TIME'].astype(np.int).astype(np.float)
    assert not np.any(pd.isna(df_trans).values)

    time_scaler = MinMaxScaler((0, 2))
    df_trans['TimestampScaled'] = time_scaler.fit_transform(
        df_trans['TIME'].values.reshape(-1, 1))

    edge_feature = torch.Tensor(
        df_trans[['RATING', 'TimestampScaled']].values)  # (E, edge_dim)
    # SOURCE and TARGET IDs are already encoded in the csv file.
    # edge_index = torch.Tensor(
    #     df_trans[['SOURCE', 'TARGET']].values.transpose()).long()  # (2, E)

    node_indices = np.sort(pd.unique(df_trans[['SOURCE', 'TARGET']].to_numpy().ravel()))
    enc = OrdinalEncoder(categories=[node_indices, node_indices])
    raw_edges = df_trans[['SOURCE', 'TARGET']].values
    edge_index = enc.fit_transform(raw_edges).transpose()
    edge_index = torch.LongTensor(edge_index)

    # num_nodes = torch.max(edge_index) + 1
    # Use dummy node features.
    node_feature = torch.ones(num_nodes, 1).float()

    edge_time = torch.FloatTensor(df_trans['TIME'].values)

    graph = Graph(
        node_feature=node_feature,
        edge_feature=edge_feature,
        edge_index=edge_index,
        edge_time=edge_time,
        directed=True
    )
    return graph


def make_graph_snapshot(g_all: Graph, snapshot_freq: str) -> List[Graph]:
    t = g_all.edge_time.numpy().astype(np.int64)
    snapshot_freq = snapshot_freq.upper()

    period_split = pd.DataFrame(
        {'Timestamp': t,
         'TransactionTime': pd.to_datetime(t, unit='s')},
        index=range(len(g_all.edge_time)))

    freq_map = {'D': '%j',  # day of year.
                'W': '%W',  # week of year.
                'M': '%m'  # month of year.
                }

    period_split['Year'] = period_split['TransactionTime'].dt.strftime(
        '%Y').astype(int)

    period_split['SubYearFlag'] = period_split['TransactionTime'].dt.strftime(
        freq_map[snapshot_freq]).astype(int)

    period2id = period_split.groupby(['Year', 'SubYearFlag']).indices

    periods = sorted(list(period2id.keys()))
    snapshot_list = list()

    for p in periods:
        # unique IDs of edges in this period.
        period_members = period2id[p]
        assert np.all(period_members == np.unique(period_members))

        if g_all.node_label == None:
            g_incr = Graph(
                node_feature=g_all.node_feature,
                edge_feature=g_all.edge_feature[period_members, :],
                edge_index=g_all.edge_index[:, period_members],
                edge_time=g_all.edge_time[period_members],
                directed=g_all.directed
            )
        else:
            g_incr = Graph(
                node_feature=g_all.node_feature,
                node_label=g_all.node_label,
                edge_feature=g_all.edge_feature[period_members, :],
                edge_index=g_all.edge_index[:, period_members],
                edge_time=g_all.edge_time[period_members],
                directed=g_all.directed
            )
        snapshot_list.append(g_incr)

    snapshot_list.sort(key=lambda x: torch.min(x.edge_time))

    return snapshot_list


def split_by_seconds(g_all, freq_sec: int):
    # Split the entire graph into snapshots.
    split_criterion = g_all.edge_time // freq_sec
    groups = torch.sort(torch.unique(split_criterion))[0]
    snapshot_list = list()
    for t in groups:
        period_members = (split_criterion == t)
        
        if g_all.node_label == None:
            g_incr = Graph(
                node_feature=g_all.node_feature,
                edge_feature=g_all.edge_feature[period_members, :],
                edge_index=g_all.edge_index[:, period_members],
                edge_time=g_all.edge_time[period_members],
                directed=g_all.directed
            )
        else:
            g_incr = Graph(
                node_feature=g_all.node_feature,
                node_label=g_all.node_label,
                edge_feature=g_all.edge_feature[period_members, :],
                edge_index=g_all.edge_index[:, period_members],
                edge_time=g_all.edge_time[period_members],
                directed=g_all.directed
            )

        snapshot_list.append(g_incr)
    return snapshot_list


def load_generic(dataset_dir: str, node_label,
                 snapshot: bool = True,
                 snapshot_freq: str = None
                 ) -> Union[deepsnap.graph.Graph,
                            List[deepsnap.graph.Graph]]:
    if node_label == None:
        g_all = load_single_dataset(dataset_dir)
    else:
        g_all = load_node_classification_dataset(dataset_dir, node_label)
    if not snapshot:
        return g_all

    if snapshot_freq.upper() not in ['D', 'W', 'M']:
        # format: '1200000s'
        # assume split by seconds (timestamp) as in EvolveGCN paper.
        freq = int(snapshot_freq.strip('s'))
        snapshot_list = split_by_seconds(g_all, freq)
    else:
        snapshot_list = make_graph_snapshot(g_all, snapshot_freq)
    num_nodes = g_all.edge_index.max() + 1

    for g_snapshot in snapshot_list:
        g_snapshot.node_states = [0 for _ in range(cfg.gnn.layers_mp)]
        g_snapshot.node_cells = [0 for _ in range(cfg.gnn.layers_mp)]
        g_snapshot.node_degree_existing = torch.zeros(num_nodes)

    # check snapshots ordering.
    prev_end = -1
    for g in snapshot_list:
        start, end = torch.min(g.edge_time), torch.max(g.edge_time)
        assert prev_end < start <= end
        prev_end = end

    return snapshot_list


def load_dataset():
    '''
    load raw datasets.
    :return: a list of networkx/deepsnap graphs, plus additional info if needed
    '''
    name = cfg.dataset.name
    dataset_dir = cfg.dataset.dir
    node_label = cfg.dataset.node_label
    # Try to load customized data format
    graphs = load_generic(os.path.join(dataset_dir, name), node_label, snapshot=cfg.transaction.snapshot, snapshot_freq=cfg.transaction.snapshot_freq)
    
    return graphs


def filter_graphs():
    '''
    Filter graphs by the min number of nodes
    :return: min number of nodes
    '''
    if cfg.dataset.task == 'graph':
        min_node = 0
    else:
        min_node = 5
    return min_node


def create_dataset():
    ## Load dataset
    time1 = time.time()
    graphs = load_dataset()

    min_node = filter_graphs()

    ## Create whole dataset
    dataset = GraphDataset(
        graphs,
        task=cfg.dataset.task,
        edge_train_mode=cfg.dataset.edge_train_mode,
        edge_message_ratio=cfg.dataset.edge_message_ratio,
        edge_negative_sampling_ratio=cfg.dataset.edge_negative_sampling_ratio,
        minimum_node_per_graph=min_node)
    
    ## Split dataset
    # Use custom data splits
    if cfg.dataset.split_method == 'chronological_temporal':
        if cfg.train.mode == 'live_update_fixed_split':
            datasets = [dataset, dataset, dataset]
        else:
            total = len(dataset)  # total number of snapshots.
            train_end = int(total * cfg.dataset.split[0])
            val_end = int(total * (cfg.dataset.split[0] + cfg.dataset.split[1]))
            datasets = [
                dataset[:train_end],
                dataset[train_end:val_end],
                dataset[val_end:]
            ]
    else:
        datasets = dataset.split(
                transductive=cfg.dataset.transductive,
                split_ratio=cfg.dataset.split,
                shuffle=cfg.dataset.shuffle)
    
    time2 = time.time()

    print('Loading dataset: {:.4}s'.format(time2 - time1))
    return datasets


def create_loader(datasets):
    loader_train = DataLoader(datasets[0], collate_fn=Batch.collate(),
                              batch_size=cfg.train.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=False)

    loaders = [loader_train]
    for i in range(1, len(datasets)):
        loaders.append(DataLoader(datasets[i], collate_fn=Batch.collate(),
                                  batch_size=cfg.train.batch_size,
                                  shuffle=False,
                                  num_workers=cfg.num_workers,
                                  pin_memory=False))

    return loaders
