"""
The more realistic training pipeline.
"""
import copy
import logging
import os
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import *


import deepsnap
import numpy as np
import torch
from config import cfg
import train_utils
from optimizer import create_optimizer, create_scheduler
from tqdm import tqdm


def node_degree(edge_index, n=None, mode='in'):
    if mode == 'in':
        index = edge_index[0, :]
    elif mode == 'out':
        index = edge_index[1, :]
    else:
        index = edge_index.flatten()
    n = edge_index.max() + 1 if n is None else n
    degree = torch.zeros(n)
    ones = torch.ones(index.shape[0])
    return degree.scatter_add_(0, index, ones)


@torch.no_grad()
def average_state_dict(dict1: dict, dict2: dict, weight: float) -> dict:
    # Average two model.state_dict() objects.
    # out = (1-w)*dict1 + w*dict2
    assert 0 <= weight <= 1
    d1 = copy.deepcopy(dict1)
    d2 = copy.deepcopy(dict2)
    out = dict()
    for key in d1.keys():
        assert isinstance(d1[key], torch.Tensor)
        param1 = d1[key].detach().clone()
        assert isinstance(d2[key], torch.Tensor)
        param2 = d2[key].detach().clone()
        out[key] = (1 - weight) * param1 + weight * param2
    return out


def precompute_edge_degree_info(dataset):
    """
    Pre-computes edge_degree_existing, edge_degree_new and keep ratio
    at each snapshot. Inplace modifications.
    """
    num_nodes = dataset[0].node_feature.shape[0]
    for t in tqdm(range(len(dataset)), desc='precompute edge deg info'):
        if t == 0:
            dataset[t].node_degree_existing = torch.zeros(num_nodes)
        else:
            dataset[t].node_degree_existing \
                = dataset[t - 1].node_degree_existing \
                  + dataset[t - 1].node_degree_new

        dataset[t].node_degree_new = node_degree(dataset[t].edge_index,
                                                 n=num_nodes)

        dataset[t].keep_ratio = train_utils.get_keep_ratio(
            existing=dataset[t].node_degree_existing,
            new=dataset[t].node_degree_new,
            mode=cfg.transaction.keep_ratio)
        dataset[t].keep_ratio = dataset[t].keep_ratio.unsqueeze(-1)


@torch.no_grad()
def get_task_batch(dataset: deepsnap.dataset.GraphDataset,
                   today: int, tomorrow: int,
                   prev_node_states: Optional[Dict[str, List[torch.Tensor]]]
                   ) -> deepsnap.graph.Graph:
    """
    Construct batch required for the task (today, tomorrow). As defined in
    batch's get_item method (used to get edge_label and get_label_index),
    edge_label and edge_label_index returned would be different everytime
    get_task_batch() is called.

    Moreover, copy node-memories (node_states and node_cells) to the batch.
    """
    assert today < tomorrow < len(dataset)
    # Get edges for message passing and prediction task.
    batch = dataset[today].clone()
    
    if cfg.dataset.task == 'link_pred':
        batch.edge_label = dataset[tomorrow].edge_label.clone()
        batch.edge_label_index = dataset[tomorrow].edge_label_index.clone()

    # Copy previous memory to the batch.
    if prev_node_states is not None:
        for key, val in prev_node_states.items():
            copied = [x.detach().clone() for x in val]
            setattr(batch, key, copied)

    batch = train_utils.move_batch_to_device(batch, cfg.device)
    return batch


@torch.no_grad()
def update_node_states(model, dataset, task: Tuple[int, int],
                       prev_node_states: Optional[
                           Dict[str, List[torch.Tensor]]]
                       ) -> Dict[str, List[torch.Tensor]]:
    """
    Perform the provided task and keep track of the latest node_states.

    Example: task = (t, t+1),
        the prev_node_states contains node embeddings at time (t-1).
        the model perform task (t, t+1):
            Input: (node embedding at t - 1, edges at t).
            Output: possible transactions at t+1.
        the model also generates node embeddings at t.

    after doing task (t, t+1), node_states contains information
    from snapshot t.
    """
    today, tomorrow = task
    batch = get_task_batch(dataset, today, tomorrow, prev_node_states).clone()
    # Let the model modify batch.node_states (and batch.node_cells).
    _, _ = model(batch)
    # Collect the updated node states.
    out = dict()
    out['node_states'] = [x.detach().clone() for x in batch.node_states]
    if isinstance(batch.node_cells[0], torch.Tensor):
        out['node_cells'] = [x.detach().clone() for x in batch.node_cells]

    return out


def train_step(model, optimizer, scheduler, dataset,
               task: Tuple[int, int],
               prev_node_states: Optional[Dict[str, torch.Tensor]]
               ) -> dict:
    """
    After receiving ground truth from a particular task, update the model by
    performing back-propagation.
    For example, on day t, the ground truth of task (t-1, t) has been revealed,
    train the model using G[t-1] for message passing and label[t] as target.
    """
    optimizer.zero_grad()
    torch.cuda.empty_cache()

    today, tomorrow = task
    model.train()
    batch = get_task_batch(dataset, today, tomorrow, prev_node_states).clone()

    pred, true = model(batch)
    # print(pred)
    # print(true)

    loss, pred_score = train_utils.compute_loss(pred, true)
    loss.backward()
    optimizer.step()

    scheduler.step()
    return {'loss': loss}


@torch.no_grad()
def evaluate_step(model, dataset, task: Tuple[int, int],
                  prev_node_states: Optional[Dict[str, List[torch.Tensor]]],
                  fast: bool = False) -> dict:
    """
    Evaluate model's performance on task = (today, tomorrow)
        where today and tomorrow are integers indexing snapshots.
    """
    today, tomorrow = task
    model.eval()
    batch = get_task_batch(dataset, today, tomorrow, prev_node_states).clone()

    pred, true = model(batch)
    loss, pred_score = train_utils.compute_loss(pred, true)

    if cfg.dataset.task == 'link_pred':
        micro_auc = roc_auc_score(y_true=true.detach().cpu().numpy(), y_score=pred_score.detach().cpu().numpy(), average='micro')
        macro_auc = roc_auc_score(y_true=true.detach().cpu().numpy(), y_score=pred_score.detach().cpu().numpy(), average='macro')

        pred_label = torch.zeros(len(pred_score))
        pred_label[pred_score >= 0.5] = 1.0
        accuracy = np.mean(true.detach().cpu().numpy() == pred_label.numpy())
        if fast:
            # skip MRR calculation for internal validation.
            return {'loss': loss.item(), 'micro_auc': micro_auc, 'macro_auc': macro_auc, 'accuracy': accuracy}

        mrr_batch = get_task_batch(dataset, today, tomorrow,
                               prev_node_states).clone()

        mrr, rck1, rck3, rck10 = train_utils.report_rank_based_eval(
            mrr_batch, model,
            num_neg_per_node=cfg.experimental.rank_eval_multiplier)

        return {'loss': loss.item(), 'mrr': mrr, 'rck1': rck1, 'rck3': rck3,
            'rck10': rck10, 'micro_auc': micro_auc, 'macro_auc': macro_auc, 'accuracy': accuracy}
    else:
        pred_int = pred_score.max(dim=1)[1]
        return {'micro f1-score':round(f1_score(true.detach().cpu().numpy(), pred_int.detach().cpu().numpy(), average='micro'), cfg.round), 
                'macro f1-score':round(f1_score(true.detach().cpu().numpy(), pred_int.detach().cpu().numpy(), average='macro'), cfg.round),
                'accuracy': round(accuracy_score(true.detach().cpu().numpy(), pred_int.detach().cpu().numpy()), cfg.round),
                'recall': round(recall_score(true.detach().cpu().numpy(), pred_int.detach().cpu().numpy(), average='macro'), cfg.round),
                'loss': loss.item()}


def train_live_update(loaders, model, optimizer, scheduler, datasets, **kwargs):
    for dataset in datasets:
        # Sometimes edge degree info is already included in dataset.
        if not hasattr(dataset[0], 'keep_ratio'):
            precompute_edge_degree_info(dataset)

    num_splits = 3  # train/val/test splits.
    # range for today in (today, tomorrow) task pairs.
    task_range = range(len(datasets[0]) - cfg.transaction.horizon)

    prev_node_states = None  # no previous state on day 0.
    # {'node_states': [Tensor, Tensor], 'node_cells: [Tensor, Tensor]}

    model_init = None  # for meta-learning only, a model.state_dict() object.

    # auc_hist = list()
    # mrr_hist = list()
    max_acc = 0
    temp = None
    for t in tqdm(task_range, desc='snapshot', leave=True):
        # current task: t --> t+1.
        # (1) Evaluate model's performance on this task, at this time, the
        # model has seen no information on t+1, this evaluation is fair.
        for i in range(1, num_splits):
            perf = evaluate_step(model, datasets[i], (t, t + 1), prev_node_states)
            if i == 2:
                print(perf)
                # print(t)
                # print(len(datasets[0]) - cfg.transaction.horizon)
                if perf['recall'] > max_acc:
                    temp = perf
                    max_acc = perf['recall'] 
                if t + 1 == len(datasets[0]) - cfg.transaction.horizon:
                    with open('results.txt', 'a+') as f:
                        contents = ''
                        for k,v in temp.items():
                            contents += k + ': ' + str(v) + ' '
                        contents += '\n'
                        f.writelines(contents) 
                # auc_hist.append(perf['micro_auc'])
                # mrr_hist.append(perf['mrr'])

        if t <= cfg.train.stop_live_update_after:
            # (2) Reveal the ground truth of task (t, t+1) and update the model
            # to prepare for the next task.
            del optimizer, scheduler  # use new optimizers.
            optimizer = create_optimizer(model.parameters())
            scheduler = create_scheduler(optimizer)

            # best model's validation loss, training epochs, and state_dict.
            best_model = {'val_loss': np.inf, 'train_epoch': 0, 'state': None}
            # keep track of how long we have NOT update the best model.
            best_model_unchanged = 0
            # after not updating the best model for `tol` epochs, stop.
            tol = cfg.train.internal_validation_tolerance

            # internal training loop (intra-snapshot cross-validation).
            # choose the best model using current validation set, prepare for
            # next task.

            if cfg.meta.is_meta and (model_init is not None):
                # For meta-learning, start fine-tuning from the pre-computed
                # initialization weight.
                model.load_state_dict(copy.deepcopy(model_init))

            for i in tqdm(range(cfg.optim.max_epoch + 1), desc='live update',
                        leave=True):
                # Start with the un-trained model (i = 0), evaluate the model.
                internal_val_perf = evaluate_step(model, datasets[1],
                                                (t, t + 1),
                                                prev_node_states, fast=True)
                # print(internal_val_perf)
                val_loss = internal_val_perf['loss']

                if val_loss < best_model['val_loss']:
                    # replace the best model with the current model.
                    best_model = {'val_loss': val_loss, 'train_epoch': i,
                                'state': copy.deepcopy(model.state_dict())}
                    best_model_unchanged = 0
                else:
                    # the current best model has dominated for these epochs.
                    best_model_unchanged += 1

                # if (i >= 2 * tol) and (best_model_unchanged >= tol):
                if best_model_unchanged >= tol:
                    # If the best model has not been updated for a while, stop.
                    break
                else:
                    # Otherwise, keep training.
                    train_perf = train_step(model, optimizer, scheduler,
                                            datasets[0], (t, t + 1),
                                            prev_node_states)

            # (3) Actually perform the update on training set to get node_states
            # contains information up to time t.
            # Use the best model selected from intra-snapshot cross-validation.
            model.load_state_dict(best_model['state'])

        if cfg.meta.is_meta:  # update meta-learning's initialization weights.
            if model_init is None:  # for the first task.
                model_init = copy.deepcopy(best_model['state'])
            else:  # for subsequent task, update init.
                if cfg.meta.method == 'moving_average':
                    new_weight = cfg.meta.alpha
                elif cfg.meta.method == 'online_mean':
                    new_weight = 1 / (t + 1)  # for t=1, the second item, 1/2.
                else:
                    raise ValueError(f'Invalid method: {cfg.meta.method}')

                # (1-new_weight)*model_init + new_weight*best_model.
                model_init = average_state_dict(model_init,
                                                best_model['state'],
                                                new_weight)

        prev_node_states = update_node_states(model, datasets[0], (t, t + 1),
                                              prev_node_states)
