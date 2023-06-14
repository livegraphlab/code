import os
import random
import numpy as np
import torch

from config import cfg, assert_cfg
from loader import create_dataset, create_loader
from optimizer import create_optimizer, create_scheduler
from model_builder import create_model
from train_standard import train_standard
from train_live_update import train_live_update
from train_live_update_fixed_split import train_live_update_fixed_split
from logger import create_logger

import argparse
from datetime import datetime

import warnings
warnings.filterwarnings("ignore")


def parse_args():
    """Parses the arguments."""
    parser = argparse.ArgumentParser(
        description='Train a classification model'
    )
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        help='Config file path',
        required=True,
        type=str
    )
    parser.add_argument(
        '--repeat',
        dest='repeat',
        help='Repeat how many random seeds',
        default=1,
        type=int
    )

    return parser.parse_args()


if __name__ == '__main__':
    # Load cmd line args
    args = parse_args()

    # Repeat for different random seeds
    for i in range(args.repeat):
        # Load config file
        cfg.merge_from_file(args.cfg_file)
        assert_cfg(cfg)

        # Set Pytorch environment
        torch.set_num_threads(cfg.num_threads)
        cfg.seed = i + 1
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

        # Set learning environment
        datasets = create_dataset()

        # required by gconv-lstm-h
        cfg.dataset.num_nodes = datasets[0][0].num_nodes

        loaders = create_loader(datasets)
        model = create_model(datasets)
        meters = create_logger(datasets, loaders)

        optimizer = create_optimizer(model.parameters())
        scheduler = create_scheduler(optimizer)

        with open('results.txt', 'a+') as f:
            f.writelines(args.cfg_file + ' ')
        
        # Start training
        for dataset, name in zip(datasets, ('train', 'validation', 'test')):
            print(f'{name} set: {len(dataset)} graphs.')
            all_edge_time = torch.cat([g.edge_time for g in dataset])
            start = int(torch.min(all_edge_time))
            start = datetime.fromtimestamp(start)
            end = int(torch.max(all_edge_time))
            end = datetime.fromtimestamp(end)
            print(f'\tRange: {start} - {end}')

        if cfg.train.mode == 'standard':
            train_standard(meters, loaders, model, optimizer, scheduler)
        elif cfg.train.mode == 'live_update_fixed_split':
            train_live_update_fixed_split(meters, loaders, model, optimizer, scheduler, datasets=datasets)
        else:
            train_live_update(loaders, model, optimizer, scheduler, datasets=datasets)
