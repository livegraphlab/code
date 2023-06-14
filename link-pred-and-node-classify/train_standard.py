import torch
import time
import logging

from config import cfg
from train_utils import compute_loss
from tqdm import tqdm


def is_eval_epoch(cur_epoch):
    """Determines if the model should be evaluated at the current epoch."""
    return (
            (cur_epoch + 1) % cfg.train.eval_period == 0 or
            cur_epoch == 0 or
            (cur_epoch + 1) == cfg.optim.max_epoch
    )


def train_epoch(logger, loader, model, optimizer, scheduler):
    model.train()
    time_start = time.time()
    for batch in loader:
        optimizer.zero_grad()
        batch.to(torch.device(cfg.device))
        pred, true = model(batch)
        loss, pred_score = compute_loss(pred, true)
        loss.backward()
        optimizer.step()
        logger.update_stats(true=true.detach().cpu(),
                            pred=pred_score.detach().cpu(),
                            loss=loss.item(),
                            lr=scheduler.get_last_lr()[0],
                            time_used=time.time() - time_start)
        time_start = time.time()
    scheduler.step()


def eval_epoch(logger, loader, model):
    model.eval()
    time_start = time.time()
    for batch in loader:
        batch.to(torch.device(cfg.device))
        pred, true = model(batch)
        loss, pred_score = compute_loss(pred, true)

        logger.update_stats(true=true.detach().cpu(),
                            pred=pred_score.detach().cpu(),
                            loss=loss.item(),
                            lr=0,
                            time_used=time.time() - time_start)

        time_start = time.time()


def train_standard(logger, loaders, model, optimizer, scheduler):
    start_epoch = 0
    num_splits = len(logger)
    for cur_epoch in tqdm(range(start_epoch, cfg.optim.max_epoch), desc='epoch', leave=True):
        train_epoch(logger[0], loaders[0], model, optimizer, scheduler)
        logger[0].write_epoch(cur_epoch)
        if is_eval_epoch(cur_epoch):
            for i in range(1, num_splits):
                eval_epoch(logger[i], loaders[i], model)
                logger[i].write_epoch(cur_epoch)
