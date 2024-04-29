import argparse
import os
from omegaconf import OmegaConf
from datetime import datetime
from utils.log_utils import get_logger, add_file_handler
import logging
import torch
import wandb
from copy import deepcopy
from hydra.utils import instantiate

import torch.nn.functional as F


import sys, os
import numpy as np
import hydra
from hydra.utils import instantiate
import wandb
from omegaconf import OmegaConf, DictConfig

import pickle
import random
import torch
from timeit import default_timer

from utils.log_utils import get_logger, add_file_handler
import logging
from utils.evaluator import DummyEvaluator, eval_to_print, eval_to_wandb
from utils.utils import count_params
from utils.scheduler import WarmupScheduler
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.loader import DataLoader

from train import get_dataset, learning_step


log = get_logger(__name__, level = logging.INFO)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def oversquashing_statistics(
    args,
    data_loader,
    model,
    optimizer,
    loss_type,
    scheduler=None,
    evaluator=None,
    retrain:bool = True
):
    model.train()
    t1 = default_timer()
    batch_timer = default_timer()
    sensitivity_matrix_all = 0
    batches = 0
    for i, batch in enumerate(data_loader):
        loss, _, batch_embedding = learning_step(
            args,
            model,
            batch,
            loss_type=loss_type,
            evaluator=evaluator,
            batch_start=i,
            calculate_embedding=True
        )
        if optimizer:
            optimizer.zero_grad()      
        layers_, nodes_, dim_, = batch_embedding.shape
        sensitivity_matrix = torch.zeros(layers_)
        for node in random.sample(range(nodes_), 10):
            for f in range(0, dim_):
                batch_embedding[-1][node, f].backward(retain_graph=True)
                batch_embedding.retain_grad()
                for hidden_layer in range(layers_):
                    breakpoint()
                    # batch_embedding[hidden_layer].retain_grad()
                    if batch_embedding[hidden_layer].grad != None:
                        sensitivity_matrix[hidden_layer] += torch.norm(batch_embedding[hidden_layer].grad, p=1).cpu().numpy()
                        print(f"grad computed {batch_embedding[hidden_layer].grad}")
                        batch_embedding[hidden_layer].grad = None
        sensitivity_matrix_all += sensitivity_matrix
        batches += 1
        if retrain and optimizer and evaluator and loss_type:
            optimizer.step()
            #optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()
        log.debug(f"Training batch time: {default_timer() - batch_timer}")
        batch_timer = default_timer()

    sensitivity_matrix_all = sensitivity_matrix_all / batches
    log.info(f'Oversquashing Analysis, by Layer: {sensitivity_matrix_all}')

    t2 = default_timer()
    return t2 - t1, sensitivity_matrix_all


@torch.no_grad()
def oversmoothing_statistics(args, test_loader, model, loss_type, evaluator=None):
    model.eval()
    t1 = default_timer()
    batch_timer = default_timer()

    batch_embedding_diffs = 0
    batches = 0
    
    for i, batch in enumerate(test_loader):

        loss, layer_embedding_diff, _ = learning_step(
            args,
            model,
            batch,
            loss_type=loss_type,
            evaluator=evaluator,
            is_training=False,
            batch_start=i,
            calculate_embedding_diff=True
        )

        batch_embedding_diffs += layer_embedding_diff
        batches += 1

        log.debug(f"Evaluation batch time: {default_timer() - batch_timer}")
        batch_timer = default_timer()

    batch_embedding_diffs = batch_embedding_diffs / batches

    log.info(f'Oversmoothing Analysis, by Layer: {batch_embedding_diffs}')
    
    t2 = default_timer()
    return t2 - t1, batch_embedding_diffs 






if __name__ == "__main__":
    # parse 3 arguments
    # python3 evaluate.py --wandb --log_model_dir /home/ricardob/data/TimePDE --group LG4 --dataset ks_uniform --model s4fno --timestamp 2024-04-04_16-46-47
    parser = argparse.ArgumentParser(description='Evaluation arguments')
    parser.add_argument('--group', type=str, help='wandb group')
    parser.add_argument('--model', type=str)
    parser.add_argument('--timestamp', type=str)
    parser.add_argument('--log_model_dir', type=str)
    parser.add_argument('--wandb', action='store_true', help='To use wandb logging or not', default=False)

    args = parser.parse_args()

    #fdir = os.path.join(args.log_model_dir, args.model, args.group,  args.timestamp)
    fdir = "pretrained/gred_mamba_zinc/2024-04-29_13-02-34" #TODO: quick fix for now 
    cfg_path = os.path.join(fdir, "cfg.yaml")
    model_path = os.path.join(fdir, "ckpt.pt")
    model_state = torch.load(model_path)

    # compose cfg
    cfg = OmegaConf.load(cfg_path)
    # change timestamp
    curr_t = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    cfg.wandb.name = f"FINAL_{cfg.model.name}_{cfg.dataset.name}_{curr_t}"
    cfg.only_evaluation = True

    #wandb
    if args.wandb:
        log.enable_wandb()
        wandb_config = OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        )
        if cfg.wandb.tags is not None:
            run = wandb.init(
                project=cfg.wandb.project,
                group=cfg.wandb.group,
                name=cfg.wandb.name,
                config=wandb_config,
                tags=list(cfg.wandb.tags))
        else:
            run = wandb.init(
                project=cfg.wandb.project,
                group=cfg.wandb.group,
                name=cfg.wandb.name,
                config=wandb_config) 

    args = cfg
    train_loader, test_loader = get_dataset(args, batch_size=args.model.batch_size)

    _data = next(iter(test_loader))
    print(f"Dataset shape: {_data}")

    model = instantiate(args.model.params)
    model.to(device)

    log.info("Loading pretrained weights")
    model.load_state_dict(model_state)

    time_1, batch_embedding_diff = oversmoothing_statistics(args, test_loader, model, "mse")
    time_2, sensitivity_matrix = oversquashing_statistics(args, test_loader, model, None, "mse", None, None, None)


# def oversquashing_statistics(
#     args,
#     data_loader,
#     model,
#     optimizer,
#     loss_type,
#     scheduler=None,
#     evaluator=None,
#     retrain:bool = True
# ):
    breakpoint()
    # for graph in test_loader:
    #     x = graph.x.to(device)
    #     edge_index = graph.edge_index.to(device)
    #     edge_attr = graph.edge_attr.to(device)
    #     batch = graph.batch.to(device)
    #     pe = graph.pe.to(device)
    #     dist_mask = graph.dist_mask.to(device)

    #     model.eval()

    #     with torch.no_grad():
    #         x_pe = model.pe_norm(pe)

    #     if model.embed_type == "embedding":
    #         x = torch.cat(
    #             (model.node_emb(x.squeeze(-1)), model.pe_lin(x_pe)), 1
    #         )
    #         edge_attr = model.edge_emb(edge_attr)
    #     elif model.embed_type == "linear":
    #         x = torch.cat(
    #             (model.node_emb(x.float()), model.pe_lin(x_pe)), 1
    #         )
    #         edge_attr = model.edge_emb(edge_attr.float())
    #     else:
    #         raise ValueError(f"This embedding type {model.embed_type} is not valid")

    #     for mpnn, post_seq_models, mlp, norm in zip(
    #         model.mpnns, model.post_seq_models, model.mlps, model.norms
    #     ):
    #         z = norm(x)
    #         h_local = mpnn(
    #             z,
    #             edge_index=edge_index,
    #             edge_attr=edge_attr,
    #             batch=batch,
    #             dist_mask=dist_mask,
    #         )
    #         h_local = F.dropout(h_local, p=model.dropout, training=model.training)
    #         # x = h_local + x


    #         if model.post_seq_model is not None:
    #             serialized_h, mask = model.serialization.serialize(
    #                 z, edge_index, batch, edge_attr
    #             )
    #             h_global = post_seq_models(serialized_h)[mask]
    #             h_global = F.dropout(h_global, p=model.dropout, training=model.training)
    #             h_global = h_global + x
    #             z = h_local + h_global
    #         else:
    #             z = h_local

    #         x = mlp(z) + z

    

    # wain until wandb uploads
    # time.sleep(10)









