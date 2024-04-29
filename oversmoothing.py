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

from torch_geometric.utils import to_dense_adj

import torch.nn.functional as F



from train import get_dataset

log = get_logger(__name__, level = logging.INFO)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    # parse 3 arguments
    # python3 evaluate.py --wandb --log_model_dir /home/ricardob/data/TimePDE --group LG4 --dataset ks_uniform --model s4fno --timestamp 2024-04-04_16-46-47
    parser = argparse.ArgumentParser(description='Evaluation arguments')
    parser.add_argument('--group', type=str, help='wandb group')
    parser.add_argument('--model', type=str)
    parser.add_argument('--timestamp', type=str)
    parser.add_argument('--log_model_dir', type=str)
    parser.add_argument('--wandb', action='store_true', help='To use wandb logging or not', default=True)

    args = parser.parse_args()

    fdir = os.path.join(args.log_model_dir, args.model, args.group,  args.timestamp)

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

    oversmoothing = torch.zeros(cfg.model.params.num_layers).to(device)

    for graph in test_loader:
        x = graph.x.to(device)
        edge_index = graph.edge_index.to(device)
        edge_attr = graph.edge_attr.to(device)
        batch = graph.batch.to(device)
        pe = graph.pe.to(device)
        if graph.get("dist_mask") is not None:
            dist_mask = graph.dist_mask.to(device)
        else: 
            dist_mask = None

        model.eval()

        with torch.no_grad():
            x_pe = model.pe_norm(pe)

        if model.embed_type == "embedding":
            x = torch.cat(
                (model.node_emb(x.squeeze(-1)), model.pe_lin(x_pe)), 1
            )
            edge_attr = model.edge_emb(edge_attr)
        elif model.embed_type == "linear":
            x = torch.cat(
                (model.node_emb(x.float()), model.pe_lin(x_pe)), 1
            )
            edge_attr = model.edge_emb(edge_attr.float())
        else:
            raise ValueError(f"This embedding type {model.embed_type} is not valid")

        for l,(mpnn, post_seq_models, mlp, norm) in enumerate(zip(
            model.mpnns, model.post_seq_models, model.mlps, model.norms
        )):
            z = norm(x)
            h_local = mpnn(
                z,
                edge_index=edge_index,
                edge_attr=edge_attr,
                batch=batch,
                dist_mask=dist_mask,
            )
            h_local = F.dropout(h_local, p=model.dropout, training=model.training)
            # x = h_local + x


            if model.post_seq_model is not None:
                serialized_h, mask = model.serialization.serialize(
                    z, edge_index, batch, edge_attr
                )
                h_global = post_seq_models(serialized_h)[mask]
                h_global = F.dropout(h_global, p=model.dropout, training=model.training)
                h_global = h_global + x
                z = h_local + h_global
            else:
                z = h_local

            x = mlp(z) + z

            # compute difference of norm of x with all its neighbours
            # x: (M, H)
            # compute count of nodes per graph
            count_nodes = torch.bincount(batch)
            for node1, node2 in edge_index.T:
                x1 = x[node1]
                x2 = x[node2]
                diff = torch.norm(x1 - x2)**2
                # divide by number of nodes of such graph

                oversmoothing[l] += diff.item() / count_nodes[batch[node1]]
        
    oversmoothing = [o.detach().cpu().numpy() / len(test_loader) for o in oversmoothing]

    # wandb log
    for l, o in enumerate(oversmoothing):
        wandb.log({f"oversmoothing/{l}": o})

    print(oversmoothing)



    # wain until wandb uploads
    # time.sleep(10)









