import argparse
import os.path as osp
from typing import Any, Dict, Optional

import torch
from torch.nn import (
    BatchNorm1d,
    Embedding,
    Linear,
    ModuleList,
    ReLU,
    Sequential,
    LayerNorm,
)
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch_geometric.transforms as T
from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_add_pool
import inspect
from typing import Any, Dict, Optional

import torch.nn.functional as F
from torch import Tensor
from torch.nn import Dropout, Linear, Sequential

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.nn.resolver import (
    activation_resolver,
    normalization_resolver,
)
from torch_geometric.typing import Adj
from torch_geometric.utils import to_dense_batch

from mamba_ssm import Mamba
from torch_geometric.utils import degree, sort_edge_index

from models.serialization import NodeOrder
from models.mpnn import GREDMamba, CustomGINEConv

from utils.utils import time_it


class GraphModel(torch.nn.Module):
    def __init__(
        self,
        mpnn_type: str,
        post_seq_model: str,
        global_serialization_type: str,
        feature_dim: int, 
        edge_dim: int, 
        classes: int,
        embed_type: str,
        d_model: int = 64,
        pe_dim: int = 8,
        num_layers: int = 10,
        
        d_state: int = 16,
        d_conv: int = 4,
        dropout: float = 0.0,
        act_kwargs: Optional[Dict[str, Any]] = None,
        act: str = "relu",
        K: int = 5,
        # mpnn_kwargs: Optional[Dict[str, Any]] = None,
        # post_seq_model_kwargs: Optional[Dict[str, Any]] = None,
        # serialization_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()


        self.embed_type = embed_type

        if embed_type == "embedding":

            self.node_emb = Embedding(
                28, d_model - pe_dim
            )  # (# max_nodes, #embedding_dim)
            self.edge_emb = Embedding(4, d_model)

        elif embed_type == "linear":

            self.node_emb = Linear(
                feature_dim, d_model - pe_dim
            )  
            self.edge_emb = Linear(
                edge_dim, d_model
            )

        else:
            raise ValueError(f"This embedding type {embed_type} is not valid")

        self.pe_lin = Linear(20, pe_dim)
        self.pe_norm = BatchNorm1d(20)

        self.mpnn_type = mpnn_type
        self.post_seq_model = post_seq_model
        self.global_serialization_type = global_serialization_type
        self.dropout = dropout

        if mpnn_type == "gine":
            nn = Sequential(
                Linear(d_model, d_model),
                ReLU(),
                Linear(d_model, d_model),
            )
            mpnn_gen = lambda: CustomGINEConv(nn)
        elif mpnn_type == "GREDMamba":
            mpnn_gen = lambda: GREDMamba(
                d_model=d_model, d_state=d_state, d_conv=d_conv, expand=1, K=K
            )
        else:
            raise ValueError(f"MPNN type {mpnn_type} not recognized")

        if post_seq_model == "mamba":
            post_seq_model_gen = lambda: Mamba(
                d_model=d_model, d_state=d_state, d_conv=d_conv, expand=1
            )
        elif post_seq_model is None: 
            post_seq_model_gen = lambda: None
        else:
            raise ValueError(f"Post sequence model {post_seq_model} not recognized")

        if global_serialization_type == "node_order":
            self.serialization = NodeOrder()
        elif global_serialization_type is None: 
            self.serialization = None
        else:
            raise ValueError(f"Serialization type {global_serialization_type} not recognized")
        
        mpl_gen = lambda: Sequential(
            Linear(d_model, d_model * 2),
            activation_resolver(act, **(act_kwargs or {})),
            Dropout(dropout),
            Linear(d_model * 2, d_model),
            Dropout(dropout),
        )

        self.mpnns = ModuleList()
        self.post_seq_models = ModuleList()
        self.mlps = ModuleList()
        self.norms = ModuleList()
        for _ in range(num_layers):
            self.mpnns.append(mpnn_gen())
            self.post_seq_models.append(post_seq_model_gen())
            self.mlps.append(mpl_gen())
            # layer norm
            self.norms.append(LayerNorm(d_model))

        self.final_mlp = Sequential(
            Linear(d_model, d_model // 2),
            ReLU(),
            Linear(d_model // 2, d_model // 4),
            ReLU(),
            Linear(d_model // 4, classes),
        )

    def forward(self, x, pe, edge_index, edge_attr, batch, dist_mask=None):
        x_pe = self.pe_norm(pe)

        if self.embed_type == "embedding":
            x = torch.cat(
                (self.node_emb(x.squeeze(-1)), self.pe_lin(x_pe)), 1
            ) 
            edge_attr = self.edge_emb(edge_attr)
        elif self.embed_type == "linear":
            x = torch.cat(
                (self.node_emb(x.float()), self.pe_lin(x_pe)), 1
            ) 
            edge_attr = self.edge_emb(edge_attr.float())
        else:
            raise ValueError(f"This embedding type {self.embed_type} is not valid")

        for mpnn, post_seq_models, mlp, norm in zip(self.mpnns, self.post_seq_models, self.mlps, self.norms):

            h_local = mpnn(x, edge_index = edge_index, edge_attr=edge_attr, batch = batch, dist_mask = dist_mask)
            h_local = F.dropout(h_local, p=self.dropout, training=self.training)
            h_local = h_local + x

            if self.post_seq_model is not None:
                serialized_h, mask = self.serialization.serialize(x, edge_index, batch, edge_attr)
                h_global = post_seq_models(serialized_h)[mask]
                h_global = F.dropout(h_global, p=self.dropout, training=self.training)
                h_global = h_global + x
                x = h_local + h_global
            else: 
                x = h_local
            
            x = mlp(x)

            x = norm(x)

        x = global_add_pool(x, batch)
        return self.final_mlp(x)
