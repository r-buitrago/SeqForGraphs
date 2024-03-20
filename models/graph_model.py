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
)
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch_geometric.transforms as T
from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_add_pool
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

class GraphModel(torch.nn.Module):
    def __init__(
        self,
        mpnn_type: str,
        post_seq_model: str,
        serialization_type: str,
        channels: int = 64,
        pe_dim: int = 8,
        num_layers: int = 10,
        d_state: int = 16,
        d_conv: int = 4,
        dropout: float = 0.0,
        act_kwargs: Optional[Dict[str, Any]] = None,
        act: str = "relu",
        # mpnn_kwargs: Optional[Dict[str, Any]] = None,
        # post_seq_model_kwargs: Optional[Dict[str, Any]] = None,
        # serialization_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        self.node_emb = Embedding(
            28, channels - pe_dim
        )  # (# max_nodes, #embedding_dim)
        self.pe_lin = Linear(20, pe_dim)
        self.pe_norm = BatchNorm1d(20)
        self.edge_emb = Embedding(4, channels)
        self.mpnn_type = mpnn_type
        self.post_seq_model = post_seq_model
        self.serialization_type = serialization_type
        self.dropout = dropout

        if mpnn_type == "gine":
            nn = Sequential(
                Linear(channels, channels),
                ReLU(),
                Linear(channels, channels),
            )
            mpnn_gen = lambda: GINEConv(nn)
        else:
            raise ValueError(f"MPNN type {mpnn_type} not recognized")

        if post_seq_model == "mamba":
            post_seq_model_gen = lambda: Mamba(
                d_model=channels, d_state=d_state, d_conv=d_conv, expand=1
            )
        else:
            raise ValueError(f"Post sequence model {post_seq_model} not recognized")

        if serialization_type == "node_order":
            self.serialization = NodeOrder()
        else:
            raise ValueError(f"Serialization type {serialization_type} not recognized")
        
        mpl_gen = lambda: Sequential(
            Linear(channels, channels * 2),
            activation_resolver(act, **(act_kwargs or {})),
            Dropout(dropout),
            Linear(channels * 2, channels),
            Dropout(dropout),
        )

        self.mpnns = ModuleList()
        self.post_seq_models = ModuleList()
        self.mlps = ModuleList()
        for _ in range(num_layers):
            self.mpnns.append(mpnn_gen())
            self.post_seq_models.append(post_seq_model_gen())
            self.mlps.append(mpl_gen())

        self.final_mlp = Sequential(
            Linear(channels, channels // 2),
            ReLU(),
            Linear(channels // 2, channels // 4),
            ReLU(),
            Linear(channels // 4, 1),
        )

    def forward(self, x, pe, edge_index, edge_attr, batch):
        x_pe = self.pe_norm(pe)
        x = torch.cat(
            (self.node_emb(x.squeeze(-1)), self.pe_lin(x_pe)), 1
        ) 
        edge_attr = self.edge_emb(edge_attr)

        for mpnn, post_seq_models, mlp in zip(self.mpnns, self.post_seq_models, self.mlps):
            h_local = mpnn(x, edge_index, edge_attr=edge_attr)
            h_local = F.dropout(h_local, p=self.dropout, training=self.training)
            h_local = h_local + x

            serialized_h, mask = self.serialization.serialize(x, edge_index, batch, edge_attr)
            h_global = post_seq_models(serialized_h)[mask]
            h_global = F.dropout(h_global, p=self.dropout, training=self.training)
            h_global = h_global + x
            x = h_local + h_global
            x = mlp(x)

        x = global_add_pool(x, batch)
        return self.final_mlp(x)
