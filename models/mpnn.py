import argparse
import os.path as osp
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv
from torch_geometric.utils import to_dense_batch

from mamba_ssm import Mamba

from torch_geometric.utils import to_dense_adj
from scipy.sparse.csgraph import floyd_warshall
import scipy.sparse.csgraph as csg

from models.serialization import GRED

import numpy as np

from einops import rearrange

from utils.utils import time_it


class CustomGINEConv(GINEConv):
    def forward(self, x, edge_index, edge_attr=None, batch = None, dist_mask = None):
        return super().forward(x, edge_index, edge_attr)

class MLP(nn.Module):
    def __init__(self, dim_h, drop_rate=0.):
        super(MLP, self).__init__()
        self.dim_h = dim_h
        self.drop_rate = drop_rate
        self.layer_norm = nn.LayerNorm(dim_h)
        self.dense1 = nn.Linear(dim_h, dim_h)
        self.dropout1 = nn.Dropout(drop_rate)
        self.dense2 = nn.Linear(dim_h, dim_h)
        self.dropout2 = nn.Dropout(drop_rate)

    def forward(self, inputs, training=False):
        x = self.layer_norm(inputs)
        x = self.dense1(x)
        x = F.gelu(x)
        x = self.dropout1(x) if training else x
        x = self.dense2(x)
        x = self.dropout2(x) if training else x
        return x + inputs
    

class GREDMamba(torch.nn.Module):
    def __init__(self, d_model, K = 5, d_state=16, d_conv=4, expand=1):
        super().__init__()

        self.mlp = MLP(d_model)
        self.self_attn = Mamba(
              d_model=d_model,
              d_state=d_state,
              d_conv=d_conv,
              expand=expand,
          )
        self.K = K
        self.local_serialization = GRED(K)

    def forward(self, x, batch, edge_index = None, edge_attr = None, dist_mask = None):
        x, mask = self.local_serialization.serialize(x, edge_index, batch, edge_attr, dist_mask = dist_mask)
        B, N, K, H = x.shape

        # Shape of x: (K+1, batch_size, num_nodes, dim_h)
        x = self.mlp(x)

        x = x.reshape(B * N, K, H) # Shape of x: (batch_size * num_nodes, K+1, dim_h)

        x = self.self_attn(x)[..., -1, :] # (batch_size * num_nodes, dim_h)
        
        x = rearrange(x, '(B N) H -> B N H', B=B, N=N, H=H)[mask.bool()]# Shape of x: (batch_size, num_nodes,dim_h)

        return x


