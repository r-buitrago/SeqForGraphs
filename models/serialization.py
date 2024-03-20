from abc import ABC, abstractmethod
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


class Serialization:
    name = "SerializationABC"
    def __init__(self):
        pass
    
    @abstractmethod
    def serialize(self, x, edge_index, batch, edge_attr = None):
        pass


class NodeOrder(Serialization):
    name = "NodeOrder"
    def serialize(self, x, edge_index, batch, edge_attr = None):
        deg = degree(edge_index[0], x.shape[0]).to(torch.long)
        order_tensor = torch.stack([batch, deg], 1).T
        _, x = sort_edge_index(order_tensor, edge_attr=x)
        h, mask = to_dense_batch(x, batch)
        return h, mask