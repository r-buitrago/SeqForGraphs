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
from torch_geometric.utils import to_dense_batch, to_dense_adj

from mamba_ssm import Mamba
from torch_geometric.utils import degree, sort_edge_index

import scipy.sparse.csgraph as csg
import numpy as np

from einops import rearrange

from timeit import default_timer as timer


def floyd_warshall(adj_matrix, K):
    shortest_paths = csg.floyd_warshall(adj_matrix, directed=False)
    k_matrix = np.transpose(
        (np.arange(K + 1)[::-1] == shortest_paths[..., None]).astype(int), (2, 0, 1)
    )

    # Testing Floyd_Warshall
    # for n in np.arange(K+1)[::-1]:
    #   assert np.sum(k_matrix[n]) == np.sum(shortest_paths == (K - n))

    return k_matrix


class GlobalSerialization:
    name = "SerializationABC"

    def __init__(self, **kwargs):
        """Returns a global serialization of the graph"""
        pass

    @abstractmethod
    def serialize(self, x, edge_index, batch, edge_attr=None):
        """
        N: number of nodes
        H: hidden dimension
        :return h: (B, N, H)
        :return mask: (B, N)"""
        pass


class LocalSerialization:
    name = "SerializationABC"

    def __init__(self, **kwargs):
        """Returns a serialization of the graph for each node"""
        pass

    @abstractmethod
    def serialize(self, x, edge_index, batch, edge_attr=None):
        """
        N: number of nodes
        :return h: (B, N, K, H)
        :return mask: (B,N)"""
        pass


def sparse_to_dense(compressed_dist_mask, n_nodes, K):
    # compressed_dist_mask (M, 4) == (grad_idx, dist, node1, node2)

    # contiguous compressed_dist_mask
    compressed_dist_mask = compressed_dist_mask.contiguous()

    compressed_dist_mask = compressed_dist_mask[compressed_dist_mask[:, 2] < n_nodes]
    compressed_dist_mask = compressed_dist_mask[compressed_dist_mask[:, 3] < n_nodes]
    compressed_dist_mask = compressed_dist_mask[compressed_dist_mask[:, 1] < K]

    unique_graph_ids, graph_ids = torch.unique(
        compressed_dist_mask[:, 0], return_inverse=True, sorted=False
    )
    B = len(unique_graph_ids)

    # covert to dense (batch_size, K+1, num_nodes, num_nodes)
    dist_mask = torch.zeros(B, K, n_nodes, n_nodes).to("cuda")

    # sorted_unique_graph_ids, _ = torch.sort(unique_graph_ids)
    # new_graph_ids = torch.searchsorted(
    #     sorted_unique_graph_ids, compressed_dist_mask[:, 0]
    # )

    node1s = compressed_dist_mask[:, 2]
    node2s = compressed_dist_mask[:, 3]
    distances = compressed_dist_mask[:, 1]

    dist_mask[graph_ids, distances, node1s, node2s] = 1

    # reverse it in distances
    dist_mask = dist_mask.flip(1)
    return dist_mask


class NodeOrder(GlobalSerialization):
    name = "NodeOrder"

    def serialize(self, x, edge_index, batch, edge_attr=None):
        deg = degree(edge_index[0], x.shape[0]).to(torch.long)
        order_tensor = torch.stack([batch, deg], 1).T
        _, x = sort_edge_index(order_tensor, edge_attr=x)
        h, mask = to_dense_batch(x, batch)
        return h, mask


class GRED(LocalSerialization):
    name = "GRED"

    def __init__(self, K):
        self.K = K

    def serialize(
        self, x, edge_index, batch, edge_attr=None, compressed_dist_mask=None
    ):
        inputs, mask = to_dense_batch(
            x, batch
        )  # Shape of inputs: (batch_size, num_nodes, dim_h)
        inputs = inputs.float().to("cuda")
        mask = mask.float().to("cuda")

        # Create distance matrix using floyd_warshall
        adj_matrix = to_dense_adj(
            edge_index, batch=batch, max_num_nodes=inputs.shape[1]
        )

        n_nodes = inputs.shape[1]

        # Compute the shortest paths using Floyd-Warshall
        if compressed_dist_mask is None:
            k_matrix = [floyd_warshall(i, self.K) for i in adj_matrix.cpu().numpy()]

            # Shape of dist_masks: (batch_size, K+1, num_nodes, num_nodes)
            dist_mask = torch.tensor(np.array(k_matrix), dtype=torch.float32).to("cuda")

        else:
            dist_mask = sparse_to_dense(compressed_dist_mask, n_nodes, self.K + 1)

        out = torch.swapaxes(dist_mask, 0, 1) @ inputs  # (K, B, N, H)
        out = rearrange(out, "k b n h -> b n k h")
        return out, mask
