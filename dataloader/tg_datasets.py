from torch_geometric.datasets import ZINC
import torch_geometric.transforms as T


def get_zinc_dataset(path, subset, split="train"):
    transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')
    return ZINC(path, subset, split=split, transform=transform)
