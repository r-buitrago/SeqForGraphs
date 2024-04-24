from torch_geometric.datasets import ZINC, LRGBDataset
import torch_geometric.transforms as T
from models.serialization import GRED
import torch
import os
from functools import partial
from torch.utils.data import Dataset


class CustomZINCDataset(Dataset):
    def __init__(
        self,
        root,
        precomputed_masks_path=None,
        subset=False,
        split="train",
        transform=None,
    ):
        super().__init__()
        self.tg_dataset = ZINC(root, subset=subset, split=split, transform=transform)
        self.precomputed_masks_path = precomputed_masks_path

    def __getitem__(self, idx):
        g = self.tg_dataset[idx]
        if self.precomputed_masks_path is not None:
            dist_mask = torch.load(
                os.path.join(self.precomputed_masks_path, f"dist_mask_{idx}.pt")
            )
        else:
            dist_mask = None
        g.dist_mask = dist_mask
        return g

    def __len__(self):
        return len(self.tg_dataset)


def get_zinc_dataset(path, subset, split="train", precomputed_masks_path=None):
    transform = T.AddRandomWalkPE(walk_length=20, attr_name="pe")
    return ZINC(
        path,
        subset=subset,
        split=split,
        transform=transform,
        # precomputed_masks_path=precomputed_masks_path,
    )

def get_lrgb_dataset(path, subset, split="train", precomputed_masks_path=None):
    transform = T.AddRandomWalkPE(walk_length=20, attr_name="pe")
    return LRGBDataset(
        path,
        name="Peptides-func",
        split=split,
        transform=transform,
        # precomputed_masks_path=precomputed_masks_path,
    )
