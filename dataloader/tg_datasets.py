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


class CustomLRGBDataset(Dataset):
    def __init__(
        self,
        path,
        name,
        precomputed_masks_path=None,
        split="train",
        transform=None,
        n_samples_max=100,
        use_distance_masks=False,
    ):
        super().__init__()
        self.tg_dataset = LRGBDataset(path, name=name, split=split, transform=transform)
        if use_distance_masks and precomputed_masks_path is not None:
            with open(precomputed_masks_path, "rb") as f:
                self.precomputed_masks = torch.load(
                    f
                )  # (M, 4) == (graph_idx, node1, node2, dist)
                # assert len(torch.unique(self.precomputed_masks[:, 0])) == n_samples_max

                # get start and end indices for each graph
                _, self.graph_counts = torch.unique(
                    self.precomputed_masks[:, 0],  return_counts=True
                )
                self.cum_counts = torch.cumsum(self.graph_counts, 0)
                self.cum_counts = torch.cat([torch.tensor([0]), self.cum_counts])
        else:
            self.precomputed_masks = None

        self.n_samples_max = n_samples_max



    def __getitem__(self, idx):
        if idx >= self.n_samples_max:
            raise IndexError
        g = self.tg_dataset[idx]
        if self.precomputed_masks is not None:
            g.dist_mask = self.precomputed_masks[self.cum_counts[idx] : self.cum_counts[idx + 1]]
            # assert torch.all(g.dist_mask[:, 0] == idx)
        return g

    def __len__(self):
        return self.n_samples_max


def get_zinc_dataset(path, subset, split="train", precomputed_masks_path=None):
    transform = T.AddRandomWalkPE(walk_length=20, attr_name="pe")
    return ZINC(
        path,
        subset=subset,
        split=split,
        transform=transform,
        # precomputed_masks_path=precomputed_masks_path,
    )


def get_lrgb_dataset(
    path,
    subset,
    n_samples_max,
    split="train",
    precomputed_masks_path=None,
    use_distance_masks=False,
):
    transform = T.AddRandomWalkPE(walk_length=20, attr_name="pe")
    return CustomLRGBDataset(
        path,
        name=subset,
        split=split,
        transform=transform,
        precomputed_masks_path=precomputed_masks_path,
        n_samples_max=n_samples_max,
        use_distance_masks=use_distance_masks,
    )
