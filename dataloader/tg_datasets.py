from torch_geometric.datasets import ZINC
import torch_geometric.transforms as T
from models.serialization import GRED

# class GREDZINC(ZINC):
#     def __init__(self, root, subset, split, transform=None, K = 5):
#         self.zinc = super().__init__(root, subset, split, transform)
#         self.dist_masks = []
#         serialization = GRED(K)
#         for idx in range(len(self.zinc)):
#             g = self.zinc[idx]
#             self.dist_masks.append(serialization.serialize(g.x, g.edge_index, g.batch))
    
#     def __getitem__(self, idx):
#         data = self.zinc[idx]
#         return data.x, data.pe, data.edge_index, data.edge_attr, data.batch, self.dist_masks[idx]




def get_zinc_dataset(path, subset, split="train"):
    transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')
    return ZINC(path, subset, split=split, transform=transform)
