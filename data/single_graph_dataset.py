import torch
from torch_geometric.data import InMemoryDataset


class SingleGraphDataset(InMemoryDataset):
    def __init__(self, root, sequence_name: str, transform=None, pre_transform=None):
        if str(root).endswith("/processed"):
            root = str(root).split("/processed")[0]
        super(SingleGraphDataset, self).__init__(root, transform, pre_transform)
        self.sequence_name = sequence_name
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [f"{self.sequence_name}.pt"]