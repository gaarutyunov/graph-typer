from functools import lru_cache

import torch
from fairseq.data import FairseqDataset
import torch.nn.functional as F


class MaskedDataset(FairseqDataset):
    def __init__(self,
                 dataset,
                 mask_ratio: float = .75):
        super().__init__()
        self.dataset = dataset
        self.mask_ratio = mask_ratio

    def __getitem__(self, index):
        return self._get_random_token_mask(index)

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        token_num = [i.size(0) for i in samples]
        max_n = max(token_num)
        token_mask = torch.cat([F.pad(i[None, ...], (0, max_n - i.size(0)), value=False) for i in samples])

        return token_mask

    def _get_random_token_mask(self, index) -> torch.Tensor:
        sample = self.dataset[index]
        nodes_with_labels_mask = sample.y != -100
        nodes_with_labels = sample.y[nodes_with_labels_mask]

        num_nodes = nodes_with_labels.size(0)
        node_mask = torch.rand((num_nodes,), device=sample.y.device) < self.mask_ratio
        node_mask_ = nodes_with_labels_mask.clone()
        node_mask_[nodes_with_labels_mask] = nodes_with_labels_mask[nodes_with_labels_mask] * node_mask

        token_count = sample.node_data.size(0) + sample.edge_data.size(0)

        token_mask = F.pad(node_mask_, (0, token_count - node_mask_.size(0)), value=False)

        return token_mask
