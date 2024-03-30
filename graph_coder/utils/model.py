from typing import Dict

import torch


def get_padded_node_mask(sample: Dict[str, torch.Tensor]) -> torch.Tensor:
    node_num = sample["net_input"]["batched_data"]["node_num"]
    edge_num = sample["net_input"]["batched_data"]["edge_num"]
    edge_index = sample["net_input"]["batched_data"]["edge_index"]
    seq_len = [n + e for n, e in zip(node_num, edge_num)]
    b = len(seq_len)
    max_len = max(seq_len)
    device = edge_index.device

    token_pos = torch.arange(max_len, device=device)[None, :].expand(b, max_len)  # [B, T]
    node_num = torch.tensor(node_num, device=device, dtype=torch.long)[:, None]  # [B, 1]

    return torch.less(token_pos, node_num)