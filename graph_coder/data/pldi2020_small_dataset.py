import os
from functools import partial
from typing import Literal

from .pldi2020_dataset import PLDI2020Dataset, filter_data
from tokengt.data import register_dataset


@register_dataset("pldi2020_small")
def pldi2020_small(cfg, split: Literal["train", "test", "valid"] = "train"):
    return PLDI2020Dataset(
        os.path.expanduser(cfg.dataset_root),
        pre_filter=partial(filter_data, max_nodes=cfg.max_nodes, max_edges=cfg.max_edges),
        split=split,
        num_classes=cfg.num_classes,
        sizes={
            "train": 1,
            "valid": 1,
            "test": 1
        }
    )
