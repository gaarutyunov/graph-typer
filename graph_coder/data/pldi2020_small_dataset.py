import os
from typing import Literal

from tokengt.data import register_dataset
from .pldi2020_dataset import PLDI2020Dataset


@register_dataset("pldi2020_small")
def pldi2020_small(cfg, split: Literal["train", "test", "valid"] = "train", **kwargs):
    return PLDI2020Dataset(
        os.path.expanduser(cfg.dataset_root),
        split=split,
        num_classes=cfg.num_classes,
        sizes={
            "train": 1,
            "valid": 1,
            "test": 1
        },
        max_tokens=cfg.max_tokens,
        num_workers=cfg.num_data_workers,
        processed_dir=cfg.processed_dir,
        mask_ratio=cfg.mask_ratio,
        **kwargs
    ).process().load_meta()
