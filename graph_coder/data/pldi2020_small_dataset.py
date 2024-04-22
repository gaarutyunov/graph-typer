import os
from functools import partial
from typing import Literal

from tokengt.data import register_dataset
from .pldi2020_dataset import PLDI2020Dataset, filter_data

__SAMPLE_SPEC__ = \
    "https://raw.githubusercontent.com/typilus/typilus/master/src/data_preparation/pldi2020-dataset-sample.spec"
__SAMPLE_PREPARE__ = \
    "https://raw.githubusercontent.com/typilus/typilus/master/src/data_preparation/scripts/prepare_data_small.sh"


@register_dataset("pldi2020_small")
def pldi2020_small(cfg, split: Literal["train", "test", "valid"] = "train"):
    return PLDI2020Dataset(
        os.path.expanduser(cfg.dataset_root),
        pre_filter=partial(filter_data, max_nodes=cfg.max_nodes, max_edges=cfg.max_edges),
        split=split,
        num_classes=cfg.num_classes,
        spec_file=__SAMPLE_SPEC__,
        prepare_file=__SAMPLE_PREPARE__,
    )
