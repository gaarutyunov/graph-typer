"""
Based on Graphormer codebase https://github.com/microsoft/Graphormer
"""

import logging
from dataclasses import dataclass, field
from typing import Dict

import numpy as np
from fairseq.data import NestedDictionaryDataset, EpochBatchIterator
from fairseq.data.iterators import StreamingEpochBatchIterator
from fairseq.tasks import register_task

from graph_coder.data.masked_dataset import MaskedDataset
from tokengt.data import DATASET_REGISTRY
from tokengt.data.dataset import EpochShuffleDataset
from tokengt.tasks.graph_prediction import GraphPredictionConfig, GraphPredictionTask

logger = logging.getLogger(__name__)


@dataclass
class NodeClassificationConfig(GraphPredictionConfig):
    dataset_root: str = field(
        default="~/data",
        metadata={"help": "Dataset root folder"},
    )

    processed_dir: str = field(
        default="processed-dir",
        metadata={"help": "Dataset processed folder"},
    )

    num_data_workers: int = field(
        default=4,
        metadata={"help": "number of data workers"},
    )

    mask_ratio: float = field(
        default=0.5,
        metadata={"help": "node masking ratio"},
    )

    max_tokens: int = field(
        default=4096,
        metadata={"help": "number tokens per graph"},
    )

    num_atoms: int = field(
        default=10000 + 512 + 2,
        metadata={"help": "number of atom types in the graph"},
    )

    num_edges: int = field(
        default=8 + 2,
        metadata={"help": "number of edge types in the graph"},
    )


@register_task("node_classification", dataclass=NodeClassificationConfig)
class NodeClassificationTask(GraphPredictionTask):
    def __init__(self, cfg):
        super(GraphPredictionTask, self).__init__(cfg)
        self.sizes: Dict[str, int] = {}
        if cfg.user_data_dir != "":
            self._import_user_defined_datasets(cfg.user_data_dir)
            if cfg.dataset_name in DATASET_REGISTRY:
                self.dataset_initializer = DATASET_REGISTRY[cfg.dataset_name]
            else:
                raise ValueError(
                    f"dataset {cfg.dataset_name} is not found in customized dataset module {cfg.user_data_dir}"
                )
        else:
            raise ValueError(
                "use --user-data-dir for custom datasets"
            )

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split (e.g., train, valid, test)."""

        assert split in ["train", "valid", "test"]

        dataset = self.dataset_initializer(self.cfg, split)

        logger.info("Loaded {0} with #samples: {1}".format(split, len(dataset)))

        self.datasets[split] = dataset
        self.sizes[split] = len(dataset)

        return self.datasets[split]

    def dataset(self, split):
        return self.datasets[split]

    def get_batch_iterator(self, dataset, max_tokens=None, max_sentences=None, max_positions=None,
                           ignore_invalid_inputs=False, required_batch_size_multiple=1, seed=1, num_shards=1,
                           shard_id=0, num_workers=0, epoch=1, data_buffer_size=0, disable_iterator_cache=False,
                           skip_remainder_batch=False, grouped_shuffling=False, update_epoch_batch_itr=False):
        return StreamingEpochBatchIterator(
            dataset=dataset,
            max_sentences=max_sentences,
            collate_fn=dataset.collater,
            buffer_size=data_buffer_size,
            num_workers=num_workers,
            epoch=epoch,
        )


