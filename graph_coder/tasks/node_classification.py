"""
Based on Graphormer codebase https://github.com/microsoft/Graphormer
"""

import logging
from dataclasses import dataclass, field

import numpy as np
from fairseq.data import NestedDictionaryDataset, NumSamplesDataset
from fairseq.tasks import register_task

from graph_coder.data.masked_dataset import MaskedDataset
from tokengt.data import DATASET_REGISTRY
from tokengt.data.dataset import BatchedDataDataset, TargetDataset, EpochShuffleDataset
from tokengt.tasks.graph_prediction import GraphPredictionConfig, GraphPredictionTask

logger = logging.getLogger(__name__)


@dataclass
class NodeClassificationConfig(GraphPredictionConfig):
    dataset_root: str = field(
        default="~/data",
        metadata={"help": "Dataset root folder"},
    )

    weights_path: str = field(
        default="~/data/processed-data/train/weights.pkl.gz",
        metadata={"help": "Weights path relative to the dataset root"},
    )

    max_nodes: int = field(
        default=10000,
        metadata={"help": "max nodes per graph"},
    )

    max_edges: int = field(
        default=20000,
        metadata={"help": "max edges per graph"},
    )

    num_atoms: int = field(
        default=10000 + 1,
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

        batched_data = self.dataset_initializer(self.cfg, split)

        batched_data = BatchedDataDataset(
            batched_data,
            max_node=self.max_nodes(),
            max_edge=self.max_edges(),
            multi_hop_max_dist=self.cfg.multi_hop_max_dist,
            spatial_pos_max=self.cfg.spatial_pos_max
        )
        masked_tokens = MaskedDataset(
            batched_data,
        )

        data_sizes = np.array([self.max_nodes()] * len(batched_data))

        target = TargetDataset(batched_data, max_node=batched_data.max_node)

        dataset = NestedDictionaryDataset(
            {
                "nsamples": NumSamplesDataset(),
                "net_input": {"batched_data": batched_data, "masked_tokens": masked_tokens},
                "target": target,
            },
            sizes=data_sizes,
        )

        if split == "train" and self.cfg.train_epoch_shuffle:
            dataset = EpochShuffleDataset(
                dataset,
                num_samples=len(dataset),
                seed=self.cfg.seed
            )

        logger.info("Loaded {0} with #samples: {1}".format(split, len(dataset)))

        self.datasets[split] = dataset
        return self.datasets[split]
