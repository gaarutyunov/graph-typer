import os
import subprocess
from typing import Union, List, Tuple, Optional, Callable, Literal, Dict, Any

import networkx as nx
import torch
from dpu_utils.utils import RichPath
from torch_geometric.data import Dataset, Data, download_url
from tqdm.auto import tqdm

from graph_coder.data.utils import sample_to_nx, split_nx, nx_to_data, nx_to_sample
from tokengt.data import register_dataset
from typilus.model.model import read_data_chunks

__SPEC__ = \
    "https://raw.githubusercontent.com/typilus/typilus/master/src/data_preparation/pldi2020-dataset.spec"
__PREPARE__ = \
    "https://raw.githubusercontent.com/typilus/typilus/master/src/data_preparation/scripts/prepare_data.sh"


def filter_data(g: nx.Graph, max_nodes: int = 512, max_edges: int = 2048) -> Tuple[int, int, bool]:
    return max_nodes, max_edges, g.number_of_nodes() > max_nodes or g.number_of_edges() > max_edges


@register_dataset("pldi2020")
def pldi2020(cfg, split: Literal["train", "test", "valid"] = "train"):
    return PLDI2020Dataset(
        os.path.expanduser(cfg.dataset_root),
        split=split,
        num_classes=cfg.num_classes,
        sizes={
            "train": 59,
            "valid": 8,
            "test": 19
        },
        max_nodes=cfg.max_nodes,
        max_edges=cfg.max_edges,
    )


class PLDI2020Dataset(Dataset):
    def __init__(self,
                 root: Optional[str] = None,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 split: Literal["train", "test", "valid"] = "train",
                 num_classes: int = 100,
                 spec_file: str = __SPEC__,
                 prepare_file: str = __PREPARE__,
                 sizes: Dict[str, int] = None,
                 max_nodes: int = 1024,
                 max_edges: int = 4096):
        if isinstance(root, str):
            root = os.path.expanduser(os.path.normpath(root))

        self.root = root
        self.split = split
        self.num_classes = num_classes
        self._len = 0
        self._idx = range(self._len)
        self._size = sizes.get(split, 1)
        self._max_nodes = max_nodes
        self._max_edges = max_edges
        self._weights = None
        self.load_meta()
        self.spec_file = spec_file
        self.prepare_file = prepare_file
        super().__init__(root, transform, pre_transform, pre_filter)

    def download(self):
        download_url(self.spec_file, self.raw_dir)
        download_url(self.prepare_file, self.raw_dir)

        subprocess.run(["chmod", "x+", "./prepare_data.sh"], shell=True, cwd=self.raw_dir)
        subprocess.run(["./prepare_data.sh"], shell=True, cwd=self.raw_dir)

    def load_meta(self) -> "PLDI2020Dataset":
        self._idx = set(self._idx)
        if os.path.exists(self.indexes_path):
            indexes_path = RichPath.create(self.indexes_path)
            indexes_set = indexes_path.read_by_file_suffix()
            self._idx = indexes_set
        self._idx = list(self._idx)
        self._len = len(self._idx)

        return self

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, 'tensorised-data', self.split)

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, 'processed-data', self.split)

    @property
    def weights_path(self):
        return os.path.join(self.processed_dir, "weights.pkl.gz")

    @property
    def indexes_path(self):
        return os.path.join(self.processed_dir, "indexes.pkl.gz")

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return list(f"chunk_{i:04d}.pkl.gz" for i in range(self._size))

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return [f"data_{i:d}.pkl.gz" for i in self._idx]

    @property
    def weights(self):
        if self._weights is None:
            weights_path = RichPath.create(self.weights_path)
            self._weights = weights_path.read_by_file_suffix()
        return self._weights

    def process(self):
        idx = 0
        indexes = []
        counter = torch.zeros(self.num_classes)

        path = RichPath.create(self.raw_dir)
        paths = path.get_filtered_files_in_dir("chunk_*")

        result_path = RichPath.create(self.processed_dir)

        for chunk in tqdm(read_data_chunks(paths), desc="Processing chunk"):
            for sample in tqdm(chunk, desc="Processing data"):
                if "raw_data" not in sample and "supernodes" not in sample["raw_data"]:
                    continue
                graph = sample_to_nx(sample)

                for graph_ in split_nx(graph, self._max_nodes, self._max_edges):
                    data_ = nx_to_data(graph_)
                    data_.idx = idx
                    data_path = result_path.join(f"data_{idx:d}.pkl.gz")
                    data_path.save_as_compressed_file(data_)

                    sample_ = nx_to_sample(graph_)
                    sample_path = result_path.join(f"sample_{idx:d}.pkl.gz")
                    sample_path.save_as_compressed_file(
                        sample_
                    )

                    indexes.append(idx)
                    counter[data_.y.item()] += 1
                    idx += 1

        indexes_path = RichPath.create(self.indexes_path)
        indexes_path.save_as_compressed_file(set(indexes))
        self.load_meta()
        weights = len(self) / (self.num_classes * counter)
        weights_path = RichPath.create(self.weights_path)
        weights_path.save_as_compressed_file(weights)

    def len(self) -> int:
        return self._len

    def get(self, idx: int) -> Data:
        path = RichPath.create(self.processed_dir).join(f"data_{self._idx[idx]:d}.pkl.gz")
        return path.read_by_file_suffix()

    def get_sample(self, idx: int) -> Dict[str, Any]:
        path = RichPath.create(self.processed_dir).join(f"sample_{self._idx[idx]:d}.pkl.gz")
        return path.read_by_file_suffix()
