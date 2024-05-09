import os
import subprocess
from functools import lru_cache
from typing import Union, List, Tuple, Optional, Callable, Literal, Dict, Any

import networkx as nx
import torch
from dpu_utils.utils import RichPath, ChunkWriter
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
            "train": 53,
            "valid": 7,
            "test": 17
        },
        max_nodes=cfg.max_nodes,
        max_edges=cfg.max_edges,
    ).load_meta()


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
                 max_edges: int = 4096,
                 max_chunk_size: int = 1000):
        if isinstance(root, str):
            root = os.path.expanduser(os.path.normpath(root))

        self.root = root
        self.split = split
        self.num_classes = num_classes
        self._len = 0
        self._idx = range(self._len)
        self._data_mapping = {}
        self._sample_mapping = {}
        self._size = sizes.get(split, 1)
        self._max_nodes = max_nodes
        self._max_edges = max_edges
        self._weights = None
        self._max_chunk_size = max_chunk_size
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
            self._idx = RichPath.create(self.indexes_path).read_by_file_suffix()
        if os.path.exists(self.data_mapping_path):
            self._data_mapping = RichPath.create(self.data_mapping_path).read_by_file_suffix()
        if os.path.exists(self.sample_mapping_path):
            self._sample_mapping = RichPath.create(self.sample_mapping_path).read_by_file_suffix()
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
    def data_mapping_path(self):
        return os.path.join(self.processed_dir, "data_mapping.pkl.gz")

    @property
    def sample_mapping_path(self):
        return os.path.join(self.processed_dir, "sample_mapping.pkl.gz")

    @property
    def indexes_path(self):
        return os.path.join(self.processed_dir, "indexes.pkl.gz")

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return list(f"chunk_{i:04d}.pkl.gz" for i in range(self._size))

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return list({chunk for chunk, _ in self._data_mapping.values()})

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

        with ChunkWriter(result_path, file_prefix="data_chunk_", max_chunk_size=1000, file_suffix=".pkl.gz") as data_writer,\
             ChunkWriter(result_path, file_prefix="sample_chunk_", max_chunk_size=1000, file_suffix=".pkl.gz") as sample_writer:
            data_ctx = {
                "idx": 0,
                "chunk": 0,
                "chunk_mapping": {}
            }
            sample_ctx = {
                "idx": 0,
                "chunk": 0,
                "chunk_mapping": {}
            }

            def add(writer: ChunkWriter, file_prefix: str, _idx: int, _data: Any, ctx: Dict[str, Any]):
                outfile = '%s%03d%s' % (file_prefix, ctx["chunk"], ".pkl.gz")
                ctx["chunk_mapping"][_idx] = [outfile, ctx["idx"]]
                writer.add(_data)

                if ctx["idx"] < self._max_chunk_size - 1:
                    ctx["idx"] += 1
                else:
                    ctx["idx"] = 0
                    ctx["chunk"] += 1

            for chunk in tqdm(read_data_chunks(paths), desc="Processing chunk"):
                for sample in tqdm(chunk, desc="Processing data"):
                    if "raw_data" not in sample and "supernodes" not in sample["raw_data"]:
                        continue
                    graph = sample_to_nx(sample)

                    for graph_ in split_nx(graph, self._max_nodes, self._max_edges):
                        data_ = nx_to_data(graph_)
                        data_.idx = idx
                        add(data_writer, "data_chunk_", idx, data_, data_ctx)

                        sample_ = nx_to_sample(graph_)
                        add(sample_writer, "sample_chunk_", idx, sample_, sample_ctx)

                        indexes.append(idx)
                        targets, counts = torch.unique(data_.y, return_counts=True)
                        counter[targets] += counts
                        idx += 1

            # save indexes
            RichPath.create(self.indexes_path).save_as_compressed_file(set(indexes))
            # load updated meta
            self.load_meta()
            # save weights
            weights = len(self) / (self.num_classes * counter)
            RichPath.create(self.weights_path).save_as_compressed_file(weights)
            # save data chunk mappings
            RichPath.create(self.data_mapping_path).save_as_compressed_file(data_ctx["chunk_mapping"])
            # save sample chunk mappings
            RichPath.create(self.sample_mapping_path).save_as_compressed_file(sample_ctx["chunk_mapping"])

    def len(self) -> int:
        return self._len

    @lru_cache(maxsize=20)
    def _get_chunk(self, path: str) -> List[Data | Dict[str, Any]]:
        return RichPath.create(os.path.join(self.processed_dir, path)).read_by_file_suffix()

    def get(self, idx: int) -> Data:
        chunk_path, chunk_idx = self._data_mapping[idx]
        chunk = self._get_chunk(chunk_path)
        return chunk[chunk_idx]

    def get_sample(self, idx: int) -> Dict[str, Any]:
        chunk_path, chunk_idx = self._sample_mapping[idx]
        chunk = self._get_chunk(chunk_path)
        return chunk[chunk_idx]
