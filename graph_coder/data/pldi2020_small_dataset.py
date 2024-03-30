import os
import subprocess
from functools import partial
from typing import Union, List, Tuple, Optional, Callable, Literal, Dict, Any

from dpu_utils.utils import RichPath
from torch_geometric.data import Dataset, Data, download_url
from tqdm.auto import tqdm

from graph_coder.data.utils import sample_to_data, split_graph, data_to_sample
from tokengt.data import register_dataset
from typilus.model.model import read_data_chunks

__SAMPLE_SPEC__ = \
    "https://raw.githubusercontent.com/typilus/typilus/master/src/data_preparation/pldi2020-dataset-sample.spec"
__SAMPLE_PREPARE__ = \
    "https://raw.githubusercontent.com/typilus/typilus/master/src/data_preparation/scripts/prepare_data_small.sh"


def filter_data(data: Data, max_nodes: int = 512, max_edges: int = 2048) -> Tuple[int, int, bool]:
    return max_nodes, max_edges, data.x.size(0) > max_nodes or data.edge_attr.size(0) > max_edges


@register_dataset("pldi2020_small")
def pldi2020_small(cfg, split: Literal["train", "test", "valid"] = "train"):
    return PLDI2020SmallDataset(
        os.path.expanduser(cfg.dataset_root),
        pre_filter=partial(filter_data, max_nodes=cfg.max_nodes, max_edges=cfg.max_edges),
        split=split
    )


class PLDI2020SmallDataset(Dataset):
    def __init__(self,
                 root: Optional[str] = None,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 split: Literal["train", "test", "valid"] = "train"):
        self.split = split
        self.__len = 291
        self.__idx = range(self.__len)
        self.indexes_path = os.path.join(root, 'processed-data', split, "indexes.pkl.gz")
        self.load_meta()
        super().__init__(root, transform, pre_transform, pre_filter)

    def download(self):
        download_url(__SAMPLE_SPEC__, self.raw_dir)
        download_url(__SAMPLE_PREPARE__, self.raw_dir)

        subprocess.run(["chmod", "x+", "./prepare_data_small.sh"], shell=True, cwd=self.raw_dir)
        subprocess.run(["./prepare_data_small.sh"], shell=True, cwd=self.raw_dir)

    def load_meta(self) -> "PLDI2020SmallDataset":
        self.__idx = set(self.__idx)
        if os.path.exists(self.indexes_path):
            indexes_path = RichPath.create(self.indexes_path)
            indexes_set = indexes_path.read_by_file_suffix()
            self.__idx = indexes_set
        self.__idx = list(self.__idx)
        self.__len = len(self.__idx)

        return self

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, 'tensorised-data', self.split)

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, 'processed-data', self.split)

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return ["chunk_0000.pkl.gz"]

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return [f"data_{i:d}.pkl.gz" for i in self.__idx]

    def process(self):
        idx = 0
        indexes = []

        path = RichPath.create(self.raw_dir)
        paths = path.get_filtered_files_in_dir("chunk_*")

        result_path = RichPath.create(self.processed_dir)

        for chunk in tqdm(read_data_chunks(paths), desc="Processing chunk"):
            for sample in tqdm(chunk, desc="Processing data"):
                if "raw_data" not in sample and "supernodes" not in sample["raw_data"]:
                    idx += 1
                    continue
                data = sample_to_data(sample)
                max_nodes, max_edges = 0, 0
                should_split = False

                if self.pre_filter is not None:
                    max_nodes, max_edges, should_split = self.pre_filter(data)

                def save(d: Data, mapping: Optional[Dict[int, int]] = None):
                    d.idx = idx
                    target_path = result_path.join(f"data_{idx:d}.pkl.gz")
                    target_path.save_as_compressed_file(d)
                    sample_path = result_path.join(f"sample_{idx:d}.pkl.gz")
                    sample_path.save_as_compressed_file(
                        data_to_sample(d, mapping, sample["raw_data"]["supernodes"], sample.get("Provenance", "?"))
                    )
                    indexes.append(idx)

                if should_split:
                    for graph, mapping in split_graph(data, max_nodes, max_edges):
                        save(graph, mapping)
                        idx += 1
                else:
                    save(data)
                    idx += 1

        indexes_path = RichPath.create(self.indexes_path)
        indexes_path.save_as_compressed_file(set(indexes))
        self.load_meta()

    def len(self) -> int:
        return self.__len

    def get(self, idx: int) -> Data:
        path = RichPath.create(self.processed_dir).join(f"data_{self.__idx[idx]:d}.pkl.gz")
        return path.read_by_file_suffix()

    def get_sample(self, idx: int) -> Dict[str, Any]:
        path = RichPath.create(self.processed_dir).join(f"sample_{self.__idx[idx]:d}.pkl.gz")
        return path.read_by_file_suffix()


if __name__ == "__main__":
    for split in ["train", "valid"]:
        dataset = PLDI2020SmallDataset(
            os.path.expanduser("~/Projects/hse/data"),
            pre_filter=partial(filter_data, max_nodes=512, max_edges=2048),
            split=split
        )
        print("split", split, "length:", len(dataset))
        print("first graph:", dataset[0])
