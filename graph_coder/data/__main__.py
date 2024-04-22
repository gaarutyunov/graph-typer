import argparse
from functools import partial

from graph_coder.data.pldi2020_dataset import PLDI2020Dataset, filter_data


def main(args: argparse.Namespace) -> None:
    dataset = PLDI2020Dataset(
        args.dataset_root,
        split=args.split,
        pre_filter=partial(filter_data, max_nodes=args.max_nodes, max_edges=args.max_edges)
    ).load_meta()
    print(dataset)
    print(dataset[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, default="~/data")
    parser.add_argument("--split", choices=["train", "valid", "test"], type=str, default="train")
    parser.add_argument("--max-nodes", default=512, type=int, help="Max number of nodes in graph")
    parser.add_argument("--max-edges", default=2048, type=int, help="Max number of edges in graph")

    args = parser.parse_args()
    main(args)
