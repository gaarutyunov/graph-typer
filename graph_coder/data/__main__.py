import argparse

from graph_coder.data.pldi2020_dataset import PLDI2020Dataset
from tokengt.data import DATASET_REGISTRY


def main(args: argparse.Namespace) -> None:
    dataset: PLDI2020Dataset = DATASET_REGISTRY.get(args.dataset_name)(
        args,
        slpit=args.split
    ).load_meta()
    print(dataset)
    print(dataset[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str, default="pldi2020", help="Dataset name")
    parser.add_argument("--dataset-root", type=str, default="~/data", help="Dataset root path")
    parser.add_argument("--split", choices=["train", "valid", "test"], type=str, default="train")
    parser.add_argument("--max-nodes", default=512, type=int, help="Max number of nodes in graph")
    parser.add_argument("--max-edges", default=2048, type=int, help="Max number of edges in graph")
    parser.add_argument("--num-classes", default=100, type=int, help="Num classes")

    args = parser.parse_args()
    main(args)
