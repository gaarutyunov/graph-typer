import argparse

from graph_coder.data.pldi2020_dataset import PLDI2020Dataset


def main(args: argparse.Namespace) -> None:
    dataset = PLDI2020Dataset(
        args.dataset_root,
        split=args.split
    )
    print(dataset)
    print(dataset[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, default="~/data")
    parser.add_argument("--split", choices=["train", "valid", "test"], type=str, default="train")

    args = parser.parse_args()
    main(args)
