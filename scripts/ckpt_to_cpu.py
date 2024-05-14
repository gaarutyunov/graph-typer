import argparse
import pathlib

import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_path")
    args = parser.parse_args()
    ckpt_path = pathlib.Path(args.ckpt_path)

    params = torch.load(ckpt_path, map_location="cpu")
    torch.save(params, ckpt_path.with_stem(ckpt_path.stem + "_cpu"))
