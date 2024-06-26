import os

import torch
from torch.hub import load_state_dict_from_url
import torch.distributed as dist

PRETRAINED_MODEL_URLS = {
}

PRETRAINED_MODEL_PATHS = {
    'pcqv2-tokengt-orf64-trained': 'ckpts/pcqv2-tokengt-orf64-trained/checkpoint_best.pt',
    'pcqv2-tokengt-lap16-trained': 'ckpts/pcqv2-tokengt-lap16-trained/checkpoint_best.pt'
}


def load_pretrained_model(pretrained_model_name_or_path):
    if os.path.exists(pretrained_model_name_or_path):
        return torch.load(pretrained_model_name_or_path)["model"]
    if pretrained_model_name_or_path not in PRETRAINED_MODEL_URLS:
        if pretrained_model_name_or_path not in PRETRAINED_MODEL_PATHS:
            raise ValueError("Unknown pretrained model name %s", pretrained_model_name_or_path)
        return torch.load(PRETRAINED_MODEL_PATHS[pretrained_model_name_or_path])["model"]
    if not dist.is_initialized():
        return load_state_dict_from_url(
            PRETRAINED_MODEL_URLS[pretrained_model_name_or_path],
            progress=True
        )["model"]
    else:
        pretrained_model = load_state_dict_from_url(
            PRETRAINED_MODEL_URLS[pretrained_model_name_or_path],
            progress=True,
            file_name=f"{pretrained_model_name_or_path}_{dist.get_rank()}"
        )["model"]
        dist.barrier()
        return pretrained_model
