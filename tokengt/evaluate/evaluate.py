"""
Modified from https://github.com/microsoft/Graphormer
"""
from typing import Dict

import torch
import numpy as np
from fairseq import checkpoint_utils, utils, options, tasks
from fairseq.logging import progress_bar
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
import ogb
import os
from pathlib import Path
from sklearn.metrics import roc_auc_score

from tokengt.pretrain import load_pretrained_model

import logging

import torch.nn.functional as F


def get_padded_node_mask(sample: Dict[str, torch.Tensor]) -> torch.Tensor:
    node_num = sample["net_input"]["batched_data"]["node_num"]
    edge_num = sample["net_input"]["batched_data"]["edge_num"]
    edge_index = sample["net_input"]["batched_data"]["edge_index"]
    seq_len = [n + e for n, e in zip(node_num, edge_num)]
    b = len(seq_len)
    max_len = max(seq_len)
    device = edge_index.device

    token_pos = torch.arange(max_len, device=device)[None, :].expand(b, max_len)  # [B, T]
    node_num = torch.tensor(node_num, device=device, dtype=torch.long)[:, None]  # [B, 1]

    return torch.less(token_pos, node_num)


def eval(args, use_pretrained, checkpoint_path=None, logger=None):
    cfg = convert_namespace_to_omegaconf(args)
    np.random.seed(cfg.common.seed)
    utils.set_torch_seed(cfg.common.seed)

    # initialize task
    task = tasks.setup_task(cfg.task)
    model = task.build_model(cfg.model)
    print(model)

    # load checkpoint
    if use_pretrained:
        model_state = load_pretrained_model(cfg.task.pretrained_model_name)
    else:
        model_state = torch.load(checkpoint_path)["model"]

    model.load_state_dict(
        model_state, strict=True, model_cfg=cfg.model
    )
    del model_state

    if torch.cuda.is_available():
        model.to(torch.cuda.current_device())
    # load dataset
    split = args.split
    task.load_dataset(split)
    batch_iterator = task.get_batch_iterator(
        dataset=task.dataset(split),
        max_tokens=cfg.dataset.max_tokens_valid,
        max_sentences=cfg.dataset.batch_size_valid,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            model.max_positions(),
        ),
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
        seed=cfg.common.seed,
        num_workers=cfg.dataset.num_workers,
        epoch=0,
        data_buffer_size=cfg.dataset.data_buffer_size,
        disable_iterator_cache=False,
    )
    itr = batch_iterator.next_epoch_itr(
        shuffle=False, set_dataset_epoch=False
    )
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple")
    )

    # infer
    y_pred = []
    y_true = []
    with torch.no_grad():
        model.eval()
        for i, sample in enumerate(progress):
            if torch.cuda.is_available():
                sample = utils.move_to_cuda(sample)
            y = model(**sample["net_input"])
            padded_node_mask = get_padded_node_mask(sample)
            y = y[padded_node_mask]
            y_pred.extend(y.detach().cpu())
            y_true.extend(sample["target"].detach().cpu())
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # save predictions
    y_pred = torch.cat([t[None, ...] for t in y_pred], dim=0)
    y_true = torch.Tensor(y_true)

    # evaluate pretrained models
    if use_pretrained:
        if cfg.task.pretrained_model_name == "pcqm4mv1_sgt_base":
            evaluator = ogb.lsc.PCQM4MEvaluator()
            input_dict = {'y_pred': y_pred, 'y_true': y_true}
            result_dict = evaluator.eval(input_dict)
            logger.info(f'PCQM4Mv1Evaluator: {result_dict}')
        elif cfg.task.pretrained_model_name == "pcqm4mv2_sgt_base":
            evaluator = ogb.lsc.PCQM4Mv2Evaluator()
            input_dict = {'y_pred': y_pred, 'y_true': y_true}
            result_dict = evaluator.eval(input_dict)
            logger.info(f'PCQM4Mv2Evaluator: {result_dict}')
    else:
        if args.metric == "auc":
            with torch.no_grad():
                classes = torch.unique(y_true).type(torch.long)
                y_pred = y_pred[:, classes]
                y_pred = F.softmax(y_pred, dim=1)
            auc = roc_auc_score(y_true, y_pred, multi_class='ovr', average='weighted', labels=classes)
            logger.info(f"auc: {auc}")
        elif args.metric == "mae":
            mae = (y_true - y_pred).abs().mean().item()
            logger.info(f"mae: {mae}")
        else:
            raise ValueError(f"Unsupported metric {args.metric}")


def main():
    parser = options.get_training_parser()
    parser.add_argument(
        "--split",
        type=str,
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
    )
    parser.add_argument(
        "--metric",
        type=str,
    )
    args = options.parse_args_and_arch(parser, modify_parser=None)
    logger = logging.getLogger(__name__)
    if args.pretrained_model_name != "none":
        eval(args, True, logger=logger)
    elif hasattr(args, "checkpoint_path"):
        eval(args, False, checkpoint_path=args.checkpoint_path, logger=logger)
    elif hasattr(args, "save_dir"):
        for checkpoint_fname in os.listdir(args.save_dir):
            checkpoint_path = Path(args.save_dir) / checkpoint_fname
            logger.info(f"evaluating checkpoint file {checkpoint_path}")
            eval(args, False, checkpoint_path, logger)


if __name__ == '__main__':
    main()
