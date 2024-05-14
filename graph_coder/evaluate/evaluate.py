"""
Modified from https://github.com/microsoft/Graphormer
"""
import json
import logging
import os
import pathlib
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from dpu_utils.mlutils import Vocabulary
from dpu_utils.utils import RichPath, ChunkWriter
from fairseq import utils, options, tasks
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import progress_bar

from typilus import Annotation
from typilus.model.utils import ignore_type_annotation
from typilus.utils.evaluator import TypePredictionEvaluator


def get_padded_node_mask(batched_data: Dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
    node_num = batched_data["node_num"]
    edge_num = batched_data["edge_num"]
    edge_index = batched_data["edge_index"]
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
    try:
        model_state = torch.load(checkpoint_path)["model"]

        model.load_state_dict(
            model_state, strict=True, model_cfg=cfg.model
        )
        del model_state
    except FileNotFoundError:
        logger.info("no checkpoint found at %s", checkpoint_path)

    if torch.cuda.is_available():
        model.to(torch.cuda.current_device())
    # load dataset
    split = args.split
    dataset = task.load_dataset(split)
    batch_iterator = task.get_batch_iterator(
        dataset=dataset,
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
        log_format="tqdm",
        log_interval=cfg.common.log_interval,
    )

    metadata_path = RichPath.create(os.path.expanduser(args.metadata_path))
    saved_data = metadata_path.read_by_file_suffix()
    metadata = saved_data['metadata']

    type_lattice_path = RichPath.create(os.path.expanduser(args.type_lattice_path))
    alias_metadata_path = RichPath.create(os.path.expanduser(args.alias_metadata_path))

    evaluator = TypePredictionEvaluator(
        type_lattice_path,
        alias_metadata_path,
        top_n=args.top_n
    )

    writer: Optional[ChunkWriter] = None

    if args.output_predictions:
        pathlib.Path(args.output_dir).mkdir(exist_ok=True, parents=True)

        writer = ChunkWriter(
            args.output_dir,
            file_prefix="prediction_",
            max_chunk_size=1000,
            file_suffix=".pkl.gz",
        )

    # infer
    with torch.no_grad():
        model.eval()
        for sample, raw_sample in progress:
            if torch.cuda.is_available():
                sample = utils.move_to_cuda(sample)
            y = model(batched_data=sample, masked_tokens=sample.get("masked_tokens"))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            raw_sample = raw_sample[0]

            provenance = raw_sample.get("Provenance", "?")

            masked_tokens = sample["masked_tokens"].squeeze()
            node_idxs = masked_tokens.nonzero()
            target_annotations = []
            input_annotations = []

            for i, (node_idx, annotation_data) in enumerate(raw_sample["supernodes"].items()):
                node_idx = int(node_idx)

                if node_idx not in node_idxs:
                    if args.output_predictions:
                        input_annotations.append(dict(
                            node_id=node_idx,
                            name=annotation_data['name'],
                            original_annotation=annotation_data['annotation'],
                            annotation_type=annotation_data['type'],
                            location=annotation_data['location']
                        ))
                    continue

                predicted_dist = {class_id_to_class(metadata, j): y[0, node_idx, j].item() for j in
                                  range(y.size(-1))}

                evaluator.add_sample(ground_truth=annotation_data['annotation'],
                                     predicted_dist=predicted_dist)

                if args.output_predictions:
                    target_annotations.append(dict(
                        node_id=node_idx,
                        name=annotation_data['name'],
                        original_annotation=annotation_data['annotation'],
                        annotation_type=annotation_data['type'],
                        location=annotation_data['location'],
                        predicted_dist=predicted_dist
                    ))

            if args.output_predictions:
                writer.add({
                    "provenance": provenance,
                    "input_annotations": input_annotations,
                    "target_annotations": target_annotations,
                })

    output = sys.stdout

    if args.output_path:
        output = open(args.output_path, mode="w")

    print(json.dumps(evaluator.metrics(), indent=2, sort_keys=True), file=output)

    if args.output_predictions:
        writer.close()


def class_id_to_class(metadata: Dict[str, Vocabulary], class_id: int) -> str:
    name = metadata['annotation_vocab'].get_name_for_id(class_id)
    if metadata['annotation_vocab'].is_unk(name):
        return 'typing.Any'
    return name


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
        "--metadata-path",
        type=str,
    )
    parser.add_argument(
        "--type-lattice-path",
        type=str,
    )
    parser.add_argument(
        "--alias-metadata-path",
        type=str,
    )
    parser.add_argument(
        "--output-path",
        type=str,
    )
    parser.add_argument(
        "--output-predictions",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--no-output-predictions",
        action="store_false",
        dest="output_predictions",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="predictions",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=1
    )
    args = options.parse_args_and_arch(parser, modify_parser=None)
    logger = logging.getLogger(__name__)
    if hasattr(args, "checkpoint_path"):
        eval(args, False, checkpoint_path=args.checkpoint_path, logger=logger)
    elif hasattr(args, "save_dir"):
        for checkpoint_fname in os.listdir(args.save_dir):
            checkpoint_path = Path(args.save_dir) / checkpoint_fname
            logger.info(f"evaluating checkpoint file {checkpoint_path}")
            eval(args, False, checkpoint_path, logger)


if __name__ == '__main__':
    main()
