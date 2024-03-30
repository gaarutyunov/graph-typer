"""
Modified from https://github.com/microsoft/Graphormer
"""
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Tuple, NamedTuple

import numpy as np
import torch
from dpu_utils.mlutils import Vocabulary
from dpu_utils.utils import RichPath
from fairseq import utils, options, tasks
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import progress_bar

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


class Annotation(NamedTuple):
    provenance: str
    node_id: int
    name: str
    location: Tuple[int, int]
    original_annotation: str
    annotation_type: str
    predicted_annotation_logprob_dist: Dict[str, float]


def eval(args, use_pretrained, checkpoint_path=None, logger=None):
    cfg = convert_namespace_to_omegaconf(args)
    np.random.seed(cfg.common.seed)
    utils.set_torch_seed(cfg.common.seed)

    # initialize task
    task = tasks.setup_task(cfg.task)
    model = task.build_model(cfg.model)
    print(model)

    # load checkpoint
    model_state = torch.load(checkpoint_path)["model"]

    model.load_state_dict(
        model_state, strict=True, model_cfg=cfg.model
    )
    del model_state

    if torch.cuda.is_available():
        model.to(torch.cuda.current_device())
    # load dataset
    split = args.split
    dataset = task.load_dataset(split)
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

    metadata_path = RichPath.create(os.path.expanduser(args.metadata_path))
    saved_data = metadata_path.read_by_file_suffix()
    metadata = saved_data['metadata']

    type_lattice_path = RichPath.create(os.path.expanduser(args.type_lattice_path))
    alias_metadata_path = RichPath.create(os.path.expanduser(args.alias_metadata_path))

    evaluator = TypePredictionEvaluator(type_lattice_path, alias_metadata_path)

    # infer
    with torch.no_grad():
        model.eval()
        for i, sample in enumerate(progress):
            if torch.cuda.is_available():
                sample = utils.move_to_cuda(sample)
            y = model(**sample["net_input"])
            padded_node_mask = get_padded_node_mask(**sample["net_input"])
            target_log_probs = y[padded_node_mask]
            target_log_probs = target_log_probs[sample["net_input"]["batched_data"]["target_node_idxs"]]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            raw_sample = dataset.defn["net_input.batched_data"].dataset.get_sample(sample["net_input"]["batched_data"]["idx"][0].item())
            if len(raw_sample['supernodes']) == 0:
                continue
            provenance = raw_sample.get("Provenance", "?")

            original_annotations = []
            for node_idx, annotation_data in raw_sample['supernodes'].items():
                node_idx = int(node_idx)

                annotation = annotation_data['annotation']
                original_annotations.append((node_idx, annotation, annotation_data['name'], annotation_data['location'], annotation_data['type']))

            assert len(original_annotations) == target_log_probs.shape[0]

            # This is also classification-specific due to class_id_to_class
            for i, (node_idx, node_type, var_name, annotation_location, annotation_type) in enumerate(original_annotations):
                annotation = Annotation(
                    provenance=provenance,
                    node_id=node_idx,
                    name=var_name,
                    original_annotation=node_type,
                    annotation_type=annotation_type,
                    predicted_annotation_logprob_dist={class_id_to_class(metadata, j): target_log_probs[i, j] for j in
                                                       range(target_log_probs.shape[1])},
                    location=annotation_location
                )
                evaluator.add_sample(ground_truth=annotation.original_annotation,
                                     predicted_dist=annotation.predicted_annotation_logprob_dist)

    output = sys.stdout

    if args.output_path:
        output = open(args.output_path, mode="w")

    print(json.dumps(evaluator.metrics(), indent=2, sort_keys=True), file=output)


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
        "--metric",
        type=str,
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
