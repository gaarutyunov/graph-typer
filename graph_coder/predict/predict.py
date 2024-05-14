import argparse
import logging
import pathlib
import re
from collections import defaultdict
from typing import Dict, Any, Tuple, List, Callable

import networkx as nx
import numpy as np
import torch
from dpu_utils.utils import RichPath
from fairseq import utils, tasks, options
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from torch_geometric.data import Data

from tokengt.data.collator import collator
from tokengt.data.wrapper import preprocess_item
from typilus import class_id_to_class
from typilus.graph_generator.graphgenerator import AstGraphGenerator
from typilus.graph_generator.type_lattice_generator import TypeLatticeGenerator

logger = logging.getLogger(__name__)


STRING_LITERAL_REGEX = re.compile('^[fub]?["\'](.*)["\']$')
STRING_LITERAL = '$StrLiteral$'
INT_LITERAL = '$IntLiteral$'
FLOAT_LITERAL = '$FloatLiteral$'


def filter_literals(token: str) -> str:
    try:
        v = int(token)
        return INT_LITERAL
    except ValueError:
        pass
    try:
        v = float(token)
        return FLOAT_LITERAL
    except ValueError:
        pass
    string_lit = STRING_LITERAL_REGEX.match(token)
    if string_lit:
        return STRING_LITERAL
    return token


def sample_to_nx(sample: Dict[str, Any]) -> nx.Graph:
    # initiate graph
    edges = []
    for v, edges_ in enumerate(sample['cg_edges']):
        for (a, b) in edges_:
            edges.append((a, b, {"type": v}))
    g = nx.Graph()
    g.add_edges_from(edges)

    # add node type
    nx.set_node_attributes(g, {k: v for k, v in enumerate(sample["cg_node_label_token_ids"])}, "type")

    # add node label
    y = np.full((len(sample["cg_node_label_token_ids"]),), -100)
    y[sample["target_node_idxs"]] = sample["variable_target_class"]
    nx.set_node_attributes(g, {k: v for k, v in enumerate(y)}, "label")

    # add supernodes
    g.graph["supernodes"] = sample["raw_data"]["supernodes"]

    # add filename
    g.graph["Provenance"] = sample.get("Provenance", "?")

    return g


def nx_to_data(graph: nx.Graph) -> Data:
    edges = []
    edges_data = []

    for a, b, data in graph.edges.data("type"):
        edges.append((a, b))
        edges_data.append(data)

    nodes = []
    labels = []

    for node, data in graph.nodes(data=True):
        nodes.append([
            data["type"], 0 if data["label"] == -100 else data["label"]
        ])
        labels.append(data["label"])

    return Data(
        y=torch.tensor(labels, dtype=torch.long),
        x=torch.tensor(nodes, dtype=torch.long),
        edge_index=torch.tensor(edges, dtype=torch.long).t(),
        edge_attr=torch.tensor(edges_data, dtype=torch.long),
    )


def graph_to_sample(raw_sample: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
    result_holder = {}
    graph_node_labels = raw_sample['nodes']
    num_nodes = len(graph_node_labels)

    # Translate node label, either using the token vocab or into a character representation:
    name = "cg_node_label"
    node_labels = np.zeros((num_nodes,), dtype=np.uint16)
    for (node, label) in enumerate(graph_node_labels):
        if metadata[f'{name}_vocab'].is_unk(label):
            label = filter_literals(label)  # UNKs that are literals will be converted to special symbols.
        node_labels[node] = metadata[f'{name}_vocab'].get_id_or_unk(label)
    result_holder[f'{name}_token_ids'] = node_labels
    result_holder['num_nodes'] = num_nodes

    assert 'edges' in raw_sample

    # Split edges according to edge_type and count their numbers:
    edges = [[] for _ in metadata['cg_edge_type_dict']]

    num_edge_types = len(metadata['cg_edge_type_dict'])
    num_incoming_edges_per_type = np.zeros((num_nodes, num_edge_types), dtype=np.uint16)
    num_outgoing_edges_per_type = np.zeros((num_nodes, num_edge_types), dtype=np.uint16)

    edges_per_type = {}
    for edge_type, edge_dict in raw_sample['edges'].items():
        edge_list = []
        for from_idx, to_idxs in edge_dict.items():
            from_idx = int(from_idx)
            for to_idx in to_idxs:
                edge_list.append((from_idx, to_idx))
        edges_per_type[edge_type] = edge_list

    for (e_type, e_type_idx) in metadata['cg_edge_type_dict'].items():
        if e_type in edges_per_type and len(edges_per_type[e_type]) > 0:
            edges[e_type_idx] = np.array(edges_per_type[e_type], dtype=np.int32)
        else:
            edges[e_type_idx] = np.zeros((0, 2), dtype=np.int32)

        # TODO: This is needed only in some configurations of the GNN!
        num_incoming_edges_per_type[:, e_type_idx] = np.bincount(edges[e_type_idx][:, 1],
                                                                 minlength=num_nodes)
        num_outgoing_edges_per_type[:, e_type_idx] = np.bincount(edges[e_type_idx][:, 0],
                                                                 minlength=num_nodes)
    assert not all(len(e) == 0 for e in edges)
    result_holder['cg_edges'] = edges
    result_holder['cg_num_incoming_edges_per_type'] = num_incoming_edges_per_type
    result_holder['cg_num_outgoing_edges_per_type'] = num_outgoing_edges_per_type
    target_node_idxs, target_class = [], []
    for node_idx, annotation_data in raw_sample['supernodes'].items():
        node_idx = int(node_idx)
        annotation = annotation_data['annotation']

        target_node_idxs.append(node_idx)
        class_id = metadata['annotation_vocab'].get_id_or_unk(annotation)
        target_class.append(class_id)

    result_holder['target_node_idxs'] = np.array(target_node_idxs, dtype=np.uint16)
    result_holder['variable_target_class'] = np.array(target_class, dtype=np.uint16)

    result_holder["raw_data"] = raw_sample

    return result_holder


def load_model(args: argparse.Namespace) -> Callable:
    cfg = convert_namespace_to_omegaconf(args)
    np.random.seed(cfg.common.seed)
    utils.set_torch_seed(cfg.common.seed)

    # initialize task
    task = tasks.setup_task(cfg.task)
    model = task.build_model(cfg.model)
    print(model)

    # load checkpoint
    try:
        model_state = torch.load(args.checkpoint_path)["model"]

        model.load_state_dict(
            model_state, strict=True, model_cfg=cfg.model
        )
        del model_state
    except FileNotFoundError:
        raise "no checkpoint found at %s".format(args.checkpoint_path)

    return model


mask_type = ": \"__MASK__\""
mask_return = " -> \"__MASK__\""


def get_masked_locations(code: str) -> Tuple[str, List[Tuple[int, int]]]:
    lines = []
    locations = []

    for i, line in enumerate(code.splitlines()):
        for _ in range(line.count(mask_type)):
            idx = line.index(mask_type)
            locations.append((i+1, idx))
            line = line[:idx] + line[idx+len(mask_type):]

        for _ in range(line.count(mask_return)):
            idx = line.index(mask_return)
            locations.append((i+1, idx))
            line = line[:idx] + line[idx+len(mask_return):]
            
        lines.append(line)

    return "\n".join(lines), locations


def main() -> None:
    args = options.parse_args_and_arch(setup(), modify_parser=None)
    type_lattice = TypeLatticeGenerator(str(args.alias_metadata_path.expanduser()))
    metadata = RichPath.create(str(args.metadata_path.expanduser())).read_by_file_suffix()
    source_code, masked_locations = get_masked_locations(args.input.read_text())

    visitor = AstGraphGenerator(source_code, type_lattice)
    graph = visitor.build()

    metadata = metadata["metadata"]

    sample = graph_to_sample(graph, metadata)
    nx = sample_to_nx(sample)
    data = nx_to_data(nx)
    data.idx = 1

    item = preprocess_item(data)

    masked_tokens = torch.zeros((item.node_data.size(0) + item.edge_data.size(0),), dtype=torch.bool)

    node_idxs = []

    for node_idx, annotation_data in nx.graph["supernodes"].items():
        location = (annotation_data["location"][0], annotation_data["location"][1] + len(annotation_data["name"]))
        if location in masked_locations:
            node_idx = int(node_idx)
            masked_tokens[node_idx] = True
            node_idxs.append(node_idx)

    model_input = {
        "batched_data": collator([item]),
        "masked_tokens": masked_tokens[None, ...],
    }

    model = load_model(args)

    model_output = model(**model_input)

    original_annotations = []
    for node_idx, annotation_data in nx.graph['supernodes'].items():
        node_idx = int(node_idx)
        if node_idx not in node_idxs:
            continue

        annotation = annotation_data['annotation']
        original_annotations.append((node_idx, annotation, annotation_data['name'], annotation_data['location'], annotation_data['type']))

    assert len(original_annotations) == model_output.shape[0]

    predicted_types = defaultdict(dict)

    # This is also classification-specific due to class_id_to_class
    for i, (node_idx, node_type, var_name, annotation_location, annotation_type) in enumerate(original_annotations):
        predicted_dist = {class_id_to_class(metadata, j): model_output[i, j].data for j in range(model_output.shape[1])}

        predicted = sorted(predicted_dist.items(), key=lambda item: item[1], reverse=True)

        predicted_types[annotation_location[0]][annotation_location[1]] = {
            "type": predicted[0][0],
            "annotation_type": annotation_type,
            "sorted": predicted
        }

    lines = source_code.splitlines()

    for i, line in enumerate(lines):
        if (i + 1) in predicted_types:
            for idx, annotation in predicted_types[i + 1].items():
                if annotation["annotation_type"] in ["variable", "parameter"]:
                    new_type = f": {annotation['type']}"
                else:
                    new_type = f"-> {annotation['type']}"
                lines[i] = line[:idx-1] + new_type + line[idx-1:]

    args.output.write_text("\n".join(lines))


def setup() -> argparse.ArgumentParser:
    parser = options.get_training_parser()

    parser.add_argument(
        "--input",
        type=pathlib.Path,
        default="example.py"
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default="example_typed.py"
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="scripts/ckpts/pldi2020-graph_coder_encoder_base-120524/checkpoint_best_cpu.pt"
    )
    parser.add_argument(
        "--alias-metadata-path",
        type=pathlib.Path,
        default="~/data/typingRules.json",
    )
    parser.add_argument(
        "--metadata-path",
        type=pathlib.Path,
        default="~/data/tensorised-data/train/metadata.pkl.gz",
    )

    return parser


if __name__ == "__main__":
    main()

