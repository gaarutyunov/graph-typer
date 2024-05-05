from functools import partial
from typing import Dict, Any, Iterable, Tuple

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data


def nx_to_sample(g: nx.Graph) -> Dict[str, Any]:
    return g.graph


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


def split_nx(g: nx.Graph, max_nodes: int, max_edges: int) -> Iterable[nx.Graph]:
    if g.number_of_nodes() <= max_nodes and g.number_of_edges() <= max_edges:
        yield _relabel_nx(g, g.graph["supernodes"])
        return

    for comp in nx.connected_components(g):
        subgraph = g.subgraph(comp).copy()
        if subgraph.number_of_nodes() <= max_nodes and subgraph.number_of_edges() <= max_edges:
            yield _relabel_nx(subgraph, subgraph.graph["supernodes"])
            continue

        for node, annotation in subgraph.graph["supernodes"].items():
            node = int(node)
            if node not in subgraph:
                continue
            kwargs = {
                "nodes": [node],
                "edges": 0
            }

            while (len(kwargs["nodes"]) < max_nodes
                   and kwargs["edges"] < max_edges):
                recurse_nx(g, node, max_nodes, max_edges, kwargs)

            new_subgraph = g.subgraph(kwargs["nodes"]).copy()

            yield _relabel_nx(new_subgraph, new_subgraph.graph["supernodes"])


def _relabel_nx(g: nx.Graph, annotations: Dict[str, Any]) -> nx.Graph:
    counter = 0
    supernodes = {}

    def relabel(node: int, ctx: Dict[str, Any]) -> int:
        counter_ = ctx["counter"]
        ctx["counter"] += 1

        if str(node) in annotations:
            supernodes[str(counter_)] = annotations[str(node)]

        return counter_

    new_graph = nx.relabel_nodes(g, partial(relabel, ctx={"counter": counter}))
    new_graph.graph["supernodes"] = supernodes

    return new_graph


def recurse_nx(g: nx.Graph, node: int, max_nodes: int, max_edges: int, kwargs: Dict[str, Any]):
    for neighbor in g.neighbors(node):
        if (len(kwargs["nodes"]) >= max_nodes
                or kwargs["edges"] >= max_edges):
            break
        if neighbor in kwargs["nodes"]:
            continue
        kwargs["nodes"].append(neighbor)
        kwargs["edges"] += 1

        recurse_nx(g, neighbor, max_nodes, max_edges, kwargs)


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
