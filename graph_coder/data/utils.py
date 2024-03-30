from collections import OrderedDict
from typing import Dict, Any, Iterable, List, Optional, Tuple

import networkx
import numpy as np
import torch
from torch_geometric.data import Data


def sample_to_data(sample: Dict[str, Any]) -> Data:
    edge_index = np.concatenate(sample["cg_edges"])
    edge_attr = [np.full((len(e),), i) for i, e in enumerate(sample["cg_edges"])]
    edge_attr = np.concatenate(edge_attr)
    y = np.full((len(sample["cg_node_label_token_ids"]),), 0)
    y[sample["target_node_idxs"]] = sample["variable_target_class"]

    return Data(
        y=torch.tensor(y.astype(np.int32), dtype=torch.long),
        x=torch.tensor(sample["cg_node_label_token_ids"][..., None].astype(np.int32), dtype=torch.long),
        edge_index=torch.tensor(np.transpose(edge_index).astype(np.int32), dtype=torch.long),
        edge_attr=torch.tensor(edge_attr[..., None].astype(np.int32), dtype=torch.long),
        target_node_idxs=torch.tensor(sample["target_node_idxs"].astype(np.int32), dtype=torch.long)
    )


def data_to_sample(data: Data,
                   mapping: Optional[Dict[int, int]] = None,
                   supernodes: Optional[OrderedDict[str, Any]] = None,
                   provenance: str = "?") -> Dict[str, Any]:
    if mapping is None:
        mapping = {}

    new_supernodes = OrderedDict()

    for node in data.target_node_idxs.tolist():
        new_supernodes[str(node)] = supernodes[str(mapping.get(node, node))]

    assert len(data.target_node_idxs) == len(new_supernodes)

    return {
        "Provenance": provenance,
        "target_node_idxs": data.target_node_idxs.numpy().astype(np.uint16),
        "variable_target_class": data.y[data.target_node_idxs].squeeze().numpy().astype(np.uint16),
        "cg_node_label_token_ids": data.x.squeeze().numpy().astype(np.uint16),
        "cg_edges": get_clusters(data.edge_index.t().numpy(), data.edge_attr.squeeze().numpy()),
        "supernodes": new_supernodes
    }


def get_clusters(X: np.ndarray, y: np.ndarray) -> List[np.ndarray]:
    s = np.argsort(y)
    return np.split(X[s], np.unique(y[s], return_index=True)[1][1:])


def split_graph(graph: Data, max_nodes: int, max_edges: int) -> Iterable[Tuple[Data, Dict[int, int]]]:
    g = networkx.Graph()
    g.add_edges_from(
        (a, b, {"type": v}) for (a, b), v in zip(graph.edge_index.t().tolist(), graph.edge_attr.tolist())
    )
    networkx.set_node_attributes(
        g,
        {i: v for i, v in enumerate(graph.x.tolist())},
        "type"
    )

    supernodes = graph.target_node_idxs.tolist()

    for comp in networkx.connected_components(g):
        if len(comp) <= max_nodes:
            subgraph = g.subgraph(comp).copy()
            data, mapping = nx_to_data(graph, subgraph, supernodes)
            if data is not None:
                yield data, mapping
            continue

        for node in supernodes:
            if node not in comp:
                continue
            kwargs = {
                "nodes_so_far": [node],
                "edges_so_far": []
            }

            while (len(kwargs["nodes_so_far"]) < max_nodes
                   and len(kwargs["edges_so_far"]) < max_edges):
                recurse_graph(g, node, max_nodes, max_edges, kwargs)

            subgraph = g.subgraph(kwargs["nodes_so_far"]).copy()
            data, mapping = nx_to_data(graph, subgraph, supernodes)
            if data is not None:
                yield data, mapping


def nx_to_data(graph: Data, subgraph: networkx.Graph, supernodes: List[int]) -> Tuple[Optional[Data], Dict[int, int]]:
    mapping = dict(zip(subgraph.nodes(), range(subgraph.number_of_nodes())))

    target_node_idxs = []
    for i, node in enumerate(supernodes):
        if node in subgraph:
            target_node_idxs.append(mapping[node])

    if len(target_node_idxs) == 0:
        return None, mapping

    new_subgraph = networkx.relabel_nodes(subgraph, mapping)

    edges = []
    edges_data = []

    for a, b, data in new_subgraph.edges.data("type"):
        edges.append((a, b))
        edges_data.append(data[0])

    return Data(
        y=graph.y[list(subgraph.nodes)],
        x=torch.tensor(list(networkx.get_node_attributes(new_subgraph, "type").values()), dtype=torch.long),
        edge_index=torch.tensor(edges, dtype=torch.long).t(),
        edge_attr=torch.tensor(edges_data, dtype=torch.long),
        target_node_idxs=torch.tensor(target_node_idxs, dtype=torch.long)
    ), dict(zip(mapping.values(), mapping.keys()))


def recurse_graph(g: networkx.Graph, node: int, max_nodes: int, max_edges: int, kwargs: Dict[str, Any]):
    for neighbor in g.neighbors(node):
        if (len(kwargs["nodes_so_far"]) >= max_nodes
                or len(kwargs["edges_so_far"]) >= max_edges):
            break
        if neighbor in kwargs["nodes_so_far"]:
            continue
        kwargs["nodes_so_far"].append(neighbor)
        kwargs["edges_so_far"].append((node, neighbor))

        recurse_graph(g, neighbor, max_nodes, max_edges, kwargs)
