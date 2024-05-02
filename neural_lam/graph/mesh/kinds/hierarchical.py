import scipy
import numpy as np
import networkx
from loguru import logger
from torch_geometric.utils.convert import from_networkx as pyg_from_networkx
import torch

from .. import mesh as mesh_graph
from ...networkx_utils import prepend_node_index


def sort_nodes_internally(nx_graph):
    # For some reason the networkx .nodes() return list can not be sorted,
    # but this is the ordering used by pyg when converting.
    # This function fixes this.
    H = networkx.DiGraph()
    H.add_nodes_from(sorted(nx_graph.nodes(data=True)))
    H.add_edges_from(nx_graph.edges(data=True))
    return H


def from_networkx_with_start_index(nx_graph, start_index):
    pyg_graph = pyg_from_networkx(nx_graph)
    pyg_graph.edge_index += start_index
    return pyg_graph


def create_hierarchical_multiscale_mesh_graph(
    xy, refinement_factor: int, max_num_levels: int
):
    G_all_levels: list[networkx.DiGraph] = mesh_graph.create_multirange_2d_mesh_graphs(
        max_num_levels=max_num_levels,
        xy=xy,
        refinement_factor=refinement_factor,
    )
    n_mesh_levels = len(G_all_levels)
    # Relabel nodes of each level with level index first

    G_all_levels = [
        prepend_node_index(graph, level_i)
        for level_i, graph in enumerate(G_all_levels)
    ]

    num_nodes_level = np.array([len(g_level.nodes) for g_level in G_all_levels])
    # First node index in each level in the hierarchical graph
    first_index_level = np.concatenate(
        (np.zeros(1, dtype=int), np.cumsum(num_nodes_level[:-1]))
    )

    # Create inter-level mesh edges
    up_graphs = []
    down_graphs = []
    for G_from, G_to in zip(
        G_all_levels[1:],
        G_all_levels[:-1],
    ):
        from_level = G_from.graph["level"]
        to_level = G_to.graph["level"]

        # start out from graph at from level
        G_down = G_from.copy()
        G_down.clear_edges()
        G_down = networkx.DiGraph(G_down)

        # Add nodes of to level
        G_down.add_nodes_from(G_to.nodes(data=True))

        # build kd tree for mesh point pos
        # order in vm should be same as in vm_xy
        v_to_list = list(G_to.nodes)
        v_from_list = list(G_from.nodes)
        v_from_xy = np.array([xy for _, xy in G_from.nodes.data("pos")])
        kdt_m = scipy.spatial.KDTree(v_from_xy)

        # add edges from mesh to grid
        for v in v_to_list:
            # find 1(?) nearest neighbours (index to vm_xy)
            neigh_idx = kdt_m.query(G_down.nodes[v]["pos"], 1)[1]
            u = v_from_list[neigh_idx]

            # add edge from mesh to grid
            G_down.add_edge(u, v)
            d = np.sqrt(
                np.sum((G_down.nodes[u]["pos"] - G_down.nodes[v]["pos"]) ** 2)
            )
            G_down.edges[u, v]["len"] = d
            G_down.edges[u, v]["vdiff"] = (
                G_down.nodes[u]["pos"] - G_down.nodes[v]["pos"]
            )
            G_down.edges[u, v]["levels"] = f"{from_level}>{to_level}"

        G_up = networkx.DiGraph()
        G_up.add_nodes_from(G_down.nodes(data=True))
        for (u, v, data) in G_down.edges(data=True):
            data["levels"] = f"{to_level}>{from_level}"
            G_up.add_edge(v, u, **data)

        up_graphs.append(G_up)
        down_graphs.append(G_down)
        
    G_up_all = networkx.compose_all(up_graphs)
    G_down_all = networkx.compose_all(down_graphs)
    
    G_m2m = networkx.union_all(G_all_levels)
    
    return dict(m2m=G_m2m, mesh_down=G_down_all, mesh_up=G_up_all)