"""
Routines for creating the graph used in the neural LAM model.

The main function in this module is `create_all_graph_components`, which
creates all the graph components used in the neural LAM model. The graph
components are:
- Grid-to-mesh graph (g2m)
- Mesh-to-grid graph (m2g)
- Mesh-to-mesh graph (m2m)



The principles for the graph generation are:

- 


1. Create networkx.DiGraph for each component of the graph.
    Each graph should contain node attributes `pos` (position) and
    edge attributes `len` (length) and `vdiff` (vector difference).

2. Merge components into a single graph and create globally unique node indices.

3. Add unique node indices to the graph components as node attributes.

2. Convert to PyTorch geometric graph, keeping both node (`pos`) and edge (`len`, `vdiff`) attributes
3. Save to disk as `{name}_edge_index.pt` and `{name}_features.pt` (only `vdiff` feature for now)


TODO:

- introduce split graph edges by attribute when saving
- merge all hierarchical graph components, but add `levels` attribute with `{from}>{to}` value
- rename graph "kind" as "m2m_connectivity"
- add `m2g` and `g2m` "method" and "method_kwargs" attributes
"""

import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import networkx
import numpy as np
import scipy.spatial
import torch
from torch_geometric.utils.convert import from_networkx as pyg_from_networkx
from loguru import logger

from ..datasets.meps.weather_dataset import WeatherDataset
from .save import save_edges, save_edges_list
from .mesh.kinds.flat import create_flat_multiscale_mesh_graph
from .networkx_utils import prepend_node_index
from . import mesh as mesh_graph
from . import grid as grid_graph


def create_all_graph_components(
    xy: np.ndarray,
    m2m_connectivity: str,
    merge_components: bool = True,
    m2m_connectivity_kwargs=dict(
        refinement_factor=3,
        max_num_levels=None,
    ),
    m2g_method="nearest_neighbours",
    m2g_method_kwargs=dict(max_num_neighbours=4),
    g2m_method="nearest_neighbour",
    g2m_method_kwargs={},
):
    graph_components: dict[networkx.DiGraph] = {}

    if m2m_connectivity == "hierarchical":
        # hierarchical mesh graph have three sub-graphs:
        # `m2m` (mesh-to-mesh), `mesh_up` (up edge connections) and `mesh_down` (down edge connections)
        hierarchical_graph_components = (
            mesh_graph.create_hierarchical_multiscale_mesh_graph(
                xy=xy,
                **m2m_connectivity_kwargs,
            )
        )
        graph_components.update(hierarchical_graph_components)
    elif m2m_connectivity == "flat_multiscale":
        graph_components["m2m"] = create_flat_multiscale_mesh_graph(
            xy=xy,
            **m2m_connectivity_kwargs,
        )
    else:
        raise NotImplementedError(f"Kind {m2m_connectivity} not implemented")

    G_grid = grid_graph.create_grid_graph_nodes(xy=xy)

    G_g2m = connect_nodes_across_graphs(
        G_source=G_grid,
        G_target=graph_components["m2m"],
        method=g2m_method,
        **g2m_method_kwargs,
    )
    graph_components["g2m"] = G_g2m

    # TODO: for the hierarchical mesh graph, we might want to only connect to the bottom layer here
    G_m2g = connect_nodes_across_graphs(
        G_source=graph_components["m2m"],
        G_target=G_grid,
        method=m2g_method,
        **m2g_method_kwargs
    )
    graph_components["m2g"] = G_m2g

    # add graph component identifier to each edge in each component graph
    for name, graph in graph_components.items():
        for edge in graph.edges:
            graph.edges[edge]["component"] = name

    if not merge_components:
        return graph_components
    else:
        # merge to single graph
        G_tot = networkx.compose(*graph_components.values())

        return G_tot


def save_graph_components(
    graph_components: dict[networkx.DiGraph], graph_dir_path
):

    # relabel nodes to integers (sorted)
    # G_int = networkx.convert_node_labels_to_integers(
    # G_tot, first_label=0, ordering="sorted"
    # )

    # # Graph to use in g2m and m2g
    # G_bottom_mesh = G_tot
    # all_mesh_nodes = G_tot.nodes(data=True)

    # # export the nx graph to PyTorch geometric
    # pyg_m2m = pyg_from_networkx(G_int)
    # m2m_graphs = [pyg_m2m]
    # return m2m_graphs, G_bottom_mesh, all_mesh_nodes

    # relabel nodes to integers (sorted)
    G_int = networkx.convert_node_labels_to_integers(
        G_tot, first_label=0, ordering="sorted"
    )

    # radius within which grid nodes are associated with a mesh node
    # (in terms of mesh distance)
    DM_SCALE = 0.67

    # distance between mesh nodes, OBS: assumes isotropic grid
    dm = np.sqrt(
        np.sum((vm.data("pos")[(0, 1, 0)] - vm.data("pos")[(0, 0, 0)]) ** 2)
    )

    if args.plot:
        plot_graph(pyg_g2m, title="Grid-to-mesh")
        plt.show()

    if args.plot:
        plot_graph(pyg_m2g, title="Mesh-to-grid")
        plt.show()

    mesh_pos = [graph.pos.to(torch.float32) for graph in m2m_graphs]

    # Divide mesh node pos by max coordinate of grid cell
    pos_max = torch.tensor(np.max(np.abs(xy), axis=(1, 2)))
    mesh_pos = [pos / pos_max for pos in mesh_pos]

    # Save mesh positions
    torch.save(
        mesh_pos, os.path.join(graph_dir_path, "mesh_features.pt")
    )  # mesh pos, in float32

    # Save m2m edges
    save_edges_list(m2m_graphs, "m2m", graph_dir_path)

    # Save g2m and m2g everything
    # g2m
    save_edges(pyg_g2m, "g2m", graph_dir_path)
    # m2g
    save_edges(pyg_m2g, "m2g", graph_dir_path)

    if args.plot:
        plot_graph(pyg_m2m, title="Mesh-to-mesh")
        plt.show()


def connect_nodes_across_graphs(
    G_source,
    G_target,
    method="nearest_neighbour",
    max_dist=None,
    max_num_neighbours=None,
):
    """
    Create a new graph containing the nodes in `G_source` and `G_target` and add
    directed edges from nodes in `G_source` to nodes in `G_target` based on the
    method specified.

    This can for example be used to create mesh-to-grid (m2g) connections where
    each grid node (the target) has connections to it from the 4 nearest mesh nodes (the source) by using
    the `nearest_neighbours` method with `max_num_neighbours=4`

    Parameters
    ----------
    G_source : networkx.Graph
        Source graph, edge connections are made from nodes in this graph (existing edges are ignored)
    G_target : networkx.Graph
        Target graph, edge connections are made to nodes in this graph (existing edges are ignored)
    method : str
        Method to use for finding neighbours in `G_source` for each node in `G_target`.
        Options are:
        - "nearest_neighbour": Find the nearest neighbour in `G_target` for each node in `G_source`
        - "nearest_neighbours": Find the `max_num_neighbours` nearest neighbours in `G_target` for each node in `G_source`
        - "within_radius": Find all neighbours in `G_target` within a distance of `max_dist` from each node in `G_source`
    max_dist : float
        Maximum distance to search for neighbours in `G_target` for each node in `G_source`
    max_num_neighbours : int
        Maximum number of neighbours to search for in `G_target` for each node in `G_source`

    Returns
    -------
    networkx.DiGraph
        Graph containing the nodes in `G_source` and `G_target` and directed edges
        from nodes in `G_source` to nodes in `G_target`
    """
    source_nodes_list = list(G_source.nodes)
    target_nodes_list = list(G_target.nodes)

    # build kd tree for source nodes (e.g. the mesh nodes when constructing m2g)
    xy_source = np.array(
        [G_source.nodes[node]["pos"] for node in G_source.nodes]
    )
    kdt_s = scipy.spatial.KDTree(xy_source)

    def _find_neighbour_node_idxs_in_source_mesh(from_node):
        xy_target = G_target.nodes[from_node]["pos"]

        if method == "nearest_neighbour":
            neigh_idx = kdt_s.query(xy_target, 1)[1]
            if max_dist is not None or max_num_neighbours is not None:
                raise Exception(
                    "to use `nearest_neighbour` you should not set `max_dist` or `max_num_neighbours`"
                )
            return [neigh_idx]
        elif method == "nearest_neighbours":
            if max_num_neighbours is None:
                raise Exception(
                    "to use `nearest_neighbours` you should set the max number with `max_num_neighbours`"
                )
            if max_dist is not None:
                raise Exception(
                    "to use `nearest_neighbours` you should not set `max_dist`"
                )
            neigh_idxs = kdt_s.query(xy_target, max_num_neighbours)[1]
            return neigh_idxs
        elif method == "within_radius":
            if max_dist is None:
                raise Exception(
                    "to use `witin_radius` method you shold set `max_dist`"
                )
            if max_num_neighbours is not None:
                raise Exception(
                    "to use `within_radius` method you should not set `max_num_neighbours`"
                )
            neigh_idxs = kdt_s.query_ball_point(
                xy_target, max_dist
            )
            return neigh_idxs
        else:
            raise NotImplementedError(method)

    G_connect = networkx.DiGraph()
    G_connect.add_nodes_from(sorted(G_source.nodes(data=True)))
    G_connect.add_nodes_from(sorted(G_target.nodes(data=True)))

    # add edges
    for target_node in target_nodes_list:
        neigh_idxs = _find_neighbour_node_idxs_in_source_mesh(target_node)
        for i in neigh_idxs:
            source_node = source_nodes_list[i]
            # add edge from source to target
            G_connect.add_edge(source_node, target_node)
            d = np.sqrt(
                np.sum((G_connect.nodes[source_node]["pos"] - G_connect.nodes[target_node]["pos"]) ** 2)
            )
            G_connect.edges[source_node, target_node]["len"] = d
            G_connect.edges[source_node, target_node]["vdiff"] = (
                G_connect.nodes[source_node]["pos"] - G_connect.nodes[target_node]["pos"]
            )

    return G_connect


def main():
    parser = ArgumentParser(description="Graph generation arguments")
    parser.add_argument(
        "--dataset",
        type=str,
        default="meps_example",
        help="Dataset to load grid point coordinates from "
        "(default: meps_example)",
    )
    parser.add_argument(
        "--graph",
        type=str,
        default="multiscale",
        help="Name to save graph as (default: multiscale)",
    )
    parser.add_argument(
        "--create-plots",
        default=False,
        action="store_true",
        help="Create graph plots during generation ",
    )
    parser.add_argument(
        "--levels",
        type=int,
        help="Limit multi-scale mesh to given number of levels, "
        "from bottom up (default: None (no limit))",
    )
    parser.add_argument(
        "--kind",
        type=str,
        default="flat",
        choices=["flat", "hierarchical"],
        help="kind of mesh graph to create (default: flat)",
    )
    args = parser.parse_args()

    # Load grid positions
    graph_dir_path = os.path.join("graphs", args.graph)
    os.makedirs(graph_dir_path, exist_ok=True)

    dataset = WeatherDataset(args.dataset)
    xy = dataset.grid_xy

    create_mesh_graphs(
        levels=args.levels,
        xy=xy,
        nx=3,
        create_plots=args.create_plots,
        hierarchical=args.kind == "hierarchical",
    )


if __name__ == "__main__":
    main()
