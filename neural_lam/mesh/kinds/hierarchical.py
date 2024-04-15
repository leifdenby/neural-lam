import scipy
import numpy as np
import networkx
from loguru import logger
from torch_geometric.utils.convert import from_networkx as pyg_from_networkx


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

def _create_hierarcical_mesh(G_all_levels, plot):
    mesh_levels = len(G_all_levels)
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
    for from_level, to_level, G_from, G_to, start_index in zip(
        range(1, mesh_levels),
        range(0, mesh_levels - 1),
        G_all_levels[1:],
        G_all_levels[:-1],
        first_index_level[: mesh_levels - 1],
    ):
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

        # relabel nodes to integers (sorted)
        G_down_int = networkx.convert_node_labels_to_integers(
            G_down, first_label=start_index, ordering="sorted"
        )  # Issue with sorting here
        G_down_int = sort_nodes_internally(G_down_int)
        pyg_down = from_networkx_with_start_index(G_down_int, start_index)

        # Create up graph, invert downwards edges
        up_edges = torch.stack(
            (pyg_down.edge_index[1], pyg_down.edge_index[0]), dim=0
        )
        pyg_up = pyg_down.clone()
        pyg_up.edge_index = up_edges

        up_graphs.append(pyg_up)
        down_graphs.append(pyg_down)

        if plot:
            plot_graph(
                pyg_down, title=f"Down graph, {from_level} -> {to_level}"
            )
            plt.show()

            plot_graph(pyg_down, title=f"Up graph, {to_level} -> {from_level}")
            plt.show()

    # Save up and down edges
    # save_edges_list(up_graphs, "mesh_up", graph_dir_path)
    # save_edges_list(down_graphs, "mesh_down", graph_dir_path)
    logger.warning("not saving")

    # Extract intra-level edges for m2m
    m2m_graphs = [
        from_networkx_with_start_index(
            networkx.convert_node_labels_to_integers(
                level_graph, first_label=start_index, ordering="sorted"
            ),
            start_index,
        )
        for level_graph, start_index in zip(G_all_levels, first_index_level)
    ]

    # For use in g2m and m2g
    G_bottom_mesh = G_all_levels[0]

    joint_mesh_graph = networkx.union_all([graph for graph in G_all_levels])
    all_mesh_nodes = joint_mesh_graph.nodes(data=True)

    return m2m_graphs, G_bottom_mesh, all_mesh_nodes