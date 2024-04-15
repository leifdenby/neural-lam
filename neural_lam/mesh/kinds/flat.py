import networkx
import numpy as np
from torch_geometric.utils.convert import from_networkx as pyg_from_networkx

from ..networkx_utils import prepend_node_index

def create_flat_mesh_graph(G_all_levels: list[networkx.Graph], nx: int):
    """
    Create flat mesh graph by merging the single-level mesh
    graphs across all levels in `G_all_levels`.

    Parameters
    ----------
    G_all_levels : list of networkx.Graph
        List of networkx graphs for each level representing the connectivity
        of the mesh on each level
    nx : int
        Connectivity of mesh graph, number of children will be nx**2
        per parent node

    Returns
    -------
    m2m_graphs : list
        List of PyTorch geometric graphs for each level
    G_bottom_mesh : networkx.Graph
        Graph representing the bottom mesh level
    all_mesh_nodes : networkx.NodeView
        All mesh nodes
    """
    # combine all levels to one graph
    G_tot = G_all_levels[0]
    for lev in range(1, len(G_all_levels)):
        nodes = list(G_all_levels[lev - 1].nodes)
        n = int(np.sqrt(len(nodes)))
        ij = (
            np.array(nodes)
            .reshape((n, n, 2))[1::nx, 1::nx, :]
            .reshape(int(n / nx) ** 2, 2)
        )
        ij = [tuple(x) for x in ij]
        G_all_levels[lev] = networkx.relabel_nodes(
            G_all_levels[lev], dict(zip(G_all_levels[lev].nodes, ij))
        )
        G_tot = networkx.compose(G_tot, G_all_levels[lev])

    # Relabel mesh nodes to start with 0
    G_tot = prepend_node_index(G_tot, 0)

    # relabel nodes to integers (sorted)
    G_int = networkx.convert_node_labels_to_integers(
        G_tot, first_label=0, ordering="sorted"
    )

    # Graph to use in g2m and m2g
    G_bottom_mesh = G_tot
    all_mesh_nodes = G_tot.nodes(data=True)

    # export the nx graph to PyTorch geometric
    pyg_m2m = pyg_from_networkx(G_int)
    m2m_graphs = [pyg_m2m]
    return m2m_graphs, G_bottom_mesh, all_mesh_nodes