# Standard library
import os
from argparse import ArgumentParser

# Third-party
import matplotlib.pyplot as plt
import networkx
import numpy as np
import scipy.spatial
import torch
from torch_geometric.utils.convert import from_networkx
from loguru import logger

from ..datasets.meps.weather_dataset import WeatherDataset
from .plot import plot_graph
from .save import save_edges, save_edges_list


def sort_nodes_internally(nx_graph):
    # For some reason the networkx .nodes() return list can not be sorted,
    # but this is the ordering used by pyg when converting.
    # This function fixes this.
    H = networkx.DiGraph()
    H.add_nodes_from(sorted(nx_graph.nodes(data=True)))
    H.add_edges_from(nx_graph.edges(data=True))
    return H


def from_networkx_with_start_index(nx_graph, start_index):
    pyg_graph = from_networkx(nx_graph)
    pyg_graph.edge_index += start_index
    return pyg_graph


def mk_2d_graph(xy, nx, ny):
    """
    Create graph with nx * ny nodes representing a 2D grid with
    positions spanning the range of xy coordinate values (first dimension
    is assumed to be x and y coordinate values respectively). Each nodes is
    connected to its eight nearest neighbours, both horizontally, vertically
    and diagonally as directed edges (which means that the graph contains two
    edges between each pair of connected nodes).

    The nodes contain a "pos" attribute with the x and y
    coordinates of the node.

    The edges contain a "len" attribute with the length of the edge
    and a "vdiff" attribute with the vector difference between the
    nodes.

    Parameters
    ----------
    xy : np.ndarray [2, N, M]
        Grid point coordinates, with first dimension representing
        x and y coordinates respectively. N and M are the number
        of grid points in the x and y direction respectively
    nx : int
        Number of nodes in x direction
    ny : int
        Number of nodes in y direction

    Returns
    -------
    networkx.DiGraph
        Graph representing the 2D grid
    """
    xm, xM = np.amin(xy[0][0, :]), np.amax(xy[0][0, :])
    ym, yM = np.amin(xy[1][:, 0]), np.amax(xy[1][:, 0])

    # avoid nodes on border
    dx = (xM - xm) / nx
    dy = (yM - ym) / ny
    lx = np.linspace(xm + dx / 2, xM - dx / 2, nx)
    ly = np.linspace(ym + dy / 2, yM - dy / 2, ny)

    mg = np.meshgrid(lx, ly)
    g = networkx.grid_2d_graph(len(ly), len(lx))

    for node in g.nodes:
        g.nodes[node]["pos"] = np.array([mg[0][node], mg[1][node]])

    # add diagonal edges
    g.add_edges_from(
        [((x, y), (x + 1, y + 1)) for x in range(nx - 1) for y in range(ny - 1)]
        + [
            ((x + 1, y), (x, y + 1))
            for x in range(nx - 1)
            for y in range(ny - 1)
        ]
    )

    # turn into directed graph
    dg = networkx.DiGraph(g)
    for u, v in g.edges():
        d = np.sqrt(np.sum((g.nodes[u]["pos"] - g.nodes[v]["pos"]) ** 2))
        dg.edges[u, v]["len"] = d
        dg.edges[u, v]["vdiff"] = g.nodes[u]["pos"] - g.nodes[v]["pos"]
        dg.add_edge(v, u)
        dg.edges[v, u]["len"] = d
        dg.edges[v, u]["vdiff"] = g.nodes[v]["pos"] - g.nodes[u]["pos"]

    return dg


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


def _create_flat_mesh(G_all_levels, nx):
    """
    Create flat mesh graph by combining all levels

    Parameters
    ----------
    G_all_levels : list
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
    pyg_m2m = from_networkx(G_int)
    m2m_graphs = [pyg_m2m]
    return m2m_graphs, G_bottom_mesh, all_mesh_nodes


def _create_mesh(levels, xy, nx=3):
    """
    Create mesh graph

    Parameters
    ----------
    levels : int
        Number of levels in mesh graph
    xy : np.ndarray
        Grid point coordinates
    nx : int
        Connectivity of mesh graph, number of children will be nx**2
        per parent node

    Returns
    -------
    None
    """
    nlev = int(np.log(max(xy.shape)) / np.log(nx))
    nleaf = nx**nlev  # leaves at the bottom = nleaf**2

    mesh_levels = nlev - 1
    if levels:
        # Limit the levels in mesh graph
        mesh_levels = min(mesh_levels, levels)

    logger.info(f"nlev: {nlev}, nleaf: {nleaf}, mesh_levels: {mesh_levels}")

    # multi resolution tree levels
    G_all_levels = []
    for lev in range(1, mesh_levels + 1):
        n = int(nleaf / (nx**lev))
        g = mk_2d_graph(xy, n, n)

        G_all_levels.append(g)

    return G_all_levels


def _create_m2m_graphs(G_all_levels, nx):
    if hierarchical:
        m2m_graphs, G_bottom_mesh, all_mesh_nodes = _create_hierarcical_mesh(
            G_all_levels=G_all_levels, mesh_levels=mesh_levels, plot=plot
        )
    else:
        m2m_graphs, G_bottom_mesh, all_mesh_nodes = _create_flat_mesh(
            G_all_levels=G_all_levels, nx=nx
        )


def prepend_node_index(graph, new_index):
    """
    Prepend node index to node tuple in graph, i.e. (i, j) -> (new_index, i, j)

    Parameters
    ----------
    graph : networkx.Graph
        Graph to relabel
    new_index : int
        New index to prepend to node tuple

    Returns
    -------
    networkx.Graph
        Graph with relabeled nodes
    """
    ijk = [tuple((new_index,) + x) for x in graph.nodes]
    to_mapping = dict(zip(graph.nodes, ijk))
    return networkx.relabel_nodes(graph, to_mapping, copy=True)


def _create_grid2mesh_graph(plot, G_bottom_mesh, all_mesh_nodes, xy):
    #
    # Grid2Mesh
    #

    # radius within which grid nodes are associated with a mesh node
    # (in terms of mesh distance)
    DM_SCALE = 0.67

    # mesh nodes on lowest level
    vm = G_bottom_mesh.nodes
    vm_xy = np.array([xy for _, xy in vm.data("pos")])
    # distance between mesh nodes
    dm = np.sqrt(
        np.sum((vm.data("pos")[(0, 1, 0)] - vm.data("pos")[(0, 0, 0)]) ** 2)
    )

    # grid nodes
    Ny, Nx = xy.shape[1:]

    G_grid = networkx.grid_2d_graph(Ny, Nx)
    G_grid.clear_edges()

    # vg features (only pos introduced here)
    for node in G_grid.nodes:
        # pos is in feature but here explicit for convenience
        G_grid.nodes[node]["pos"] = np.array([xy[0][node], xy[1][node]])

    # add 1000 to node key to separate grid nodes (1000,i,j) from mesh nodes
    # (i,j) and impose sorting order such that vm are the first nodes
    G_grid = prepend_node_index(G_grid, 1000)

    # build kd tree for grid point pos
    # order in vg_list should be same as in vg_xy
    vg_list = list(G_grid.nodes)
    vg_xy = np.array([[xy[0][node[1:]], xy[1][node[1:]]] for node in vg_list])
    kdt_g = scipy.spatial.KDTree(vg_xy)

    # now add (all) mesh nodes, include features (pos)
    G_grid.add_nodes_from(all_mesh_nodes)

    # Re-create graph with sorted node indices
    # Need to do sorting of nodes this way for indices to map correctly to pyg
    G_g2m = networkx.Graph()
    G_g2m.add_nodes_from(sorted(G_grid.nodes(data=True)))

    # turn into directed graph
    G_g2m = networkx.DiGraph(G_g2m)

    # add edges
    for v in vm:
        # find neighbours (index to vg_xy)
        neigh_idxs = kdt_g.query_ball_point(vm[v]["pos"], dm * DM_SCALE)
        for i in neigh_idxs:
            u = vg_list[i]
            # add edge from grid to mesh
            G_g2m.add_edge(u, v)
            d = np.sqrt(
                np.sum((G_g2m.nodes[u]["pos"] - G_g2m.nodes[v]["pos"]) ** 2)
            )
            G_g2m.edges[u, v]["len"] = d
            G_g2m.edges[u, v]["vdiff"] = (
                G_g2m.nodes[u]["pos"] - G_g2m.nodes[v]["pos"]
            )

    pyg_g2m = from_networkx(G_g2m)

    return pyg_g2m


def _create_mesh2grid_graph(G_g2m, vm, vm_xy, vg_list, plot):
    """
    Create mesh-to-grid graph

    Parameters
    ----------
    G_g2m : networkx.DiGraph
        Graph with edges from grid to mesh
    vm : networkx.NodeView
        Mesh nodes
    vm_xy : np.ndarray
        Mesh node coordinates
    vg_list : list
        List of grid nodes
    """

    # start out from Grid2Mesh and then replace edges
    G_m2g = G_g2m.copy()
    G_m2g.clear_edges()

    # build kd tree for mesh point pos
    # order in vm should be same as in vm_xy
    vm_list = list(vm)
    kdt_m = scipy.spatial.KDTree(vm_xy)

    # add edges from mesh to grid
    for v in vg_list:
        # find 4 nearest neighbours (index to vm_xy)
        neigh_idxs = kdt_m.query(G_m2g.nodes[v]["pos"], 4)[1]
        for i in neigh_idxs:
            u = vm_list[i]
            # add edge from mesh to grid
            G_m2g.add_edge(u, v)
            d = np.sqrt(
                np.sum((G_m2g.nodes[u]["pos"] - G_m2g.nodes[v]["pos"]) ** 2)
            )
            G_m2g.edges[u, v]["len"] = d
            G_m2g.edges[u, v]["vdiff"] = (
                G_m2g.nodes[u]["pos"] - G_m2g.nodes[v]["pos"]
            )

    # relabel nodes to integers (sorted)
    G_m2g_int = networkx.convert_node_labels_to_integers(
        G_m2g, first_label=0, ordering="sorted"
    )
    pyg_m2g = from_networkx(G_m2g_int)

    return pyg_m2g


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
        "--plot",
        type=int,
        default=0,
        help="If graphs should be plotted during generation "
        "(default: 0 (false))",
    )
    parser.add_argument(
        "--levels",
        type=int,
        help="Limit multi-scale mesh to given number of levels, "
        "from bottom up (default: None (no limit))",
    )
    parser.add_argument(
        "--hierarchical",
        type=int,
        default=0,
        help="Generate hierarchical mesh graph (default: 0, no)",
    )
    args = parser.parse_args()

    # Load grid positions
    graph_dir_path = os.path.join("graphs", args.graph)
    os.makedirs(graph_dir_path, exist_ok=True)

    dataset = WeatherDataset(args.dataset)
    xy = dataset.grid_xy

    # Create mesh graph
    _create_mesh(
        levels=args.levels,
        plot=args.plot,
        hierarchical=args.hierarchical,
        xy=xy,
    )
    if plot:
        fig, _ = plot_graph(from_networkx(g), title=f"Mesh graph, level {lev}")
        fig.savefig(f"mesh_{lev}.png")
        plt.show()

    # Create grid-to-mesh graph
    pyg_g2m = _create_grid2mesh_graph(
        plot=args.plot,
        G_bottom_mesh=G_bottom_mesh,
        all_mesh_nodes=all_mesh_nodes,
        xy=xy,
    )
    # Create mesh-to-grid graph
    pyg_m2g = _create_mesh2grid_graph(
        G_g2m=G_g2m, vm=vm, vm_xy=vm_xy, vg_list=vg_list
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

    import ipdb

    ipdb.set_trace()

    return m2m_graphs, G_bottom_mesh, all_mesh_nodes

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


if __name__ == "__main__":
    main()
