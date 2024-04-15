# Standard library
import os
from argparse import ArgumentParser

# Third-party
import matplotlib.pyplot as plt
import networkx
import numpy as np
import scipy.spatial
import torch
from torch_geometric.utils.convert import from_networkx as pyg_from_networkx
from loguru import logger

from ..datasets.meps.weather_dataset import WeatherDataset
from .plot import plot_graph
from .save import save_edges, save_edges_list
from .kinds.flat import create_flat_mesh_graph
from .networkx_utils import prepend_node_index


def create_single_level_2d_mesh_graph(xy, nx, ny):
    """
    Create directed graph with nx * ny nodes representing a 2D grid with
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


def _create_2d_mesh_graphs(levels, xy, nx=3):
    """
    Create 2D grid mesh graphs for each level in the multi-scale mesh graph

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
    G_all_levels : list of networkx.Graph
        List of networkx graphs for each level representing the connectivity
        of the mesh within each level
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
        g = create_single_level_2d_mesh_graph(xy, n, n)
        for node in g.nodes:
            g.nodes[node]["level"] = lev
        G_all_levels.append(g)

    return G_all_levels

def create_grid_graph_nodes(xy, level_id=1000):
    """
    Create a networkx.Graph comprising only nodes for each (x,y)-point in the `xy` coordinate
    array (the attribute `pos` giving the (x,y)-coordinate value) and with
    node label (level_id, i, j)
    """
    # grid nodes
    Ny, Nx = xy.shape[1:]

    G_grid = networkx.grid_2d_graph(Ny, Nx)
    G_grid.clear_edges()

    # vg features (only pos introduced here)
    for node in G_grid.nodes:
        # pos is in feature but here explicit for convenience
        G_grid.nodes[node]["pos"] = np.array([xy[0][node], xy[1][node]])
        G_grid.nodes[node]["level"] = level_id

    # add `level_id` (default to 1000) to node key to separate grid nodes (1000,i,j) from mesh nodes
    # (i,j) and impose sorting order such that vm are the first nodes
    G_grid = prepend_node_index(G_grid, level_id)

    return G_grid


def _create_grid2mesh_graph(G_mesh, G_grid, xy, max_dist):
    #
    # Grid2Mesh
    #

    # mesh nodes to connect to
    vm = G_mesh.nodes

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

    pyg_g2m = pyg_from_networkx(G_g2m)

    return pyg_g2m


def create_cross_level_graph(G_source, G_target, method="nearest_neighbour", max_dist=None, max_num_neighbours=None):
    """
    Create a new graph containing the nodes in `G_source` and `G_target` and add
    directed edges from nodes in `G_source` to nodes in `G_target` based on the
    method specified.

    Parameters
    ----------
    G_source : networkx.Graph
        Source graph, edge connections are made from nodes in this graph
    G_target : networkx.Graph
        Target graph, edge connections are made to nodes in this graph
    method : str
        Method to use for finding neighbours in `G_target` for each node in `G_source`.
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

    # build kd tree for grid point pos
    xy_target = np.array([G_target.nodes[node]["pos"] for node in G_target.nodes])
    kdt_g = scipy.spatial.KDTree(xy_target)

    def _find_neighbour_node_idxs_in_target_mesh(from_node):
        xy_source = G_source.nodes[from_node]["pos"]

        if method == "nearest_neighbour":
            neigh_idx = kdt_g.query(xy_source, 1)[1]
            if max_dist is not None or max_num_neighbours is not None:
                raise Exception("to use `nearest_neighbour` you should not set `max_dist` or `max_num_neighbours`")
            return [neigh_idx]
        elif method == "nearest_neighbours":
            if max_num_neighbours is None:
                raise Exception("to use `nearest_neighbours` you should set the max number with `max_num_neighbours`")
            if max_dist is not None:
                raise Exception("to use `nearest_neighbours` you should not set `max_dist`")
            neigh_idxs = kdt_g.query(xy_source, max_num_neighbours)[1]
            return neigh_idxs
        elif method == "within_radius":
            if max_dist is None:
                raise Exception("to use `witin_radius` method you shold set `max_dist`")
            if max_num_neighbours is not None:
                raise Exception("to use `within_radius` method you should not set `max_num_neighbours`")
            neigh_target_nodes_idxs = kdt_g.query_ball_point(xy_source, max_dist)
            return neigh_target_nodes_idxs
        else:
            raise NotImplementedError(method)

    # Re-create graph with sorted node indices
    # Need to do sorting of nodes this way for indices to map correctly to pyg
    G_g2m = networkx.DiGraph()
    G_g2m.add_nodes_from(sorted(G_source.nodes(data=True)))
    G_g2m.add_nodes_from(sorted(G_target.nodes(data=True)))

    # add edges
    for v in source_nodes_list:
        neigh_idxs = _find_neighbour_node_idxs_in_target_mesh(v)
        for i in neigh_idxs:
            u = target_nodes_list[i]
            # add edge from grid to mesh
            G_g2m.add_edge(u, v)
            d = np.sqrt(
                np.sum((G_g2m.nodes[u]["pos"] - G_g2m.nodes[v]["pos"]) ** 2)
            )
            G_g2m.edges[u, v]["len"] = d
            G_g2m.edges[u, v]["vdiff"] = (
                G_g2m.nodes[u]["pos"] - G_g2m.nodes[v]["pos"]
            )
            
    return G_g2m


def _create_mesh2grid_graph(G_g2m, vm, vm_xy, vg_list, plot):
    """
    Create mesh-to-grid graph

    Parameters
    ----------
    G_g2m : networkx.DiGraph
        Graph with edges from grid to mesh
    vm_xy : np.ndarray
        Mesh node coordinates
    vg_list : list
        List of grid nodes
    """

    # start out from Grid2Mesh and then replace edges
    G_m2g = G_g2m.copy()
    G_m2g.clear_edges()

    # mesh nodes on lowest level
    vm = G_bottom_mesh.nodes
    vm_xy = np.array([xy for _, xy in vm.data("pos")])

    # build kd tree for grid point pos
    # order in vg_list should be same as in vg_xy
    vg_list = list(G_grid.nodes)
    vg_xy = np.array([[xy[0][node[1:]], xy[1][node[1:]]] for node in vg_list])
    kdt_g = scipy.spatial.KDTree(vg_xy)

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
    pyg_m2g = pyg_from_networkx(G_m2g_int)

    return pyg_m2g

def create_mesh_graphs(
    levels: int,
    xy: np.ndarray,
    kind: str,
    nx: int = 3,
    create_plots: bool = False,
):
    # Create mesh graphs for all levels
    G_all_levels = _create_2d_mesh_graphs(
        levels=levels,
        xy=xy,
        nx=nx
    )

    if create_plots:
        for lev, g in enumerate(G_all_levels):
            s = 8.0
            Nx, Ny = xy.shape[1:]
            r = Nx / Ny
            fig, ax = plt.subplots(figsize=(s * r + 2.0, s), dpi=200)
            ax.scatter(xy[0], xy[1], color="r", marker=".", s=0.5, alpha=0.5)
            plot_graph(pyg_from_networkx(g), ax=ax, title=f"Mesh graph, level {lev}")
            fig.savefig(f"mesh_{lev}.png")
            plt.close(fig)

    # create m2m graphs

    if kind == "hierarchical":
        m2m_graphs, G_bottom_mesh, all_mesh_nodes = _create_hierarcical_mesh(
            G_all_levels=G_all_levels, mesh_levels=mesh_levels, plot=plot
        )
    elif kind == "flat":
        m2m_graphs, G_bottom_mesh, all_mesh_nodes = create_flat_mesh_graph(
            G_all_levels=G_all_levels, nx=nx
        )
    else:
        raise NotImplementedError(f"Kind {kind} not implemented")

    # radius within which grid nodes are associated with a mesh node
    # (in terms of mesh distance)
    DM_SCALE = 0.67

    # distance between mesh nodes, OBS: assumes isotropic grid
    dm = np.sqrt(
        np.sum((vm.data("pos")[(0, 1, 0)] - vm.data("pos")[(0, 0, 0)]) ** 2)
    )
    
    max_dist = 0.42424242  # TODO
    


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
        help="Create graph plots during generation "
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
