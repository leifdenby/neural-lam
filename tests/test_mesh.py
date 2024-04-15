import neural_lam.mesh.create as create_mesh
import neural_lam.mesh.plot as plot_mesh

from loguru import logger
import numpy as np
from torch_geometric.utils.convert import from_networkx as pyg_from_networkx


def _create_fake_xy(N=10):
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    xy = np.meshgrid(x, y)
    xy = np.stack(xy, axis=0)
    return xy


def test_create_mesh():
    xy = _create_fake_xy(N=4)
    mesh_graph = create_mesh.create_single_level_2d_mesh_graph(xy=xy, nx=5, ny=5)

    # use networkx to make a plot of the graph and save it as a png
    import networkx as nx
    import matplotlib.pyplot as plt

    pos = {node: mesh_graph.nodes[node]["pos"] for node in mesh_graph.nodes()}
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    nx.draw_networkx(
        ax=ax, G=mesh_graph, pos=pos, with_labels=True, hide_ticks=False
    )

    ax.scatter(xy[0, ...], xy[1, ...], color="r")
    ax.axison = True

    fig.savefig("test_mesh.png")

    lev = 0
    s = 8.0
    Nx, Ny = xy.shape[1:]
    r = Nx / Ny
    fig, ax = plt.subplots(figsize=(s * r + 2.0, s), dpi=200)
    ax.scatter(xy[0], xy[1], color="r", marker=".", s=0.5, alpha=0.5)
    g = mesh_graph
    plot_mesh.plot_graph(pyg_from_networkx(g), ax=ax, title=f"Mesh graph, level {lev}")
    fig.savefig(f"mesh_{lev}.png")


def test_full():
    xy = _create_fake_xy(N=64)
    create_mesh.create_mesh_graphs(
        xy=xy, levels=4, create_plots=True, kind="flat"
    )