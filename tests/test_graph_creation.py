import neural_lam.graph.create as graph_creation
import neural_lam.graph.plot as graph_plotting

from loguru import logger
import numpy as np
from torch_geometric.utils.convert import from_networkx as pyg_from_networkx
import pytest


def _create_fake_xy(N=10):
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    xy = np.meshgrid(x, y)
    xy = np.stack(xy, axis=0)
    return xy


def test_create_single_level_mesh_graph():
    xy = _create_fake_xy(N=4)
    mesh_graph = graph_creation.create_single_level_2d_mesh_graph(xy=xy, nx=5, ny=5)

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
    graph_plotting.plot_graph(pyg_from_networkx(g), ax=ax, title=f"Mesh graph, level {lev}")
    fig.savefig(f"mesh_{lev}.png")


@pytest.mark.parametrize("kind", ["graphcast", "keissler", "oscarsson_hierarchical"])
@pytest.mark.parametrize("merge_components", [True, False])
def test_create_graph_architype(merge_components, kind):
    xy = _create_fake_xy(N=64)
    fn_name = f"create_{kind}_graph"
    fn = getattr(graph_creation.architypes, fn_name)

    fn(xy_grid=xy, merge_components=merge_components)
