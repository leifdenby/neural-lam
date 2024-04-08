import neural_lam.mesh.create as create_mesh
import neural_lam.mesh.plot as plot_mesh

from loguru import logger
import numpy as np


def _create_fake_xy():
    x = np.linspace(0, 1, 10)
    y = np.linspace(0, 1, 10)
    xy = np.meshgrid(x, y)
    xy = np.stack(xy, axis=0)
    return xy


def test_create_mesh():
    xy = _create_fake_xy()
    mesh_graph = create_mesh.mk_2d_graph(xy=xy, nx=10, ny=10)

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
