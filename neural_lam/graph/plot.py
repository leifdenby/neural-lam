# Third-party
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase
import numpy as np
import networkx
import torch_geometric as pyg
from torch_geometric.utils.convert import from_networkx

from . import networkx_utils as nx_utils


def nx_draw_with_pos(g, with_labels=False, **kwargs):
    pos = {node: g.nodes[node]["pos"] for node in g.nodes()}
    ax = kwargs.pop("ax", None)
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 10))
    networkx.draw_networkx(
        ax=ax, G=g, pos=pos, hide_ticks=False, with_labels=with_labels, **kwargs
    )
    
    return ax


def _get_graph_attr_values(g, attr_name, component="edges"):
    if component == "edges":
        features = list(g.edges(data=True))[0][2].keys()
    elif component == "nodes":
        features = list(g.nodes(data=True))[0][1].keys()
    else:
        raise ValueError(f"component {component} not in ['edges', 'nodes']")

    if not attr_name in features:
        raise ValueError(
            f"feature {attr_name} not in {component} features {features}"
        )

    if component == "edges":
        attr_vals = np.array([g.edges[edge][attr_name] for edge in g.edges()])
    elif component == "nodes":
        attr_vals = np.array([g.nodes[node][attr_name] for node in g.nodes()])

    attr_vals_for_plot = dict()

    if len(attr_vals.shape) > 1:
        raise NotImplementedError("Can't use multi-dimensional features for colors")
    elif np.issubdtype(attr_vals.dtype, np.str_):
        unique_strings = np.unique(attr_vals)
        val_str_map = {s: i for (i, s) in enumerate(unique_strings)}
        plot_values = np.array([val_str_map[s] for s in attr_vals])
        attr_vals_for_plot["values"] = plot_values
        attr_vals_for_plot["discrete_labels"] = val_str_map
    elif np.issubdtype(attr_vals.dtype, np.integer):
        unique_ints = np.unique(attr_vals)
        val_int_map = {val: i for (i, val) in enumerate(unique_ints)}
        plot_values = np.array([val_int_map[s] for s in attr_vals])
        attr_vals_for_plot["values"] = plot_values
        attr_vals_for_plot["discrete_labels"] = val_int_map
    elif np.issubdtype(attr_vals.dtype, np.floating):
        attr_vals_for_plot["values"] = attr_vals
    else:
        raise NotImplementedError(f"Array feature values of type {type(attr_vals[0])} not supported")

    return attr_vals_for_plot

def _create_graph_attr_legend(discrete_labels, cmap, attr_kind, attr_name, loc, norm):
    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=label,
            markerfacecolor=cmap(norm(val)),
        )
        for (label, val) in discrete_labels.items()
    ]
    legend = plt.legend(
        handles=legend_handles, title=f"{attr_kind} {attr_name}", loc=loc
    )
    return legend

def _create_graph_attr_colorbar(ax, cmap, norm, attr_name, attr_kind, loc):
    if loc == "upper left":
        ax_inset = ax.inset_axes([0.05, 0.94, 0.1, 0.02])
    elif loc == "upper right":
        ax_inset = ax.inset_axes([0.87, 0.94, 0.1, 0.02])
    else:
        raise ValueError(f"loc {loc} not in ['upper left', 'upper right']")
    
    # set the facecolor of the inset axes to white
    ax_inset.set_facecolor('white')
    
    cbar = ColorbarBase(ax=ax_inset, cmap=cmap, norm=norm, orientation="horizontal")

    ax_inset.set_title(f"{attr_kind} {attr_name}", fontsize=10)
    fig = ax.figure
    bbox = ax_inset.get_tightbbox(fig.canvas.get_renderer())
    x0, y0, width, height = bbox.transformed(fig.transFigure.inverted()).bounds
    # slightly increase the very tight bounds:
    xpad = 0.05 * width
    ypad = 0.05 * height
    fig.add_artist(plt.Rectangle((x0-xpad, y0-ypad), width+2*xpad, height+2*ypad, edgecolor='lightgrey', linewidth=1, fill=False))
    return cbar


def nx_draw_with_pos_and_attr(
    g,
    ax=None,
    with_labels=False,
    edge_color_attr=None,
    node_color_attr=None,
    node_zorder_attr=None,
    node_size=300,
    **kwargs
):
        
    if node_zorder_attr is not None:
        g = nx_utils.sort_nodes_internally(g, node_attribute=node_zorder_attr)

    if edge_color_attr is not None:
        edge_attr_vals = _get_graph_attr_values(
            g, edge_color_attr, component="edges"
        )

        if "cmap" not in kwargs:
            if "discrete_labels" in edge_attr_vals:
                kwargs["edge_cmap"] = plt.get_cmap("tab20")
            else:
                kwargs["edge_cmap"] = plt.get_cmap("viridis")
        kwargs["edge_color"] = edge_attr_vals["values"]
        kwargs["edge_vmin"] = min(edge_attr_vals["values"])
        kwargs["edge_vmax"] = max(edge_attr_vals["values"])

    if node_color_attr is not None:
        node_attr_vals = _get_graph_attr_values(
            g, node_color_attr, component="nodes"
        )
        if "cmap" not in kwargs:
            if "discrete_labels" in node_attr_vals:
                kwargs["cmap"] = plt.get_cmap("tab20")
            else:
                kwargs["cmap"] = plt.get_cmap("viridis")
        kwargs["node_color"] = node_attr_vals["values"]
        kwargs["vmin"] = min(node_attr_vals["values"])
        kwargs["vmax"] = max(node_attr_vals["values"])

    ax = nx_draw_with_pos(
        g,
        ax=ax,
        node_size=node_size,
        arrows=True,
        with_labels=False,
        connectionstyle="arc3, rad=0.1",
        **kwargs,
    )

    legends = []
    
    if node_color_attr is not None:
        norm = Normalize(vmin=kwargs["vmin"], vmax=kwargs["vmax"])
        if "discrete_labels" in node_attr_vals:
            legend = _create_graph_attr_legend(
                discrete_labels=node_attr_vals["discrete_labels"],
                cmap=kwargs["cmap"],
                attr_kind="node",
                attr_name=node_color_attr,
                loc="upper left",
                norm=norm
            )
            legends.append(legend)
        else:
            _create_graph_attr_colorbar(ax=ax, cmap=kwargs["cmap"], norm=norm, attr_name=node_color_attr, loc="upper left", attr_kind="node")
            
    if edge_color_attr is not None:
        norm = Normalize(vmin=kwargs["edge_vmin"], vmax=kwargs["edge_vmax"])
        if "discrete_labels" in edge_attr_vals:
            legend = _create_graph_attr_legend(
                discrete_labels=edge_attr_vals["discrete_labels"],
                cmap=kwargs["edge_cmap"],
                attr_kind="edge",
                attr_name=edge_color_attr,
                loc="upper right",
                norm=norm
            )
            legends.append(legend)
        else:
            _create_graph_attr_colorbar(ax=ax, cmap=kwargs["edge_cmap"], norm=norm, attr_name=edge_color_attr, loc="upper right", attr_kind="edge")


    for legend in legends:
        ax.add_artist(legend)

    return ax


def plot_pyg_graph(graph, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8), dpi=200)  # W,H
    edge_index = graph.edge_index
    pos = graph.pos

    # Fix for re-indexed edge indices only containing mesh nodes at
    # higher levels in hierarchy
    edge_index = edge_index - edge_index.min()

    if pyg.utils.is_undirected(edge_index):
        # Keep only 1 direction of edge_index
        edge_index = edge_index[:, edge_index[0] < edge_index[1]]  # (2, M/2)
    # TODO: indicate direction of directed edges

    # Move all to cpu and numpy, compute (in)-degrees
    degrees = (
        pyg.utils.degree(edge_index[1], num_nodes=pos.shape[0]).cpu().numpy()
    )

    edge_index = edge_index.cpu().numpy()
    pos = pos.cpu().numpy()

    # Plot edges
    from_pos = pos[edge_index[0]]  # (M/2, 2)
    to_pos = pos[edge_index[1]]  # (M/2, 2)
    edge_lines = np.stack((from_pos, to_pos), axis=1)
    ax.add_collection(
        matplotlib.collections.LineCollection(
            edge_lines, lw=0.4, colors="black", zorder=1
        )
    )

    # Plot nodes
    node_scatter = ax.scatter(
        pos[:, 0],
        pos[:, 1],
        c=degrees,
        s=3,
        marker="o",
        zorder=2,
        cmap="viridis",
        clim=None,
    )

    plt.colorbar(node_scatter, aspect=50)

    if title is not None:
        ax.set_title(title)

    return ax
