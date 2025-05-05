# Standard library
import os
from argparse import ArgumentParser
from pathlib import Path

# Third-party
import numpy as np
import torch
import weather_model_graphs as wmg
from loguru import logger
from torch_geometric.utils.convert import from_networkx

# Local
from .config import load_config_and_datastore
from .datastore.base import BaseRegularGridDatastore


def create_graph(
    graph_dir_path: str,
    xy: np.ndarray,
    n_max_levels: int,
    hierarchical: bool,
):
    """
    Create graph components from `xy` grid coordinates and store in
    `graph_dir_path`.

    Parameters
    ----------
    graph_dir_path : str
        Path to store the graph components.
    xy : np.ndarray
        Grid coordinates, expected to be of shape (Nx, Ny, 2).
    n_max_levels : int
        Limit multi-scale mesh to given number of levels, from bottom up
        (default: None (no limit)).
    hierarchical : bool
        Generate hierarchical mesh graph (default: False).
    create_plot : bool
        If graphs should be plotted during generation (default: False).

    Returns
    -------
    None

    """
    if hierarchical:
        graph_kind = "oskarsson_hierarchical"
    else:
        if n_max_levels is None or n_max_levels == 0:
            graph_kind = "keisler"
        else:
            graph_kind = "graphcast"

    logger.info(f"Creating {graph_kind} graph")

    graph_fn = getattr(wmg.create.archetype, f"create_{graph_kind}_graph")
    # Third-party
    import ipdb

    ipdb.set_trace()
    nx_graph = graph_fn(coords=xy)

    split_nodes_by = wmg.split.DEFAULT_NODE_SPLITS[graph_kind]
    split_edges_by = wmg.split.DEFAULT_EDGE_SPLITS[graph_kind]
    dt_graph = wmg.save.graph_to_datatree(
        graph=nx_graph,
        split_nodes_by=split_nodes_by,
        split_edges_by=split_edges_by,
    )

    fp_graph = Path(graph_dir_path) / "{graph_kind}.nc"
    logger.info(f"Saving graph to {fp_graph}")
    dt_graph.to_netcdf(fp_graph, overwrite=True)


def create_graph_from_datastore(
    datastore: BaseRegularGridDatastore,
    output_root_path: str,
    n_max_levels: int = None,
    hierarchical: bool = False,
):
    xy = datastore.get_xy(category="state", stacked=True)

    create_graph(
        graph_dir_path=output_root_path,
        xy=xy,
        n_max_levels=n_max_levels,
        hierarchical=hierarchical,
    )


def cli(input_args=None):
    parser = ArgumentParser(description="Graph generation arguments")
    parser.add_argument(
        "--config_path",
        type=str,
        help="Path to neural-lam configuration file",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="multiscale",
        help="Name to save graph as (default: multiscale)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="If graphs should be plotted during generation "
        "(default: False)",
    )
    parser.add_argument(
        "--levels",
        type=int,
        help="Limit multi-scale mesh to given number of levels, "
        "from bottom up (default: None (no limit))",
    )
    parser.add_argument(
        "--hierarchical",
        action="store_true",
        help="Generate hierarchical mesh graph (default: False)",
    )
    args = parser.parse_args(input_args)

    assert (
        args.config_path is not None
    ), "Specify your config with --config_path"

    # Load neural-lam configuration and datastore to use
    _, datastore = load_config_and_datastore(config_path=args.config_path)

    create_graph_from_datastore(
        datastore=datastore,
        output_root_path=os.path.join(datastore.root_path, "graph", args.name),
        n_max_levels=args.levels,
        hierarchical=args.hierarchical,
    )


if __name__ == "__main__":
    cli()
