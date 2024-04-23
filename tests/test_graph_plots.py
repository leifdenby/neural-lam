import numpy as np
import matplotlib.pyplot as plt
import pytest
from loguru import logger
import tempfile

import neural_lam.graph.create.base as graph_creation
import neural_lam.graph.plot as graph_plots

def _create_fake_xy(N=10):
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    xy = np.meshgrid(x, y)
    xy = np.stack(xy, axis=0)
    return xy


def test_plot():
    xy = _create_fake_xy(10)

    graph_components = graph_creation.create_all_graph_components(
        m2m_connectivity="flat_multiscale",
        xy=xy,
        merge_components=False,
        m2m_connectivity_kwargs=dict(
            max_num_levels=3,
            refinement_factor=2,
        ),
    )
    def _is_ndarray(val):
        return isinstance(val, np.ndarray)
    
    def _is_valid_color_attr(val):
        return isinstance(val, (int, float, str))
    
    fig, ax = plt.subplots()
    for graph in graph_components.values():
        node_attrs = list(list(graph.nodes(data=True))[0][1].keys())
        edge_attrs = list(list(graph.edges(data=True))[0][2].keys())
        
        for edge_attr in edge_attrs + []:
            for node_attr in node_attrs + []:
                should_raise = None
                if not _is_valid_color_attr(list(graph.edges(data=True))[0][2][edge_attr]):
                    should_raise = NotImplementedError
                elif not _is_valid_color_attr(list(graph.nodes(data=True))[0][1][node_attr]):
                    should_raise = NotImplementedError

                def fn():
                    graph_plots.nx_draw_with_pos_and_attr(
                        graph, ax=ax, edge_color_attr=edge_attr, node_color_attr=node_attr
                    )
                    
                if should_raise is not None:
                    with pytest.raises(should_raise):
                        fn()
                else:
                    fn()

    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        fig.savefig(f.name)