# Standard library
from dataclasses import dataclass, fields
from pathlib import Path
from typing import List, Sequence

# Third-party
import torch

# Local
from .utils import BufferList


def _as_list(value):
    if isinstance(value, (list, tuple, BufferList)):
        return list(value)
    return [value]


@dataclass(frozen=True)
class GraphEdgesAndFeatures:
    """
    Container for graph edge indices and static features.
    """

    hierarchical: bool

    g2m_edge_index: torch.Tensor
    m2g_edge_index: torch.Tensor
    m2m_edge_index: torch.Tensor | BufferList
    mesh_up_edge_index: Sequence[torch.Tensor] | BufferList
    mesh_down_edge_index: Sequence[torch.Tensor] | BufferList

    g2m_features: torch.Tensor
    m2g_features: torch.Tensor
    m2m_features: torch.Tensor | BufferList
    mesh_up_features: Sequence[torch.Tensor] | BufferList
    mesh_down_features: Sequence[torch.Tensor] | BufferList
    mesh_static_features: torch.Tensor | BufferList

    def as_dict(self):
        return {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if field.name != "hierarchical"
        }

    def sizes(self) -> "GraphSizes":
        return GraphSizes.from_graph(self)


@dataclass(frozen=True)
class GraphSizes:
    """
    Sizes derived from graph edges and features, used when constructing MLPs.
    """

    hierarchical: bool
    num_levels: int
    mesh_level_sizes: List[int]
    num_mesh_nodes: int
    num_mesh_nodes_ignore: int

    g2m_edges: int
    m2g_edges: int
    m2m_edge_counts: List[int]
    mesh_up_edge_counts: List[int]
    mesh_down_edge_counts: List[int]
    edge_split_sections: List[int]

    g2m_dim: int
    m2g_dim: int
    m2m_dim: int
    mesh_dim: int
    mesh_up_dim: int | None
    mesh_down_dim: int | None

    @staticmethod
    def from_graph(graph: GraphEdgesAndFeatures) -> "GraphSizes":
        mesh_static_list = _as_list(graph.mesh_static_features)
        m2m_edge_list = _as_list(graph.m2m_edge_index)
        m2m_feat_list = _as_list(graph.m2m_features)
        mesh_up_edge_list = _as_list(graph.mesh_up_edge_index)
        mesh_down_edge_list = _as_list(graph.mesh_down_edge_index)
        mesh_up_feat_list = _as_list(graph.mesh_up_features)
        mesh_down_feat_list = _as_list(graph.mesh_down_features)

        mesh_level_sizes = [feat.shape[0] for feat in mesh_static_list]
        num_mesh_nodes = sum(mesh_level_sizes)
        num_mesh_nodes_ignore = num_mesh_nodes - mesh_level_sizes[0]

        m2m_edge_counts = [edge.shape[1] for edge in m2m_edge_list]
        mesh_up_edge_counts = [edge.shape[1] for edge in mesh_up_edge_list]
        mesh_down_edge_counts = [edge.shape[1] for edge in mesh_down_edge_list]
        edge_split_sections = (
            m2m_edge_counts + mesh_up_edge_counts + mesh_down_edge_counts
        )

        mesh_up_dim = (
            mesh_up_feat_list[0].shape[1] if mesh_up_feat_list else None
        )
        mesh_down_dim = (
            mesh_down_feat_list[0].shape[1] if mesh_down_feat_list else None
        )

        return GraphSizes(
            hierarchical=graph.hierarchical,
            num_levels=len(mesh_static_list),
            mesh_level_sizes=mesh_level_sizes,
            num_mesh_nodes=num_mesh_nodes,
            num_mesh_nodes_ignore=num_mesh_nodes_ignore,
            g2m_edges=graph.g2m_edge_index.shape[1],
            m2g_edges=graph.m2g_edge_index.shape[1],
            m2m_edge_counts=m2m_edge_counts,
            mesh_up_edge_counts=mesh_up_edge_counts,
            mesh_down_edge_counts=mesh_down_edge_counts,
            edge_split_sections=edge_split_sections,
            g2m_dim=graph.g2m_features.shape[1],
            m2g_dim=graph.m2g_features.shape[1],
            m2m_dim=m2m_feat_list[0].shape[1],
            mesh_dim=mesh_static_list[0].shape[1],
            mesh_up_dim=mesh_up_dim,
            mesh_down_dim=mesh_down_dim,
        )


def load_graph(graph_dir_path, device="cpu") -> GraphEdgesAndFeatures:
    """Load all tensors representing the graph from `graph_dir_path`."""

    graph_dir_path = Path(graph_dir_path)

    def loads_file(fn):
        return torch.load(
            graph_dir_path / fn,
            map_location=device,
            weights_only=True,
        )

    # Load edges (edge_index)
    m2m_edge_index = BufferList(
        loads_file("m2m_edge_index.pt"), persistent=False
    )  # List of (2, M_m2m[l])
    g2m_edge_index = loads_file("g2m_edge_index.pt")  # (2, M_g2m)
    m2g_edge_index = loads_file("m2g_edge_index.pt")  # (2, M_m2g)

    n_levels = len(m2m_edge_index)
    hierarchical = n_levels > 1  # Not just single level mesh graph

    # Load static edge features
    # List of (M_m2m[l], d_edge_f)
    m2m_features = loads_file("m2m_features.pt")
    g2m_features = loads_file("g2m_features.pt")  # (M_g2m, d_edge_f)
    m2g_features = loads_file("m2g_features.pt")  # (M_m2g, d_edge_f)

    # Normalize by dividing with longest edge (found in m2m)
    longest_edge = max(
        torch.max(level_features[:, 0]) for level_features in m2m_features
    )  # Col. 0 is length
    m2m_features = BufferList(
        [level_features / longest_edge for level_features in m2m_features],
        persistent=False,
    )
    g2m_features = g2m_features / longest_edge
    m2g_features = m2g_features / longest_edge

    # Load static node features
    mesh_static_features = loads_file(
        "mesh_features.pt"
    )  # List of (N_mesh[l], d_mesh_static)

    # Some checks for consistency
    assert (
        len(m2m_features) == n_levels
    ), "Inconsistent number of levels in mesh"
    assert (
        len(mesh_static_features) == n_levels
    ), "Inconsistent number of levels in mesh"

    mesh_up_edge_index: Sequence[torch.Tensor] | BufferList
    mesh_down_edge_index: Sequence[torch.Tensor] | BufferList
    mesh_up_features: Sequence[torch.Tensor] | BufferList
    mesh_down_features: Sequence[torch.Tensor] | BufferList

    if hierarchical:
        # Load up and down edges and features
        mesh_up_edge_index = BufferList(
            loads_file("mesh_up_edge_index.pt"), persistent=False
        )  # List of (2, M_up[l])
        mesh_down_edge_index = BufferList(
            loads_file("mesh_down_edge_index.pt"), persistent=False
        )  # List of (2, M_down[l])

        mesh_up_features = loads_file(
            "mesh_up_features.pt"
        )  # List of (M_up[l], d_edge_f)
        mesh_down_features = loads_file(
            "mesh_down_features.pt"
        )  # List of (M_down[l], d_edge_f)

        # Rescale
        mesh_up_features = BufferList(
            [
                edge_features / longest_edge
                for edge_features in mesh_up_features
            ],
            persistent=False,
        )
        mesh_down_features = BufferList(
            [
                edge_features / longest_edge
                for edge_features in mesh_down_features
            ],
            persistent=False,
        )

        mesh_static_features = BufferList(
            mesh_static_features, persistent=False
        )
    else:
        # Extract single mesh level
        m2m_edge_index = m2m_edge_index[0]
        m2m_features = m2m_features[0]
        mesh_static_features = mesh_static_features[0]

        mesh_up_edge_index = []
        mesh_down_edge_index = []
        mesh_up_features = []
        mesh_down_features = []

    return GraphEdgesAndFeatures(
        hierarchical=hierarchical,
        g2m_edge_index=g2m_edge_index,
        m2g_edge_index=m2g_edge_index,
        m2m_edge_index=m2m_edge_index,
        mesh_up_edge_index=mesh_up_edge_index,
        mesh_down_edge_index=mesh_down_edge_index,
        g2m_features=g2m_features,
        m2g_features=m2g_features,
        m2m_features=m2m_features,
        mesh_up_features=mesh_up_features,
        mesh_down_features=mesh_down_features,
        mesh_static_features=mesh_static_features,
    )
