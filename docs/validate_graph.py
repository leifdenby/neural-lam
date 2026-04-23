"""
Standalone CLI validator for neural-lam on-disk graph directories.
The specification text is embedded directly within the check functions.

Run with:
    python docs/validate_graph.py --graph_dir <path-to-graph-dir>
"""

# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy>=1.24.2",
#   "torch>=2.3.0",
# ]
# ///

# Standard library
import json
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Third-party
import torch

# Constants based on Spec
MESH_FEATURE_DIM = 2
EDGE_INDEX_DTYPE = torch.int64
FEATURE_DTYPE = torch.float32


@dataclass
class GraphValidationReport:
    graph_dir: str
    hierarchical: bool
    num_levels: int
    num_mesh_nodes_per_level: list[int]
    num_mesh_nodes_total: int
    num_grid_nodes: int
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0


def _load_pt(path: Path) -> Any:
    return torch.load(path, map_location="cpu", weights_only=True)


def _check_edge_index(
    *,
    name: str,
    edge_index: torch.Tensor,
    errors: list[str],
    expected_sender_range: tuple[int, int] | None = None,
    expected_receiver_range: tuple[int, int] | None = None,
) -> int:
    """
    ### Edge indices
    - Row `0` MUST be sender node index, row `1` MUST be receiver node index.
    - Dtype MUST be `torch.int64`.
    - Shape MUST be `[2, E]`, where `E` is the number of edges.
    """
    if not isinstance(edge_index, torch.Tensor):
        errors.append(f"{name}: Expected torch.Tensor, got {type(edge_index)}")
        return 0

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        errors.append(f"{name}: Expected shape [2, E], got {tuple(edge_index.shape)}")
        return 0

    if edge_index.dtype != EDGE_INDEX_DTYPE:
        errors.append(f"{name}: Expected dtype {EDGE_INDEX_DTYPE}, got {edge_index.dtype}")

    if edge_index.shape[1] == 0:
        errors.append(f"{name}: Contains zero edges")
        return 0

    if expected_sender_range is not None:
        s_min, s_max = expected_sender_range
        if edge_index[0].min() < s_min or edge_index[0].max() >= s_max:
            errors.append(f"{name}: Senders out of range [{s_min}, {s_max})")

    if expected_receiver_range is not None:
        r_min, r_max = expected_receiver_range
        if edge_index[1].min() < r_min or edge_index[1].max() >= r_max:
            errors.append(f"{name}: Receivers out of range [{r_min}, {r_max})")

    return edge_index.shape[1]


def _check_edge_features(
    *,
    name: str,
    features: torch.Tensor,
    expected_num_edges: int,
    errors: list[str],
) -> None:
    """
    ### Edge features
    - The shape MUST be `[E_component, N_f]`.
    - `N_f` MUST be consistent across all edge feature tensors in the graph.
    - Dtype MUST be `torch.float32`.
    """
    if not isinstance(features, torch.Tensor):
        errors.append(f"{name}: Expected torch.Tensor, got {type(features)}")
        return

    if features.ndim != 2:
        errors.append(f"{name}: Expected shape [E, N_f], got {features.shape}")
        return

    if features.shape[0] != expected_num_edges:
        errors.append(f"{name}: Row count ({features.shape[0]}) must match edge count ({expected_num_edges})")

    if features.dtype != FEATURE_DTYPE:
        errors.append(f"{name}: Expected dtype {FEATURE_DTYPE}, got {features.dtype}")


def _check_mesh_features(
    *,
    name: str,
    mesh_features: torch.Tensor,
    errors: list[str],
    warnings: list[str],
) -> None:
    """
    ### Mesh node features
    - `mesh_features` entries MUST have shape `[N_level, 2]`.
    - Columns MUST be x/y coordinates.
    - Dtype MUST be `torch.float32`.
    """
    if not isinstance(mesh_features, torch.Tensor):
        errors.append(f"{name}: Expected torch.Tensor, got {type(mesh_features)}")
        return

    if mesh_features.ndim != 2 or mesh_features.shape[1] != MESH_FEATURE_DIM:
        errors.append(f"{name}: Expected shape [N, {MESH_FEATURE_DIM}], got {mesh_features.shape}")
        return

    if mesh_features.dtype != FEATURE_DTYPE:
        errors.append(f"{name}: Expected dtype {FEATURE_DTYPE}, got {mesh_features.dtype}")

    # Warning for normalization (Global coordinates will exceed 1.0)
    max_val = torch.abs(mesh_features).max().item()
    if max_val > 1.001:
        warnings.append(f"{name}: Max coordinate {max_val:.2f} > 1.0 (Typical for global/unnormalized graphs)")


def validate_graph_directory(graph_dir_path: str | Path) -> GraphValidationReport:
    graph_dir = Path(graph_dir_path)
    errors, warnings = [], []

    required_files = [
        "m2m_edge_index.pt", "g2m_edge_index.pt", "m2g_edge_index.pt",
        "m2m_features.pt", "g2m_features.pt", "m2g_features.pt", "mesh_features.pt"
    ]

    # Collect all missing files before failing
    missing = [f for f in required_files if not (graph_dir / f).exists()]
    for f in missing:
        errors.append(f"Missing required file: {f}")

    if errors:
        return GraphValidationReport(str(graph_dir), False, 0, [], 0, 0, errors, warnings)

    # Load all core components
    m2m_edge_index = _load_pt(graph_dir / "m2m_edge_index.pt")
    m2m_features = _load_pt(graph_dir / "m2m_features.pt")
    mesh_features = _load_pt(graph_dir / "mesh_features.pt")
    g2m_edge_index = _load_pt(graph_dir / "g2m_edge_index.pt")
    m2g_edge_index = _load_pt(graph_dir / "m2g_edge_index.pt")
    g2m_features = _load_pt(graph_dir / "g2m_features.pt")
    m2g_features = _load_pt(graph_dir / "m2g_features.pt")

    # Validate List Structures (Hierarchical support)
    for name, obj in [("m2m_edge_index", m2m_edge_index), ("m2m_features", m2m_features), ("mesh_features", mesh_features)]:
        if not isinstance(obj, list):
            errors.append(f"{name}.pt: MUST contain a list of tensors")

    if errors:
        return GraphValidationReport(str(graph_dir), False, 0, [], 0, 0, errors, warnings)

    L = len(mesh_features)
    hierarchical = L > 1
    
    # Check Mesh Levels
    mesh_nodes_per_level = []
    total_mesh_nodes = 0
    level_offsets = []

    for i in range(L):
        _check_mesh_features(name=f"mesh_features[{i}]", mesh_features=mesh_features[i], errors=errors, warnings=warnings)
        count = mesh_features[i].shape[0] if hasattr(mesh_features[i], 'shape') else 0
        mesh_nodes_per_level.append(count)
        level_offsets.append(total_mesh_nodes)
        total_mesh_nodes += count

    # Check M2M Edges
    for i in range(L):
        n_edges = _check_edge_index(
            name=f"m2m_edge_index[{i}]", 
            edge_index=m2m_edge_index[i], 
            errors=errors,
            expected_sender_range=(level_offsets[i], level_offsets[i] + mesh_nodes_per_level[i]),
            expected_receiver_range=(level_offsets[i], level_offsets[i] + mesh_nodes_per_level[i])
        )
        _check_edge_features(name=f"m2m_features[{i}]", features=m2m_features[i], expected_num_edges=n_edges, errors=errors)

    # Hierarchical Up/Down
    if hierarchical:
        up_files = ["mesh_up_edge_index.pt", "mesh_up_features.pt", "mesh_down_edge_index.pt", "mesh_down_features.pt"]
        for f in up_files:
            if not (graph_dir / f).exists():
                errors.append(f"Missing hierarchical file: {f}")
        
        if not errors:
            up_idx = _load_pt(graph_dir / "mesh_up_edge_index.pt")
            up_feat = _load_pt(graph_dir / "mesh_up_features.pt")
            down_idx = _load_pt(graph_dir / "mesh_down_edge_index.pt")
            down_feat = _load_pt(graph_dir / "mesh_down_features.pt")

            for i in range(L - 1):
                n_up = _check_edge_index(
                    name=f"mesh_up_edge_index[{i}]", edge_index=up_idx[i], errors=errors,
                    expected_sender_range=(level_offsets[i], level_offsets[i] + mesh_nodes_per_level[i]),
                    expected_receiver_range=(level_offsets[i+1], level_offsets[i+1] + mesh_nodes_per_level[i+1])
                )
                _check_edge_features(name=f"mesh_up_features[{i}]", features=up_feat[i], expected_num_edges=n_up, errors=errors)

    # G2M and M2G (Infers Grid Nodes)
    n_g2m = _check_edge_index(name="g2m_edge_index", edge_index=g2m_edge_index, errors=errors)
    _check_edge_features(name="g2m_features", features=g2m_features, expected_num_edges=n_g2m, errors=errors)
    
    n_m2g = _check_edge_index(name="m2g_edge_index", edge_index=m2g_edge_index, errors=errors)
    _check_edge_features(name="m2g_features", features=m2g_features, expected_num_edges=n_m2g, errors=errors)

    # Validate Index Space Logic: "Mesh nodes MUST come first. Grid nodes MUST follow after all mesh nodes."
    num_grid_nodes = 0
    if m2g_edge_index.ndim == 2:
        m2g_rec_min = m2g_edge_index[1].min().item()
        if m2g_rec_min < total_mesh_nodes:
            errors.append(f"m2g_edge_index: Receiver indices MUST start after mesh nodes (>= {total_mesh_nodes})")
        
        num_grid_nodes = int(m2g_edge_index[1].max().item() - total_mesh_nodes + 1)

    return GraphValidationReport(
        str(graph_dir), hierarchical, L, mesh_nodes_per_level, total_mesh_nodes, max(0, num_grid_nodes), errors, warnings
    )

def cli():
    parser = ArgumentParser(description="Validate Neural-LAM graph storage", formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--graph_dir", type=str, required=True, help="Path to graph directory")
    args = parser.parse_args()

    report = validate_graph_directory(args.graph_dir)
    print(f"\n--- Validation Report: {report.graph_dir} ---")
    print(f"Status: {'PASS' if report.ok else 'FAIL'}")
    print(f"Mesh Levels: {report.num_levels} | Total Mesh Nodes: {report.num_mesh_nodes_total} | Grid Nodes: {report.num_grid_nodes}")

    if report.errors:
        print("\nERRORS:")
        for e in report.errors: print(f"  - {e}")
    if report.warnings:
        print("\nWARNINGS:")
        for w in report.warnings: print(f"  - {w}")

if __name__ == "__main__":
    cli()