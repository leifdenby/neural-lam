# Third-party
from torch import nn


class EncodeProcessDecodeGraph:
    def __init__(self):
        self._embedding_blueprints = {}
        self._embedding_networks = {}

    def _register_embedder(
        self, identifier, n_features_in, n_features_out, kind
    ):
        self._embedding_blueprints[identifier] = dict(
            n_features_in=n_features_in,
            n_features_out=n_features_out,
            kind=kind,
        )

    def _construct_embedders(self):
        for identifier, blueprint in self._embedding_blueprints.items():
            n_in = blueprint["n_features_in"]
            n_out = blueprint["n_features_out"]
            if blueprint["kind"] == "linear_single":
                self._embedding_networks[identifier] = nn.Linear(n_in, n_out)
            else:
                raise ValueError(f"Unknown kind: {blueprint['kind']}")


class KeislerGraph(EncodeProcessDecodeGraph):
    def __init__(
        self, hidden_dim_size=512, n_grid_features=10, n_edge_features=2
    ):
        super().__init__()

        self._register_embedder(
            identifier="grid_node",
            n_features_in=n_grid_features,
            n_features_out=hidden_dim_size,
            kind="linear_single",
        )
        self._register_embedder(
            identifier="g2m_and_m2g_edge",
            n_features_in=n_edge_features,
            n_features_out=hidden_dim_size,
            kind="linear_single",
        )
        self._register_embedder(
            identifier="m2m_edge",
            n_features_in=n_edge_features,
            n_features_out=hidden_dim_size,
            kind="linear_single",
        )

        self._construct_embedders()
