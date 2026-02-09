# Third-party
import torch
from torch import nn

# Local
from ..config import NeuralLAMConfig
from ..datastore import BaseDatastore
from ..graph_data import GraphEdgesAndFeatures, GraphSizes
from ..interaction_net import InteractionNet
from .base_hi_graph_model import BaseHiGraphModel


class HiLAMParallel(BaseHiGraphModel):
    """
    Version of HiLAM where all message passing in the hierarchical mesh (up,
    down, inter-level) is ran in parallel.

    This is a somewhat simpler alternative to the sequential message passing
    of Hi-LAM.
    """

    def __init__(
        self,
        args,
        config: NeuralLAMConfig,
        datastore: BaseDatastore,
        graph: GraphEdgesAndFeatures,
        graph_sizes: GraphSizes,
    ):
        super().__init__(
            args,
            config=config,
            datastore=datastore,
            graph=graph,
            graph_sizes=graph_sizes,
        )

        # Processor GNNs
        # Create the complete edge_index combining all edges for processing
        total_edge_index_list = (
            list(self.m2m_edge_index)
            + list(self.mesh_up_edge_index)
            + list(self.mesh_down_edge_index)
        )
        total_edge_index = torch.cat(total_edge_index_list, dim=1)
        self.register_buffer(
            "total_edge_index", total_edge_index, persistent=False
        )
        self.edge_split_sections = graph_sizes.edge_split_sections

        if args.processor_layers == 0:
            self.processor_nets = nn.ModuleList()
        else:
            processor_nets = [
                InteractionNet(
                    args.hidden_dim,
                    hidden_layers=args.hidden_layers,
                    edge_chunk_sizes=self.edge_split_sections,
                    aggr_chunk_sizes=self.level_mesh_sizes,
                )
                for _ in range(args.processor_layers)
            ]
            self.processor_nets = nn.ModuleList(processor_nets)

    def hi_processor_step(
        self, mesh_rep_levels, mesh_same_rep, mesh_up_rep, mesh_down_rep
    ):
        """
        Internal processor step of hierarchical graph models.
        Between mesh init and read out.

        Each input is list with representations, each with shape

        mesh_rep_levels: (B, N_mesh[l], d_h)
        mesh_same_rep: (B, M_same[l], d_h)
        mesh_up_rep: (B, M_up[l -> l+1], d_h)
        mesh_down_rep: (B, M_down[l <- l+1], d_h)

        Returns same lists
        """

        # First join all node and edge representations to single tensors
        mesh_rep = torch.cat(mesh_rep_levels, dim=1)  # (B, N_mesh, d_h)
        mesh_edge_rep = torch.cat(
            mesh_same_rep + mesh_up_rep + mesh_down_rep, axis=1
        )  # (B, M_mesh, d_h)

        # Here, update mesh_*_rep and mesh_rep
        for net in self.processor_nets:
            mesh_rep, mesh_edge_rep = net(
                mesh_rep,
                mesh_rep,
                mesh_edge_rep,
                edge_index=self.total_edge_index,
            )

        # Split up again for read-out step
        mesh_rep_levels = list(
            torch.split(mesh_rep, self.level_mesh_sizes, dim=1)
        )
        mesh_edge_rep_sections = torch.split(
            mesh_edge_rep, self.edge_split_sections, dim=1
        )

        mesh_same_rep = mesh_edge_rep_sections[: self.num_levels]
        mesh_up_rep = mesh_edge_rep_sections[
            self.num_levels : self.num_levels + (self.num_levels - 1)
        ]
        mesh_down_rep = mesh_edge_rep_sections[
            self.num_levels + (self.num_levels - 1) :
        ]  # Last are down edges

        # TODO: We return all, even though only down edges really are used
        # later
        return mesh_rep_levels, mesh_same_rep, mesh_up_rep, mesh_down_rep
