import torch
import os


def save_edges(graph, name, base_path):
    torch.save(
        graph.edge_index, os.path.join(base_path, f"{name}_edge_index.pt")
    )
    edge_features = torch.cat((graph.len.unsqueeze(1), graph.vdiff), dim=1).to(
        torch.float32
    )  # Save as float32
    torch.save(edge_features, os.path.join(base_path, f"{name}_features.pt"))


def save_edges_list(graphs, name, base_path):
    torch.save(
        [graph.edge_index for graph in graphs],
        os.path.join(base_path, f"{name}_edge_index.pt"),
    )
    edge_features = [
        torch.cat((graph.len.unsqueeze(1), graph.vdiff), dim=1).to(
            torch.float32
        )
        for graph in graphs
    ]  # Save as float32
    torch.save(edge_features, os.path.join(base_path, f"{name}_features.pt"))
