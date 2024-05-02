import networkx


def prepend_node_index(graph, new_index):
    """
    Prepend node index to node tuple in graph, i.e. (i, j) -> (new_index, i, j)

    Parameters
    ----------
    graph : networkx.Graph
        Graph to relabel
    new_index : int
        New index to prepend to node tuple

    Returns
    -------
    networkx.Graph
        Graph with relabeled nodes
    """
    ijk = [tuple((new_index,) + x) for x in graph.nodes]
    to_mapping = dict(zip(graph.nodes, ijk))
    return networkx.relabel_nodes(graph, to_mapping, copy=True)


def sort_nodes_internally(nx_graph, node_attribute=None, edge_attribute=None):
    # For some reason the networkx .nodes() return list can not be sorted,
    # but this is the ordering used by pyg when converting.
    # This function fixes this.
    H = networkx.DiGraph()
    if node_attribute is not None:
        H.add_nodes_from(sorted(nx_graph.nodes(data=True), key=lambda x: x[1][node_attribute]))
    else:
        H.add_nodes_from(sorted(nx_graph.nodes(data=True)))
    
    if edge_attribute is not None:
        H.add_edges_from(sorted(nx_graph.edges(data=True), key=lambda x: x[2][edge_attribute]))
    else:
        H.add_edges_from(nx_graph.edges(data=True))
    return H